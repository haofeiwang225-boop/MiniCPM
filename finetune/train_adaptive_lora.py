"""Single-device official RaLoRA baseline and optional two-phase adaptive extension."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import transformers
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers import default_data_collator, set_seed

from adaptive_lora.official import attach_ralora, first_probe
from adaptive_lora.adaptation import adaptive_step
from adaptive_lora.checkpoint import save_adapter, merge_for_export


def boolean(value):
    if value.lower() not in ('true', 'false'):
        raise argparse.ArgumentTypeError('Expected true or false')
    return value.lower() == 'true'


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model_name_or_path', default='openbmb/MiniCPM-2B-sft-bf16')
    p.add_argument('--model_family', choices=['minicpm', 'minicpm5'], default='minicpm')
    p.add_argument('--train_data_path', type=Path, default=Path(__file__).parent/'data/AdvertiseGenChatML/train.json')
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    p.add_argument('--analysis_device', default='cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--batch_size', type=int, default=1)
    p.add_argument('--model_max_length', type=int, default=256)
    p.add_argument('--max_steps', type=int, default=200)
    p.add_argument('--learning_rate', type=float, default=5e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--target_modules', nargs='+', default=['q_proj', 'v_proj'])
    p.add_argument('--lora_rank', type=int, default=64)
    p.add_argument('--lora_alpha', type=float, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.1)
    p.add_argument('--weight_a_init_method', default='kaiming', choices=['kaiming', 'normal', 'orthogonal', 'xavier'])
    p.add_argument('--gradient_est_n_steps', type=int, default=2)
    p.add_argument('--ralora_min_rank', type=int, default=1)
    p.add_argument('--ralora_max_rank', type=int, default=64)
    p.add_argument('--ralora_erank_max_power', type=int, default=4)
    p.add_argument('--ralora_forward_method', choices=['for', 'concat', 'einsum'], default='concat')
    p.add_argument('--ralora_dynamic_scaling', type=boolean, default=False)
    p.add_argument('--ralora_rank_stablize', type=boolean, default=False)
    p.add_argument('--adaptive_lora', type=boolean, default=False)
    p.add_argument('--adaptive_probe_step', type=int, default=100)
    p.add_argument('--adaptive_probe_batches', type=int, default=2)
    p.add_argument('--adaptive_alignment_threshold', type=float, default=0.6)
    p.add_argument('--adaptive_residual_energy_threshold', type=float, default=0.1)
    p.add_argument('--adaptive_rank_max', type=int, default=256)
    p.add_argument('--adaptive_rank_step_cap', type=int, default=4)
    p.add_argument('--residual_init_scale', type=float, default=0.001)
    p.add_argument('--direction_correction', choices=['none', 'blend'], default='none')
    p.add_argument('--direction_blend', type=float, default=0.05)
    p.add_argument('--adaptive_stats_dir', type=Path)
    p.add_argument('--gradient_checkpointing', type=boolean, default=True)
    p.add_argument('--export_merged', type=boolean, default=False)
    p.add_argument('--tiny_model', action='store_true', help='Synthetic tiny Llama training smoke only')
    args = p.parse_args()
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        p.error('Phase-boundary reconstruction currently supports one device only')
    if args.adaptive_lora and not 0 < args.adaptive_probe_step < args.max_steps:
        p.error('Probe step must lie strictly inside training')
    for name in ('max_steps', 'batch_size', 'gradient_est_n_steps', 'adaptive_probe_batches',
                 'lora_rank', 'ralora_min_rank', 'ralora_max_rank', 'adaptive_rank_max', 'adaptive_rank_step_cap'):
        if getattr(args, name) < 1:
            p.error(f'{name} must be positive')
    if args.ralora_min_rank > args.ralora_max_rank or args.ralora_erank_max_power < 0:
        p.error('Invalid initial rank/split limits')
    if args.residual_init_scale < 0 or not 0 <= args.direction_blend <= 1:
        p.error('Invalid initialization/correction scale')
    if not 0 <= args.adaptive_alignment_threshold <= 1 or not 0 <= args.adaptive_residual_energy_threshold <= 1:
        p.error('Thresholds must be in [0,1]')
    args.adaptive_stats_dir = args.adaptive_stats_dir or args.output_dir/'adaptive_stats'
    return args


class TinyTrainingData(Dataset):
    def __init__(self):
        rng = torch.Generator().manual_seed(123)
        self.tokens = torch.randint(0, 64, (8, 16), generator=rng)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        x = self.tokens[i]
        return {'input_ids': x, 'labels': x.clone(), 'attention_mask': torch.ones_like(x)}


def make_optimizer(model, args, steps):
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: max(0, 1-step/steps))
    return optimizer, scheduler


def cycle_batches(loader):
    while True:
        yield from loader


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('LOG_FOLDER', str((args.output_dir/'official_logs').resolve()))
    os.environ.setdefault('EXPERIMENT_NAME', args.output_dir.name)
    set_seed(args.seed)
    tokenizer = None
    if args.tiny_model:
        model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                                num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2))
        dataset = TinyTrainingData()
    else:
        # Model-family selection is explicit; never relabel MiniCPM5 as MiniCPM-2B.
        from finetune import SupervisedDataset
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True, local_files_only=True)
        if tokenizer.chat_template is None:
            template = Path(args.model_name_or_path)/'chat_template.jinja'
            if not template.is_file():
                raise ValueError('No official chat template found')
            tokenizer.chat_template = template.read_text()
        tokenizer.pad_token = tokenizer.eos_token
        from adaptive_lora.model_loading import load_training_model
        model = load_training_model(args.model_name_or_path, getattr(torch, args.dtype), args.model_family)
        expected_type = 'minicpm' if args.model_family == 'minicpm' else 'llama'
        if model.config.model_type != expected_type:
            raise ValueError(f'Model type does not match explicit family {args.model_family}')
        if args.model_family == 'minicpm5':
            from adaptive_lora.training_data import ChatTrainingDataset
            dataset = ChatTrainingDataset(args.train_data_path, tokenizer, args.model_max_length)
        else:
            dataset = SupervisedDataset(str(args.train_data_path), tokenizer, args.model_max_length)
    model.to(args.device)
    original_use_cache = model.config.use_cache
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    calibration = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=default_data_collator)
    iterator = cycle_batches(calibration)
    batches = [next(iterator) for _ in range(max(args.gradient_est_n_steps, args.adaptive_probe_batches))]
    names = attach_ralora(model, args)
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run_config.update(model_class=type(model).__name__, train_samples=len(dataset), targets=names,
                      python=sys.executable, torch_version=str(torch.__version__),
                      transformers_version=transformers.__version__,
                      model_type=model.config.model_type, model_config=model.config.to_dict(),
                      target_weight_shapes={n: list(model.get_submodule(n).weight.shape) for n in names},
                      baseline='local_official_ralora', phase2_optimizer='fresh_adamw_and_linear_scheduler')
    from adaptive_lora.official import OFFICIAL
    run_config['official_commit'] = subprocess.check_output(['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip()
    (args.output_dir/'run_config.json').write_text(json.dumps(run_config, indent=2))
    first_probe(model, batches[:args.gradient_est_n_steps], args)
    optimizer, scheduler = make_optimizer(model, args, args.max_steps)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                    generator=torch.Generator().manual_seed(args.seed), collate_fn=default_data_collator)
    train_batches = cycle_batches(train_loader)
    with (args.output_dir/'train.jsonl').open('x') as log:
        for step in range(1, args.max_steps+1):
            model.train()
            batch = {k: v.to(args.device) for k, v in next(train_batches).items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            record = {'step': step, 'loss': float(loss.detach()), 'lr': scheduler.get_last_lr()[0]}
            log.write(json.dumps(record)+'\n')
            log.flush()
            print(record, flush=True)
            if args.adaptive_lora and step == args.adaptive_probe_step:
                save_adapter(model, names, args.output_dir/'phase1_adapter.pt', args.model_name_or_path)
                # No active optimizer is allowed across the structural change.
                del optimizer, scheduler
                adaptive_step(model, names, batches[:args.adaptive_probe_batches], args)
                save_adapter(model, names, args.output_dir/'phase2_initial_adapter.pt', args.model_name_or_path)
                optimizer, scheduler = make_optimizer(model, args, args.max_steps-step)
    save_adapter(model, names, args.output_dir/'adapter.pt', args.model_name_or_path)
    if args.export_merged:
        model = merge_for_export(model, names)
        model.config.use_cache = original_use_cache
        model.save_pretrained(args.output_dir/'merged')
        if tokenizer is not None:
            tokenizer.save_pretrained(args.output_dir/'merged')
    print(f'Completed {args.max_steps} training updates; saved to {args.output_dir}')


if __name__ == '__main__':
    main()
