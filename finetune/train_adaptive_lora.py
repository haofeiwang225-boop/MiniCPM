"""Single-device official RaLoRA baseline and optional two-phase adaptive extension."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
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
    p.add_argument('--model_max_length', type=int, default=None)
    p.add_argument('--dynamic_length', type=boolean, default=False)
    p.add_argument('--max_steps', type=int, default=200)
    p.add_argument('--learning_rate', type=float, default=5e-5)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--target_modules', nargs='+', default=['q_proj', 'v_proj'])
    p.add_argument('--lora_structure', choices=['dense', 'block'], default='dense')
    p.add_argument('--dense_init_mode', choices=['gid', 'vanilla', 'budget_matched', 'shuffled_gid', 'replay_gid'], default='gid')
    p.add_argument('--rank_manifest', type=Path)
    p.add_argument('--dense_lora_param_budget', type=int)
    p.add_argument('--dense_rank_min', type=int, default=1)
    p.add_argument('--dense_rank_max', type=int, default=64)
    p.add_argument('--lora_rank', type=int, default=64)
    p.add_argument('--lora_alpha', type=float, default=32)
    p.add_argument('--lora_dropout', type=float, default=0.1)
    p.add_argument('--weight_a_init_method', default='kaiming', choices=['kaiming', 'normal', 'orthogonal', 'xavier'])
    p.add_argument('--gradient_est_n_steps', type=int, default=8)
    p.add_argument('--initial_probe_only', action='store_true', help='Save initialized adapter and exit before optimizer creation')
    p.add_argument('--ralora_min_rank', type=int, default=1)
    p.add_argument('--ralora_max_rank', type=int, default=64)
    p.add_argument('--ralora_erank_max_power', type=int, default=4)
    p.add_argument('--ralora_forward_method', choices=['for', 'concat', 'einsum'], default='concat')
    p.add_argument('--ralora_dynamic_scaling', type=boolean, default=False)
    p.add_argument('--ralora_rank_stablize', type=boolean, default=False)
    p.add_argument('--adaptive_lora', type=boolean, default=False)
    p.add_argument('--adaptive_probe_step', type=int, default=100)
    p.add_argument('--adaptive_probe_batches', type=int, default=8)
    p.add_argument('--adaptive_direction_efficiency_threshold', type=float, default=0.8)
    p.add_argument('--adaptive_oracle_coverage_threshold', type=float, default=0.9)
    p.add_argument('--adaptive_residual_energy_target', type=float, default=0.90)
    p.add_argument('--adaptive_rank_init', choices=['zero_b_residual_v', 'symmetric_residual'],
                   default='zero_b_residual_v')
    p.add_argument('--save_diagnostic_checkpoints', type=boolean, default=True)
    p.add_argument('--diagnostic_steps', nargs='+', type=int, default=[],
                   help='Read-only probes during one continuous warm-up, without optimizer reset')
    p.add_argument('--adaptive_residual_energy_threshold', type=float, default=0.1)
    p.add_argument('--adaptive_rank_max', type=int, default=256)
    p.add_argument('--adaptive_rank_step_cap', type=int, default=4)
    p.add_argument('--residual_init_scale', type=float, default=0.001)
    p.add_argument('--direction_correction', choices=['none', 'blend', 'geometric', 'reset_only', 'geometric_v2'], default='none')
    p.add_argument('--rsc_selection_samples', type=int, default=8)
    p.add_argument('--rsc_selection_indices', nargs='+', type=int)
    p.add_argument('--rsc_proposal_descent_margin', type=float, default=0.)
    p.add_argument('--rsc_selection_descent_margin', type=float, default=0.)
    p.add_argument('--rsc_loss_min_improvement', type=float, default=0.)
    p.add_argument('--rsc_trial_strengths', nargs='+', type=float, default=[1., .5, .25])
    p.add_argument('--reset_side_manifest', type=Path)
    p.add_argument('--direction_correction_steps', nargs='+', type=int, default=[])
    p.add_argument('--direction_function_tolerance', type=float, default=0.05)
    p.add_argument('--direction_min_gain', type=float, default=1e-3)
    p.add_argument('--direction_eta_start', type=float, default=1.0)
    p.add_argument('--direction_eta_decay', type=float, default=0.5)
    p.add_argument('--direction_eta_min', type=float, default=0.03125)
    p.add_argument('--direction_blend', type=float, default=0.05)
    p.add_argument('--adaptive_stats_dir', type=Path)
    p.add_argument('--gradient_checkpointing', type=boolean, default=True)
    p.add_argument('--export_merged', type=boolean, default=False)
    p.add_argument('--tiny_model', action='store_true', help='Synthetic tiny Llama training smoke only')
    args = p.parse_args()
    if args.dense_init_mode != 'gid':
        if args.lora_structure != 'dense' or args.adaptive_lora or args.direction_correction != 'none' or args.diagnostic_steps:
            p.error('Rank controls require dense, adaptive_lora=false, no correction or diagnostic probes')
        if args.dense_init_mode != 'vanilla' and args.dense_lora_param_budget is None:
            p.error('Rank control requires explicit actual parameter budget')
        if args.dense_init_mode in ('shuffled_gid', 'replay_gid') and (args.rank_manifest is None or not args.rank_manifest.is_file()):
            p.error('Shuffled/replay GID requires an existing rank_manifest')
        if args.dense_init_mode in ('vanilla', 'budget_matched') and args.rank_manifest is not None:
            p.error('Vanilla/budget control must not read a GID manifest')
    if args.model_max_length is None:
        args.model_max_length = 2048 if args.dynamic_length else 256
    if args.dynamic_length and args.batch_size != 1:
        p.error('Dynamic length with default_data_collator requires batch_size=1')
    if args.dynamic_length and args.model_family != 'minicpm5' and not args.tiny_model:
        p.error('Dynamic chat length requires model_family=minicpm5')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        p.error('Phase-boundary reconstruction currently supports one device only')
    if args.lora_structure == 'dense' and args.gradient_est_n_steps != args.adaptive_probe_batches:
        p.error('Dense initial and periodic probes must use the same calibration batch count')
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
    if any(not 0 <= x <= 1 for x in (args.adaptive_direction_efficiency_threshold,
            args.adaptive_oracle_coverage_threshold, args.adaptive_residual_energy_threshold)):
        p.error('Thresholds must be in [0,1]')
    if not 0 < args.adaptive_residual_energy_target <= 1:
        p.error('Residual energy target must be in (0,1]')
    if args.diagnostic_steps and (args.adaptive_lora or args.direction_correction == 'blend'):
        p.error('Diagnostic steps require adaptive_lora=false and none/geometric/reset_only correction')
    if any(not 0 < step <= args.max_steps for step in args.diagnostic_steps):
        p.error('Diagnostic steps must lie inside training')
    if args.dense_rank_min < 1 or args.dense_rank_max < args.dense_rank_min:
        p.error('Invalid Dense rank bounds')
    if args.dense_lora_param_budget is not None and args.dense_lora_param_budget < 1:
        p.error('Dense parameter budget must be positive')
    if args.direction_correction in ('geometric', 'reset_only', 'geometric_v2'):
        if args.lora_structure != 'dense' or args.adaptive_lora:
            p.error('Geometric/reset_only mode requires Dense LoRA and adaptive_lora=false; expansion is forbidden')
        if not args.direction_correction_steps:
            p.error('Geometric/reset_only mode requires explicit direction_correction_steps')
    elif args.direction_correction_steps:
        p.error('Correction steps require direction_correction=geometric/reset_only')
    if args.reset_side_manifest is not None:
        if args.direction_correction != 'reset_only':
            p.error('reset_side_manifest requires direction_correction=reset_only')
        if not args.reset_side_manifest.is_file():
            p.error('reset_side_manifest must be an existing file')
    steps = args.direction_correction_steps
    if steps != sorted(set(steps)) or any(not 0 < t < args.max_steps for t in steps):
        p.error('Correction steps must be unique, increasing, and inside training')
    if set(steps) & set(args.diagnostic_steps):
        p.error('Correction and diagnostic steps must not overlap')
    from adaptive_lora.direction_correction import _eta_grid
    try:
        _eta_grid(args.direction_eta_start,args.direction_eta_decay,args.direction_eta_min)
    except ValueError as error:
        p.error(str(error))
    import math
    if not math.isfinite(args.direction_function_tolerance) or args.direction_function_tolerance < 0:
        p.error('Invalid LoRA-update trust region')
    if not math.isfinite(args.direction_min_gain) or args.direction_min_gain <= 0:
        p.error('direction_min_gain must be positive and finite')
    if args.direction_correction == 'geometric_v2':
        if args.batch_size != 1 or args.rsc_selection_samples < 1:
            p.error('RSC requires batch_size=1 and positive selection sample count')
        for name in ('rsc_proposal_descent_margin', 'rsc_selection_descent_margin', 'rsc_loss_min_improvement'):
            if not math.isfinite(getattr(args,name)) or getattr(args,name) < 0:
                p.error(name + ' must be finite and nonnegative')
        strengths = args.rsc_trial_strengths
        if not strengths or len(strengths)>8 or any(not math.isfinite(x) or not 0<x<=1 for x in strengths) or any(a<=b for a,b in zip(strengths,strengths[1:])):
            p.error('RSC trial strengths must be 1..8 strictly decreasing values in (0,1]')
        from adaptive_lora.rsc_calibration import training_source
        try:
            training_source(args.train_data_path, args.tiny_model)
        except (ValueError, OSError) as error:
            p.error(str(error))
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
    started = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
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
            dataset = ChatTrainingDataset(args.train_data_path, tokenizer, args.model_max_length,
                                          dynamic_length=args.dynamic_length)
        else:
            dataset = SupervisedDataset(str(args.train_data_path), tokenizer, args.model_max_length)
    model.to(args.device)
    original_use_cache = model.config.use_cache
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    calibration = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=default_data_collator)
    iterator = cycle_batches(calibration)
    batches = [next(iterator) for _ in range(args.gradient_est_n_steps)]
    from adaptive_lora.calibration import write_calibration_indices, load_calibration_batches
    if args.lora_structure == 'dense' or args.adaptive_lora or args.diagnostic_steps:
        args.calibration_indices_path = args.output_dir/'calibration_indices.json'
        args.calibration_indices_sha256 = write_calibration_indices(
            args.calibration_indices_path, dataset, args.batch_size, args.adaptive_probe_batches,
            'synthetic_training' if args.tiny_model else args.train_data_path.resolve())

    def checkpoint_batches():
        return load_calibration_batches(args.calibration_indices_path,
            args.calibration_indices_sha256, dataset, default_data_collator)

    if args.direction_correction == 'geometric_v2':
        from adaptive_lora.rsc_calibration import write_rsc_manifest
        args.rsc_manifest_path = args.output_dir/'rsc_training_samples.json'
        args.rsc_manifest_sha256 = write_rsc_manifest(args.rsc_manifest_path, dataset, args)

    from adaptive_lora.dense_init import attach_dense, dense_first_probe
    names = (attach_dense(model, args) if args.lora_structure == 'dense'
             else attach_ralora(model, args))
    run_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    run_config.update(model_class=type(model).__name__, train_samples=len(dataset), targets=names,
                      python=sys.executable, torch_version=str(torch.__version__),
                      transformers_version=transformers.__version__,
                      model_type=model.config.model_type, model_config=model.config.to_dict(),
                      target_weight_shapes={n: list(model.get_submodule(n).weight.shape) for n in names},
                      baseline=('gid_dense_lora' if args.lora_structure == 'dense' else 'local_official_ralora'),
                      phase2_optimizer=('fresh_adamw_and_linear_scheduler' if args.adaptive_lora else 'not_used'),
                      diagnostic_protocol='fixed_training_calibration_restore_rng_no_optimizer_reset')
    from adaptive_lora.official import OFFICIAL
    from adaptive_lora.data_provenance import file_sha256, git_version
    run_config.update(git_version())
    run_config['train_data_sha256'] = None if args.tiny_model else file_sha256(args.train_data_path)
    run_config['official_commit'] = subprocess.check_output(['git', '-C', str(OFFICIAL), 'rev-parse', 'HEAD'], text=True).strip()
    (args.output_dir/'run_config.json').write_text(json.dumps(run_config, indent=2))
    if args.dense_init_mode != 'gid':
        from adaptive_lora.control_initialization import initialize_control
        initialize_control(model, names, args)
    elif args.lora_structure == 'dense':
        dense_first_probe(model, names, checkpoint_batches(), args)
    else:
        first_probe(model, batches[:args.gradient_est_n_steps], args)
    if args.initial_probe_only:
        save_adapter(model, names, args.output_dir/'initial_adapter.pt', args.model_name_or_path)
        print('Initial probe sanity completed; no optimizer created and no training updates.', flush=True)
        return
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
            if step in args.direction_correction_steps and args.direction_correction == 'reset_only':
                from adaptive_lora.reset_only import reset_only_step
                reset_only_step(model, names, optimizer, scheduler, step, args.output_dir,
                                side_manifest=args.reset_side_manifest)
            elif step in args.direction_correction_steps and args.direction_correction == 'geometric_v2':
                from adaptive_lora.adaptation_v2 import geometric_v2_step
                correction_dir = args.output_dir/'correction_checkpoints_v2'
                correction_dir.mkdir(exist_ok=True)
                save_adapter(model,names,correction_dir/f'step_{step}_pre.pt',args.model_name_or_path)
                geometric_v2_step(model,names,dataset,default_data_collator,args,optimizer,scheduler,step)
                save_adapter(model,names,correction_dir/f'step_{step}_post.pt',args.model_name_or_path)
            elif step in args.direction_correction_steps:
                from adaptive_lora.adaptation import geometric_correction_step
                correction_dir = args.output_dir/'correction_checkpoints'
                correction_dir.mkdir(exist_ok=True)
                save_adapter(model,names,correction_dir/f'step_{step}_pre.pt',args.model_name_or_path)
                geometric_correction_step(model,names,checkpoint_batches(),args,optimizer,scheduler,step)
                save_adapter(model,names,correction_dir/f'step_{step}_post.pt',args.model_name_or_path)
            if step in args.diagnostic_steps:
                if args.save_diagnostic_checkpoints:
                    checkpoint_dir = args.output_dir/'checkpoints'
                    checkpoint_dir.mkdir(exist_ok=True)
                    save_adapter(model, names, checkpoint_dir/f'step_{step}.pt', args.model_name_or_path)
                diagnostic_args = copy.copy(args)
                diagnostic_args.adaptive_stats_dir = args.output_dir/'diagnostics'/f'step_{step}'
                summary = adaptive_step(model, names, checkpoint_batches(),
                                        diagnostic_args, diagnostic_only=True)
                with (args.output_dir/'warmup_trajectory.jsonl').open('a') as trajectory:
                    trajectory.write(json.dumps(dict(step=step, **summary))+'\n')
            if args.adaptive_lora and step == args.adaptive_probe_step:
                save_adapter(model, names, args.output_dir/'phase1_adapter.pt', args.model_name_or_path)
                # No active optimizer is allowed across the structural change.
                del optimizer, scheduler
                adaptive_step(model, names, checkpoint_batches(), args)
                save_adapter(model, names, args.output_dir/'phase2_initial_adapter.pt', args.model_name_or_path)
                optimizer, scheduler = make_optimizer(model, args, args.max_steps-step)
    save_adapter(model, names, args.output_dir/'adapter.pt', args.model_name_or_path)
    if args.export_merged:
        model = merge_for_export(model, names)
        model.config.use_cache = original_use_cache
        model.save_pretrained(args.output_dir/'merged')
        if tokenizer is not None:
            tokenizer.save_pretrained(args.output_dir/'merged')
    metrics = dict(elapsed_seconds=time.perf_counter()-started, steps=args.max_steps,
        actual_lora_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
        peak_allocated_bytes=torch.cuda.max_memory_allocated() if args.device.startswith('cuda') else None,
        peak_reserved_bytes=torch.cuda.max_memory_reserved() if args.device.startswith('cuda') else None)
    (args.output_dir/'training_resources.json').write_text(json.dumps(metrics, indent=2))
    print(f'Completed {args.max_steps} training updates; saved to {args.output_dir}')


if __name__ == '__main__':
    main()
