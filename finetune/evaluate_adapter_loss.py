"""Independent teacher-forced dev NLL evaluation; no training calibration or optimizer."""
import argparse
import gc
import json
import math
from pathlib import Path

import torch
from transformers import AutoTokenizer, default_data_collator
from adaptive_lora.training_data import ChatTrainingDataset
from adaptive_lora.model_loading import load_training_model
from adaptive_lora.checkpoint import load_adapter
from adaptive_lora.data_provenance import file_sha256, git_version, save_source_metadata


def evaluate_dataset(model, dataset, device, max_samples=None):
    modes = [(module, module.training) for module in model.modules()]
    losses = []
    devices = sorted({p.device.index for p in model.parameters() if p.device.type == 'cuda'})
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            for index in range(len(dataset) if max_samples is None else min(len(dataset), max_samples)):
                batch = {k: v.to(device) for k, v in default_data_collator([dataset[index]]).items()}
                count = int((batch['labels'][:, 1:] != -100).sum())
                if count == 0:
                    raise ValueError(f'No effective supervised tokens at dev row {index}')
                output = model(**batch)
                # Causal-LM loss is a mean over nonignored shifted labels, not examples.
                loss = float(output.loss.double())
                if not math.isfinite(loss):
                    raise ValueError(f'Nonfinite dev loss at row {index}')
                losses.append(dict(index=index, effective_tokens=count, nll_sum=loss * count))
    finally:
        for module, mode in modes:
            module.training = mode
    tokens = sum(row['effective_tokens'] for row in losses)
    if not tokens:
        raise ValueError('Empty validation dataset')
    nll = math.fsum(row['nll_sum'] for row in losses) / tokens
    return dict(num_samples=len(losses), effective_tokens=tokens, token_weighted_nll=nll,
                ppl=math.exp(nll) if nll < 709 else None, ppl_overflow=nll >= 709), losses


def load_evaluation_model(model_path, dtype, model_family, device, adapter_path=None):
    model = load_training_model(model_path, dtype, model_family)
    model.to(device)
    if adapter_path is not None:
        load_adapter(model, adapter_path)
    model.config.use_cache = False
    return model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', default='/home/hongchang/whf/models/MiniCPM5-2B')
    p.add_argument('--model_family', choices=['minicpm', 'minicpm5'], default='minicpm5')
    p.add_argument('--dev_data', type=Path, default=Path(__file__).parent/'adaptive_lora/data/metamath_pilot_v1/dev.json')
    p.add_argument('--source_data', type=Path, default=Path('/home/hongchang/whf/datasets/MetaMathQA/MetaMathQA-395K.jsonl'))
    p.add_argument('--control_adapter', type=Path, required=True)
    p.add_argument('--geo_adapter', type=Path, required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--max_length', type=int, default=2048)
    p.add_argument('--max_samples', type=int)
    p.add_argument('--device', default='cuda')
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default='bfloat16')
    args = p.parse_args()
    if args.max_samples is not None and args.max_samples <= 0:
        p.error('max_samples must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = save_source_metadata(args.dev_data, args.source_data, args.output_dir/'dev.source_metadata.json')
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(git_version(), dev_sha256=file_sha256(args.dev_data),
                  control_adapter_sha256=file_sha256(args.control_adapter),
                  geo_adapter_sha256=file_sha256(args.geo_adapter), protocol='held_out_dev_teacher_forced_token_weighted',
                  dynamic_length=True, batch_size=1)
    (args.output_dir/'run_config.json').write_text(json.dumps(config, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=True)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (Path(args.model)/'chat_template.jinja').read_text()
    tokenizer.pad_token = tokenizer.eos_token
    dataset = ChatTrainingDataset(args.dev_data, tokenizer, args.max_length, dynamic_length=True)
    summary = {}
    for name, adapter in [('Base', None), ('Control', args.control_adapter), ('Geo', args.geo_adapter)]:
        model = load_evaluation_model(args.model, getattr(torch, args.dtype), args.model_family, args.device, adapter)
        summary[name], rows = evaluate_dataset(model, dataset, args.device, args.max_samples)
        with (args.output_dir/f'{name.lower()}.jsonl').open('x') as stream:
            for row in rows:
                stream.write(json.dumps(dict(**row, source=provenance['samples'][row['index']]))+'\n')
        print(name, summary[name], flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (args.output_dir/'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
