"""Read-only PRE-adapter diagnostic; never resume training or create an optimizer."""
import argparse
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from transformers import AutoTokenizer, default_data_collator

from adaptive_lora.adaptation_v2 import compute_rsc_candidates
from adaptive_lora.checkpoint import load_adapter
from adaptive_lora.data_provenance import file_sha256, git_version
from adaptive_lora.direction_correction_v2 import preserved_training_state
from adaptive_lora.model_loading import load_training_model
from adaptive_lora.official import LinearWithLoRA
from adaptive_lora.rsc_calibration import training_source, write_rsc_manifest, load_rsc_samples
from adaptive_lora.training_data import ChatTrainingDataset

ROOT = Path(__file__).resolve().parents[1]


def repo_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def verified_sha(path, expected, description):
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f'Missing independent expected SHA256 for {description}')
    actual = file_sha256(path)
    if actual.lower() != expected.lower():
        raise ValueError(f'{description} SHA256 mismatch: {path}')
    return actual


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_config', required=True, type=Path)
    p.add_argument('--pre_adapter', required=True, type=Path)
    p.add_argument('--expected_adapter_sha256', required=True,
                   help='Trusted PRE adapter digest from the historical record')
    p.add_argument('--output_dir', required=True, type=Path)
    p.add_argument('--device', default='cuda')
    p.add_argument('--analysis_device', default=None, help='Default: historical configuration')
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default=None,
                   help='Optional explicit base precision override; recorded separately')
    p.add_argument('--rsc_proposal_descent_margin', type=float, default=0.)
    p.add_argument('--rsc_selection_descent_margin', type=float, default=0.)
    p.add_argument('--rsc_loss_min_improvement', type=float, default=0.)
    p.add_argument('--rsc_trial_strengths', type=float, nargs='+', default=[1., .5, .25])
    return p.parse_args()


def replay(options):
    config_path = repo_path(options.run_config)
    adapter_path = repo_path(options.pre_adapter)
    out = repo_path(options.output_dir)
    config = json.loads(config_path.read_text())
    if config.get('model_family') != 'minicpm5' or config.get('tiny_model'):
        raise ValueError('Replay requires the historical MiniCPM5 training run')
    if config.get('batch_size') != 1 or config.get('lora_structure') != 'dense':
        raise ValueError('Replay requires historical batch_size=1 Dense LoRA')
    if adapter_path.name != 'step_1600_pre.pt':
        raise ValueError('Expected the historical step_1600_pre.pt, not a POST adapter')
    adapter_sha = verified_sha(adapter_path, options.expected_adapter_sha256, 'PRE adapter')
    train_path = repo_path(config['train_data_path'])
    training_source(train_path, False)
    data_sha = verified_sha(train_path, config.get('train_data_sha256'), 'training data')
    calibration_path = repo_path(config['calibration_indices_path'])
    calibration_sha = verified_sha(calibration_path, config.get('calibration_indices_sha256'), 'calibration')
    calibration = json.loads(calibration_path.read_text())
    if (calibration.get('split') != 'train' or calibration.get('batch_size') != 1
            or calibration.get('num_calibration_batches') != 8
            or calibration.get('batch_indices') != [[i] for i in range(8)]):
        raise ValueError('Historical Proposal must be exactly the original training samples 0..7')
    if repo_path(calibration['dataset_source']) != train_path:
        raise ValueError('Historical calibration points to different training data')
    model_path = repo_path(config['model_name_or_path'])
    payload = torch.load(adapter_path, map_location='cpu', weights_only=True)
    if repo_path(payload['base_model']) != model_path:
        raise ValueError('PRE adapter and run_config reference different base models')
    if set(payload['layers']) != set(config['targets']) or any(
            spec.get('structure') != 'dense' for spec in payload['layers'].values()):
        raise ValueError('Historical target modules/adapter structure do not match run_config')
    # Never reuse an old directory, even if empty. All writes stay under this new root.
    out.mkdir(parents=True, exist_ok=False)
    report = dict(mode='offline_diagnostic_only', historical_training_state_restored=False,
                  optimizer_created=False, scheduler_created=False, optimizer_state_reset_count=0,
                  historical_run_config=config, run_config_sha256=file_sha256(config_path),
                  adapter_path=str(adapter_path), adapter_sha256=adapter_sha,
                  train_data_sha256=data_sha, calibration_sha256=calibration_sha,
                  overrides={k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()},
                  **git_version())
    report['source_sha256'] = {str(path.relative_to(ROOT)): file_sha256(path) for path in (
        Path(__file__), ROOT/'finetune/adaptive_lora/adaptation_v2.py',
        ROOT/'finetune/adaptive_lora/direction_correction_v2.py',
        ROOT/'finetune/adaptive_lora/direction_correction.py',
        ROOT/'finetune/adaptive_lora/rsc_calibration.py')}
    originals, metadata, layers, rows, candidates = {}, [], {}, [], []
    model = None
    try:
        seed = config['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True, local_files_only=True)
        if tokenizer.chat_template is None:
            tokenizer.chat_template = (model_path/'chat_template.jinja').read_text()
        tokenizer.pad_token = tokenizer.eos_token
        dataset = ChatTrainingDataset(train_path, tokenizer, config['model_max_length'],
                                      dynamic_length=config['dynamic_length'])
        if len(dataset) != calibration['dataset_length'] or len(dataset) != config['train_samples']:
            raise ValueError('Historical training sample count changed')
        args = SimpleNamespace(**config)
        args.train_data_path = train_path
        args.output_dir = out
        args.tiny_model = False
        args.calibration_indices_path = calibration_path
        args.calibration_indices_sha256 = calibration_sha
        args.rsc_selection_indices = list(range(8, 16))
        args.rsc_selection_samples = 8
        args.analysis_device = options.analysis_device or config['analysis_device']
        args.dtype = options.dtype or config['dtype']
        args.direction_correction = 'geometric_v2'
        args.adaptive_lora = False
        for name in ('rsc_proposal_descent_margin', 'rsc_selection_descent_margin',
                     'rsc_loss_min_improvement', 'rsc_trial_strengths'):
            setattr(args, name, getattr(options, name))
        args.rsc_manifest_path = out/'rsc_training_samples.json'
        args.rsc_manifest_sha256 = write_rsc_manifest(args.rsc_manifest_path, dataset, args)
        proposal, selection, manifest = load_rsc_samples(dataset, default_data_collator, args)
        report.update(training_samples=manifest, manifest_sha256=args.rsc_manifest_sha256,
                      actual_configuration={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()})
        model = load_training_model(str(model_path), getattr(torch, args.dtype), 'minicpm5')
        model.to(options.device)
        model.config.use_cache = False
        if config.get('gradient_checkpointing'):
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        names = load_adapter(model, adapter_path)
        layers = {name: model.get_submodule(name) for name in names}
        if any(type(layer) is not LinearWithLoRA for layer in layers.values()):
            raise ValueError('Replay accepts only ordinary Dense LinearWithLoRA')
        # Fail rather than silently quantize FP32 historical factors on load.
        for name, layer in layers.items():
            for key in ('weight_a', 'weight_b'):
                saved, loaded = payload['layers'][name]['state'][key], getattr(layer, key)
                if loaded.dtype != saved.dtype or not torch.equal(loaded.detach().cpu(), saved):
                    raise ValueError(f'PRE factor was not loaded exactly: {name}.{key}')
        originals = {p: p.detach().cpu().clone() for layer in layers.values()
                     for p in (layer.weight_a, layer.weight_b)}
        metadata = [(name, p, p._version, p.requires_grad, tuple(p.shape)) for name,p in model.named_parameters()]
        rank_scaling = {name: (layer.lora_rank, layer.lora_scaler) for name,layer in layers.items()}
        devices = {p.device for p in model.parameters() if p.is_cuda}
        if torch.device(args.analysis_device).type == 'cuda':
            devices.add(torch.device(args.analysis_device))
        for device in devices:
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        with preserved_training_state(model):
            qualified, values, joint, timings = compute_rsc_candidates(
                model, layers, proposal, selection, args, rows, candidates)
        report.update(joint_loss_check=joint, stage_timings_seconds=timings,
                      would_accept=values is not None, committed=False,
                      gpu_peak_memory_bytes={str(d): dict(allocated=torch.cuda.max_memory_allocated(d),
                                                          reserved=torch.cuda.max_memory_reserved(d)) for d in devices},
                      gpu_memory_scope='loaded PRE model plus candidate computation',
                      parameter_dtypes=sorted({str(p.dtype) for p in model.parameters()}))
        selected = {c['module']: c for c in qualified}
        changed_names = ({c['module'] for c,value in zip(qualified,values)
                          if not torch.equal(c['parameter'].detach(),value)} if values is not None else set())
        for row in rows:
            row['would_apply'] = row['module'] in changed_names
            row['would_apply_side'] = selected[row['module']]['side'] if row['would_apply'] else 'none'
            row['would_apply_strength'] = joint['accepted_strength'] if row['would_apply'] else 0.
            if row['direction_accepted']:
                row['reason'] = ('offline_would_accept' if row['would_apply'] else
                                 'rounded_to_no_change' if values is not None else joint['reason'])
        # Save all proposed factor targets separately from JSON diagnostics, never as an adapter.
        torch.save({c['module']: dict(side=c['side'], target=c['target']) for c in candidates}, out/'candidate_factors.pt')
        if {name:(layer.lora_rank,layer.lora_scaler) for name,layer in layers.items()} != rank_scaling:
            raise RuntimeError('Offline computation changed rank/scaling')
    except BaseException as error:
        report.update(error=repr(error), aborted=True, committed=False)
        raise
    finally:
        violations = []
        if model is not None and metadata:
            current = dict(model.named_parameters())
            for name,p,version,flag,shape in metadata:
                if current.get(name) is not p or p.requires_grad != flag or tuple(p.shape) != shape:
                    violations.append(f'parameter metadata: {name}')
                if p not in originals and p._version != version:
                    violations.append(f'base/unrelated parameter changed: {name}')
            with torch.no_grad():
                for p, old in originals.items():
                    if not torch.equal(p.detach().cpu(), old):
                        violations.append('factor not restored by computation')
                        p.copy_(old.to(p))
            report['adapter_factors_restored_to_pre'] = all(
                torch.equal(p.detach().cpu(), old) for p,old in originals.items())
            report['parameter_identities_preserved'] = all(current.get(n) is p for n,p,_,_,_ in metadata)
        report['invariant_violations'] = violations
        for path,digest in ((adapter_path,adapter_sha),(train_path,data_sha),(calibration_path,calibration_sha)):
            verified_sha(path,digest,'read-only input after replay')
        with (out/'modules.jsonl').open('x') as stream:
            for row in rows:
                stream.write(json.dumps(row,allow_nan=False)+'\n')
        with (out/'summary.json').open('x') as stream:
            json.dump(report,stream,indent=2,allow_nan=False)
        if violations:
            raise RuntimeError('Offline PRE invariant failed: '+ '; '.join(violations))


if __name__ == '__main__':
    replay(get_args())
