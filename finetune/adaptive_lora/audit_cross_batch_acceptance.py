"""Read-only cross-batch audit; never train, reset Adam, or gate on audit NLL.

Run from the repository root with PYTHONPATH=finetune python -m
adaptive_lora.audit_cross_batch_acceptance --help. Proposal/Selector slopes use
mean per-example training loss; reported independent NLL is token-weighted.
Only committed v1 primitives are required (no untracked v2 dependencies).
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch
from transformers import default_data_collator

from evaluate_adapter_loss import evaluate_dataset
from .gradient_probe import probe_weight
from .official import LinearWithLoRA, dense_factors
from .residual_gid import svd_with_gid
from .subspace_matching import match_subspaces, orth
from .residual_gradient import residual_gradient
from .tangent_metrics import tangent_metrics
from .direction_correction import (plan_geometric_direction_correction,
    relative_A_projection_change, relative_B_projection_change)
from .data_provenance import file_sha256, git_version


@contextmanager
def preserved_state(model):
    """Restore modes, buffers, gradient objects/values, flags and all RNG streams."""
    params = [(p, p.requires_grad, p.grad, None if p.grad is None else p.grad.clone())
              for p in model.parameters()]
    modes = [(m, m.training) for m in model.modules()]
    buffers = [(b, b.clone()) for b in model.buffers()]
    py, np_state = random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices):
            try:
                model.eval()
                yield
            finally:
                with torch.no_grad():
                    for p, flag, grad, value in params:
                        p.requires_grad_(flag)
                        p.grad = grad
                        if grad is not None and not torch.equal(grad, value):
                            grad.copy_(value)
                    for b, old in buffers:
                        if not torch.equal(b, old):
                            b.copy_(old)
                for module, training in modes:
                    module.training = training
    finally:
        random.setstate(py)
        np.random.set_state(np_state)


def validate_groups(groups, size, fingerprints=None):
    if set(groups) != {'proposal', 'selector', 'audit'}:
        raise ValueError('Require proposal, selector and audit groups')
    seen, hashes = set(), set()
    for name, indices in groups.items():
        if not indices or any(type(i) is not int or not 0 <= i < size for i in indices):
            raise ValueError(f'Invalid indices: {name}')
        if len(set(indices)) != len(indices) or seen.intersection(indices):
            raise ValueError('Sample indices overlap or repeat')
        seen.update(indices)
        if fingerprints is not None:
            current = {fingerprints[i] for i in indices}
            if current & hashes:
                raise ValueError('Identical sample content across calibration groups')
            hashes.update(current)


def generate_candidates(model, names, batches, args):
    """Compose existing v1 probe/residual/SVD/planner without changing geometry."""
    candidates, rows = [], []
    for name in names:
        layer = model.get_submodule(name)
        if type(layer) is not LinearWithLoRA:
            raise ValueError('Audit supports ordinary Dense LoRA only')
        g = probe_weight(model, layer, batches).to(args.analysis_device)
        a, b = (v.to(g.device) for v in dense_factors(layer))
        u, s, vh, gid = svd_with_gid(g)
        qa, qb, ca, cb = match_subspaces(a, b, u, s, vh)
        residual = residual_gradient(g, qa, qb)
        if residual.norm() <= 1e-6 * g.norm():
            residual.zero_()
        metrics = tangent_metrics(g, residual, s, layer.lora_rank, ca, cb)
        plan, factors = plan_geometric_direction_correction(
            g, a, b, metrics['oracle_tangent_coverage'], layer.lora_scaler,
            residual_threshold=args.adaptive_residual_energy_threshold,
            function_tolerance=args.direction_function_tolerance,
            min_gain=args.direction_min_gain, eta_start=args.direction_eta_start,
            eta_decay=args.direction_eta_decay, eta_min=args.direction_eta_min)
        row = dict(module=name, gid_full=gid, **metrics, plan=plan)
        rows.append(row)
        if factors is not None:
            side = plan['applied_side']
            if side not in ('A', 'B'):
                raise ValueError('Expected exactly one candidate factor per layer')
            p = layer.weight_a if side == 'A' else layer.weight_b
            candidates.append(dict(module=name, layer=layer, side=side, parameter=p,
                                   target=factors[0 if side == 'A' else 1].to(p).detach().clone()))
    return candidates, rows


def directional_slopes(model, candidates, batches):
    """Exact factor autograd dot actual stored-precision delta at unchanged PRE."""
    if not candidates:
        return {}
    params = [c['parameter'] for c in candidates]
    if len(set(params)) != len(params) or not batches:
        raise ValueError('Require unique candidate parameters and nonempty batches')
    slopes = {c['module']: [] for c in candidates}
    with preserved_state(model), torch.enable_grad():
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        for batch in batches:
            if any(v.shape[0] != 1 for v in batch.values()):
                raise ValueError('Slopes require batch_size=1')
            loss = model(**{k: v.to(params[0].device) for k, v in batch.items()}).loss
            if loss.numel() != 1 or not torch.isfinite(loss):
                raise ValueError('Nonfinite or nonscalar slope loss')
            grads = torch.autograd.grad(loss, params)
            for c, grad in zip(candidates, grads):
                delta = c['target'].to(c['parameter']).double() - c['parameter'].detach().double()
                slope = float((grad.double() * delta).sum())
                if not math.isfinite(slope):
                    raise ValueError('Nonfinite slope')
                slopes[c['module']].append(slope)
    return {n: dict(per_sample=v, mean=math.fsum(v)/len(v)) for n, v in slopes.items()}


def candidate_check(c, tolerance):
    layer, side = c['layer'], c['side']
    value = c['target'].to(c['parameter'])
    a = value if side == 'A' else layer.weight_a.detach()
    b = value if side == 'B' else layer.weight_b.detach()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return dict(feasible=False, reason='nonfinite')
    fn = relative_A_projection_change if side == 'A' else relative_B_projection_change
    relative, _, _ = fn(layer.weight_a.detach(), layer.weight_b.detach(), value)
    ranks = [orth(a.T).shape[1], orth(b).shape[1]]
    feasible = ranks == [layer.lora_rank]*2 and math.isfinite(relative) and relative <= tolerance
    return dict(feasible=feasible, relative_update_change=relative, ranks=ranks,
                reason='feasible' if feasible else 'rank_or_trust_region')


@contextmanager
def temporary_candidates(candidates):
    originals = [(c['parameter'], c['parameter'].detach().clone()) for c in candidates]
    try:
        with torch.no_grad():
            for c in candidates:
                c['parameter'].copy_(c['target'])
        yield
    finally:
        with torch.no_grad():
            for p, old in originals:
                p.copy_(old)
                if not torch.equal(p, old):
                    raise RuntimeError('Failed to restore PRE factor exactly')


def audit_candidates(model, candidates, dataset, groups, tolerance):
    """All/filtered policies are fixed before independent audit samples are scored."""
    validate_groups(groups, len(dataset))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid trust tolerance')
    if len({c['module'] for c in candidates}) != len(candidates):
        raise ValueError('Duplicate candidate module')
    with preserved_state(model):
        batches = {k: [default_data_collator([dataset[i]]) for i in groups[k]]
                   for k in ('proposal', 'selector')}
        pd = directional_slopes(model, candidates, batches['proposal'])
        sd = directional_slopes(model, candidates, batches['selector'])
        rows, feasible, selected = [], [], []
        for c in candidates:
            name = c['module']
            check = candidate_check(c, tolerance)
            passed = check['feasible'] and pd[name]['mean'] < 0 and sd[name]['mean'] < 0
            rows.append(dict(module=name, side=c['side'], proposal=pd[name], selector=sd[name],
                             check=check, selected=passed,
                             reason='both_down' if passed else check['reason'] if not check['feasible']
                             else 'nonnegative_directional_derivative'))
            if check['feasible']:
                feasible.append(c)
            if passed:
                selected.append(c)
        results = {}
        audit_data = [dataset[i] for i in groups['audit']]
        for policy, updates in [('original', []), ('all_candidates', feasible), ('filtered', selected)]:
            with temporary_candidates(updates):
                summary, samples = evaluate_dataset(model, audit_data, next(model.parameters()).device)
            for row, index in zip(samples, groups['audit']):
                row['index'] = index
            results[policy] = dict(**summary, samples=samples, num_updated_modules=len(updates))
        baseline = results['original']['token_weighted_nll']
        for result in results.values():
            result['delta_from_original'] = result['token_weighted_nll'] - baseline
    return dict(candidates=rows, policies=results, all_candidates_definition='all storage-precision rank/trust-feasible v1 candidates',
                slope_loss='mean_per_sample', audit_loss='token_weighted',
                audit_used_for_selection=False, optimizer_created=False, scheduler_created=False)


def verified(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or file_sha256(path) != expected:
        raise ValueError(f'Missing or mismatched expected SHA256: {path}')
    return expected


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run_config', type=Path, required=True)
    p.add_argument('--pre_adapter', type=Path, required=True)
    p.add_argument('--expected_adapter_sha256', required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--analysis_device', default=None, help='Defaults to --device')
    p.add_argument('--dtype', choices=['float32', 'bfloat16'], default=None,
                   help='Base precision override; saved LoRA factors must load exactly')
    p.add_argument('--selector_indices', nargs='+', type=int, default=list(range(8, 16)))
    p.add_argument('--audit_indices', nargs='+', type=int, default=list(range(16, 48)))
    options = p.parse_args()
    if options.output_dir.exists():
        p.error('Output directory already exists')
    cfg = json.loads(options.run_config.read_text())
    if cfg['model_family'] != 'minicpm5' or cfg['lora_structure'] != 'dense' or cfg['batch_size'] != 1:
        p.error('Require MiniCPM5 Dense LoRA batch_size=1')
    data = Path(cfg['train_data_path'])
    calibration_path = Path(cfg['calibration_indices_path'])
    hashes = {str(data): verified(data, cfg.get('train_data_sha256')),
              str(calibration_path): verified(calibration_path, cfg.get('calibration_indices_sha256')),
              str(options.pre_adapter): verified(options.pre_adapter, options.expected_adapter_sha256)}
    calibration = json.loads(calibration_path.read_text())
    if (calibration['split'] != 'train' or Path(calibration['dataset_source']).resolve() != data.resolve()
            or data.name != 'train.json' or calibration['batch_size'] != 1
            or any(len(b) != 1 for b in calibration['batch_indices'])):
        p.error('Require historical singleton training calibration; dev/test forbidden')
    raw = json.loads(data.read_text())
    fingerprints = [hashlib.sha256(json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest() for r in raw]
    groups = dict(proposal=[b[0] for b in calibration['batch_indices']],
                  selector=options.selector_indices, audit=options.audit_indices)
    validate_groups(groups, len(raw), fingerprints)
    if len(groups['proposal']) != 8 or len(raw) != calibration['dataset_length'] or len(raw) != cfg['train_samples']:
        p.error('Historical K=8 / dataset size mismatch')
    payload = torch.load(options.pre_adapter, map_location='cpu', weights_only=True)
    if Path(payload['base_model']).resolve() != Path(cfg['model_name_or_path']).resolve() or set(payload['layers']) != set(cfg['targets']):
        p.error('PRE base model / target modules mismatch')
    if any(s.get('structure') != 'dense' for s in payload['layers'].values()):
        p.error('Only Dense PRE adapters supported')
    options.output_dir.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    from .model_loading import load_training_model
    from .checkpoint import load_adapter
    from .training_data import ChatTrainingDataset
    random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    tokenizer = AutoTokenizer.from_pretrained(cfg['model_name_or_path'], trust_remote_code=True, local_files_only=True)
    if tokenizer.chat_template is None:
        tokenizer.chat_template = (Path(cfg['model_name_or_path'])/'chat_template.jinja').read_text()
    tokenizer.pad_token = tokenizer.eos_token
    dataset = ChatTrainingDataset(data, tokenizer, cfg['model_max_length'], dynamic_length=cfg['dynamic_length'])
    model = load_training_model(cfg['model_name_or_path'], getattr(torch, options.dtype or cfg['dtype']), 'minicpm5').to(options.device)
    model.config.use_cache = False
    names = load_adapter(model, options.pre_adapter)
    originals = {}
    for name in names:
        for key in ('weight_a', 'weight_b'):
            param = getattr(model.get_submodule(name), key)
            saved = payload['layers'][name]['state'][key]
            if saved.dtype != param.dtype or not torch.equal(param.detach().cpu(), saved):
                raise ValueError(f'PRE adapter did not load exactly: {name}.{key}')
            originals[param] = param.detach().clone()
    metadata = [(n, v, v._version, v.requires_grad, tuple(v.shape)) for n, v in model.named_parameters()]
    ranks = {n: (model.get_submodule(n).lora_rank, model.get_submodule(n).lora_scaler) for n in names}
    report = dict(mode='offline_diagnostic_only', **git_version(), config=cfg, input_sha256=hashes,
                  run_config_sha256=file_sha256(options.run_config), groups=groups,
                  sample_sha256={k: [fingerprints[i] for i in v] for k, v in groups.items()},
                  dtype=options.dtype or cfg['dtype'], historical_training_state_restored=False,
                  device=options.device, analysis_device=options.analysis_device or options.device,
                  warning='Audit samples are training data; never use audit NLL to tune selection.',
                  tensor_version_note='Temporary copy_ restores values, not Tensor version counters.')
    try:
        with preserved_state(model):
            batches = [default_data_collator([dataset[i]]) for i in groups['proposal']]
            planning_args = SimpleNamespace(**cfg)
            planning_args.analysis_device = options.analysis_device or options.device
            candidates, plans = generate_candidates(model, names, batches, planning_args)
            report.update(audit_candidates(model, candidates, dataset, groups, cfg['direction_function_tolerance']))
            report['geometric_plans'] = plans
    except BaseException as error:
        report['error'] = repr(error)
        raise
    finally:
        violations = []
        current = dict(model.named_parameters())
        for n, param, version, flag, shape in metadata:
            if current.get(n) is not param or param.requires_grad != flag or tuple(param.shape) != shape:
                violations.append(n + ': parameter metadata')
            if param not in originals and param._version != version:
                violations.append(n + ': base parameter version')
        with torch.no_grad():
            for param, old in originals.items():
                if not torch.equal(param, old):
                    violations.append('PRE factor not restored')
                    param.copy_(old)
        if ranks != {n: (model.get_submodule(n).lora_rank, model.get_submodule(n).lora_scaler) for n in names}:
            violations.append('rank/scaling')
        for path, digest in hashes.items():
            verified(path, digest)
        report['invariant_violations'] = violations
        report['pre_factors_restored'] = all(torch.equal(p, old) for p, old in originals.items())
        with (options.output_dir/'results.json').open('x') as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        if violations:
            raise RuntimeError(str(violations))


if __name__ == '__main__':
    main()
