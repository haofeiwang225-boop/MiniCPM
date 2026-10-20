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
from .official import LinearWithLoRA
from .subspace_matching import orth
from .direction_correction import relative_A_projection_change, relative_B_projection_change
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


def extract_historical_candidates(pre, post, tolerance, history):
    """Recover actual PRE/POST changes, never regenerate geometric directions.

    orth() is used only for small-factor numerical-rank validation, not gradient
    SVD, GID estimation, or candidate planning. Any invalid historical change
    aborts the entire audit; there is no feasible-subset fallback.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid historical trust tolerance')
    if pre.get('format_version') != post.get('format_version'):
        raise ValueError('PRE/POST format mismatch')
    if Path(pre['base_model']).resolve() != Path(post['base_model']).resolve():
        raise ValueError('PRE/POST base model mismatch')
    if not pre['layers'] or set(pre['layers']) != set(post['layers']):
        raise ValueError('PRE/POST target modules mismatch')
    candidates, rows = [], []
    for name, old in pre['layers'].items():
        new = post['layers'][name]
        if old.get('structure') != 'dense' or new.get('structure') != 'dense':
            raise ValueError(f'Non-Dense checkpoint: {name}')
        if old['config'] != new['config'] or old['scaling'] != new['scaling']:
            raise ValueError(f'PRE/POST rank/config/scaling mismatch: {name}')
        rank = old['config']['lora_rank']
        if type(rank) is not int or rank <= 0 or not math.isfinite(old['scaling']):
            raise ValueError(f'Invalid rank/scaling: {name}')
        if set(old['state']) != {'weight_a', 'weight_b'} or set(new['state']) != set(old['state']):
            raise ValueError(f'Unexpected or missing factors: {name}')
        expected = {'weight_a': (rank, old['config']['in_features']),
                    'weight_b': (old['config']['out_features'], rank)}
        changed = []
        for key, side in [('weight_a', 'A'), ('weight_b', 'B')]:
            x, y = old['state'][key], new['state'][key]
            if (not torch.is_tensor(x) or not torch.is_tensor(y)
                    or tuple(x.shape) != expected[key] or x.shape != y.shape
                    or x.dtype != y.dtype or not x.is_floating_point()
                    or not torch.isfinite(x).all() or not torch.isfinite(y).all()):
                raise ValueError(f'PRE/POST factor shape/dtype/value mismatch: {name}.{key}')
            if not torch.equal(x, y):
                changed.append(side)
        if len(changed) > 1:
            raise ValueError(f'Both factors changed: {name}')
        side = changed[0] if changed else 'none'
        row = dict(module=name, rank=rank, scaling=old['scaling'], side=side)
        if changed:
            a, b = old['state']['weight_a'], old['state']['weight_b']
            if orth(a.T).shape[1] != rank or orth(b).shape[1] != rank:
                raise ValueError(f'PRE factor rank deficient: {name}')
            key = 'weight_a' if side == 'A' else 'weight_b'
            layer = SimpleNamespace(weight_a=a, weight_b=b, lora_rank=rank)
            c = dict(module=name, side=side, target=new['state'][key].clone(),
                     parameter=old['state'][key], layer=layer)
            check = candidate_check(c, tolerance)
            if not check['feasible']:
                raise ValueError(f'Historical candidate violates rank/trust: {name}: {check}')
            row['check'] = check
            candidates.append(dict(module=name, side=side, target=c['target']))
        rows.append(row)
    counts = {side: sum(r['side'] == side for r in rows) for side in ('A', 'B', 'none')}
    expected = history.get('actual_changed_sides', {})
    if (history.get('invariants_passed') is not True or history.get('step') != 1600
            or set(expected) != set(counts)
            or any(type(n) is not int or n < 0 for n in expected.values())
            or history.get('num_modules') != len(rows) or counts != expected):
        raise ValueError(f'Historical side-count mismatch or invalid audit: actual={counts}, '
                         f'expected={expected}, modules={len(rows)}')
    return candidates, rows, counts


def bind_historical_candidates(model, pre, records):
    """Require exact loaded PRE values and exact representability of POST."""
    for name, spec in pre['layers'].items():
        layer = model.get_submodule(name)
        if (type(layer) is not LinearWithLoRA or layer.lora_rank != spec['config']['lora_rank']
                or layer.lora_scaler != spec['scaling']):
            raise ValueError(f'Loaded PRE structure mismatch: {name}')
        for key in ('weight_a', 'weight_b'):
            p, saved = getattr(layer, key), spec['state'][key]
            if p.dtype != saved.dtype or not torch.equal(p.detach().cpu(), saved.cpu()):
                raise ValueError(f'Loaded PRE factor mismatch: {name}.{key}')
    candidates = []
    for record in records:
        layer = model.get_submodule(record['module'])
        p = layer.weight_a if record['side'] == 'A' else layer.weight_b
        target = record['target']
        if target.dtype != p.dtype or target.shape != p.shape:
            raise ValueError(f'POST factor cannot be loaded exactly: {record["module"]}')
        candidates.append(dict(record, layer=layer, parameter=p, target=target.to(p.device)))
    return candidates


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
        checks = [candidate_check(c, tolerance) for c in candidates]
        for c, check in zip(candidates, checks):
            if not check['feasible']:
                raise ValueError(f'Historical candidate violates rank/trust: {c["module"]}: {check}')
        batches = {k: [default_data_collator([dataset[i]]) for i in groups[k]]
                   for k in ('proposal', 'selector')}
        pd = directional_slopes(model, candidates, batches['proposal'])
        sd = directional_slopes(model, candidates, batches['selector'])
        rows, selected = [], []
        for c, check in zip(candidates, checks):
            name = c['module']
            passed = check['feasible'] and pd[name]['mean'] < 0 and sd[name]['mean'] < 0
            rows.append(dict(module=name, side=c['side'], proposal=pd[name], selector=sd[name],
                             check=check, selected=passed,
                             reason='both_down' if passed else check['reason'] if not check['feasible']
                             else 'nonnegative_directional_derivative'))
            if passed:
                selected.append(c)
        results = {}
        audit_data = [dataset[i] for i in groups['audit']]
        for policy, updates in [('original', []), ('all_candidates', candidates), ('filtered', selected)]:
            with temporary_candidates(updates):
                summary, samples = evaluate_dataset(model, audit_data, next(model.parameters()).device)
            for row, index in zip(samples, groups['audit']):
                row['index'] = index
            results[policy] = dict(**summary, samples=samples, num_updated_modules=len(updates))
        baseline = results['original']['token_weighted_nll']
        for result in results.values():
            result['delta_from_original'] = result['token_weighted_nll'] - baseline
    return dict(candidates=rows, policies=results, all_candidates_definition='all historical PRE/POST changed factors; invalid candidates abort',
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
    p.add_argument('--post_adapter', type=Path, required=True)
    p.add_argument('--historical_audit', type=Path, required=True,
                   help='Existing cpu_audit.json: checkpoint hashes and actual_changed_sides')
    p.add_argument('--expected_adapter_sha256', required=True)
    p.add_argument('--output_dir', type=Path, required=True)
    p.add_argument('--device', default='cuda')
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
    history = json.loads(options.historical_audit.read_text())
    verified(options.pre_adapter, history.get('checkpoint_pre_sha256'))
    hashes[str(options.post_adapter)] = verified(options.post_adapter, history.get('checkpoint_post_sha256'))
    hashes[str(options.historical_audit)] = file_sha256(options.historical_audit)
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
    post = torch.load(options.post_adapter, map_location='cpu', weights_only=True)
    records, changes, counts = extract_historical_candidates(
        payload, post, cfg['direction_function_tolerance'], history)
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
    candidates = bind_historical_candidates(model, payload, records)
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
                  device=options.device,
                  pre_sha256=hashes[str(options.pre_adapter)], post_sha256=hashes[str(options.post_adapter)],
                  historical_changes=changes, actual_changed_sides=counts,
                  historical_expected_sides=history['actual_changed_sides'], historical_counts_match=True,
                  num_candidates=len(candidates), candidate_source='historical_PRE_POST_difference',
                  warning='Audit samples are training data; never use audit NLL to tune selection.',
                  tensor_version_note='Temporary copy_ restores values, not Tensor version counters.')
    # Only reserve output after all input validation and model/adapter loading succeed.
    options.output_dir.mkdir(parents=True, exist_ok=False)
    try:
        with preserved_state(model):
            report.update(audit_candidates(model, candidates, dataset, groups, cfg['direction_function_tolerance']))
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
