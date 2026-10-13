"""Legacy or manifest-matched factor Adam-state reset control; no gradient or geometric operations."""
import copy
import json
import hashlib
import random
from pathlib import Path

import numpy as np
import torch
from .official import LinearWithLoRA


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate manifest key: {key}')
        result[key] = value
    return result


def load_reset_manifest(path, target_names, step):
    raw = Path(path).read_bytes()
    manifest = json.loads(raw, object_pairs_hook=_unique_object)
    if not isinstance(manifest, dict):
        raise ValueError('Reset manifest must be an object')
    if type(manifest.get('step')) is not int or manifest['step'] != step:
        raise ValueError('Reset manifest step mismatch')
    if type(manifest.get('num_modules')) is not int or manifest['num_modules'] != len(target_names):
        raise ValueError('Reset manifest num_modules mismatch')
    sides = manifest.get('sides')
    if not isinstance(sides, dict) or set(sides) != set(target_names):
        raise ValueError('Reset manifest modules must exactly match targets (missing/extra module)')
    if any(not isinstance(side, str) or side not in ('A', 'B', 'none') for side in sides.values()):
        raise ValueError('Reset manifest side must be A/B/none')
    return sides, hashlib.sha256(raw).hexdigest()


def reset_only_step(model, target_names, optimizer, scheduler, step, output_dir, side_manifest=None):
    layers = [(name, model.get_submodule(name)) for name in target_names]
    if not layers or len(set(target_names)) != len(layers):
        raise ValueError('Reset-only requires nonempty unique target modules')
    if any(type(layer) is not LinearWithLoRA for _, layer in layers):
        raise ValueError('Reset-only requires ordinary Dense LinearWithLoRA')
    sides = {name: 'B' for name in target_names}
    manifest_hash = None
    if side_manifest is not None:
        sides, manifest_hash = load_reset_manifest(side_manifest, target_names, step)
    destination = Path(output_dir) / 'reset_only' / f'step_{step}'
    destination.mkdir(parents=True, exist_ok=False)
    parameters = [(p, tuple(p.shape), p.requires_grad, p._version) for p in model.parameters()]
    parameter_values = {p: p.detach().cpu().clone() for p, _, _, _ in parameters}
    count = sum(p.numel() for p, _, trainable, _ in parameters if trainable)
    snapshots = [(name, layer, layer.lora_rank, layer.lora_scaler,
                  [(p, parameter_values[p]) for p in
                   (layer.weight, layer.weight_a, layer.weight_b)]) for name, layer in layers]
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    groups_before = [(group, list(group['params']),
                      copy.deepcopy({k: v for k, v in group.items() if k != 'params'}))
                     for group in optimizer.param_groups]
    lrs = [group['lr'] for group in optimizer.param_groups]
    states = {p: (state, {k: (id(v), v._version) if torch.is_tensor(v) else copy.deepcopy(v)
                         for k, v in state.items()}) for p, state in optimizer.state.items()}
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    changed = {getattr(layer, 'weight_' + sides[name].lower())
               for name, layer in layers if sides[name] != 'none'}
    reset_count = sum(p in optimizer.state for p in changed)
    for p in changed:
        optimizer.state.pop(p, None)

    def require(condition, message):
        if not condition:
            raise RuntimeError('Reset-only invariant failed: ' + message)

    current = list(model.parameters())
    require(len(current) == len(parameters), 'parameter count')
    for actual, (p, shape, trainable, version) in zip(current, parameters):
        require(actual is p and tuple(p.shape) == shape and p.requires_grad == trainable
                and p._version == version and torch.equal(p.detach().cpu(), parameter_values[p]),
                'parameter identity/shape/value/trainability')
    for name, layer, rank, scaling, values in snapshots:
        require(layer.lora_rank == rank and layer.lora_scaler == scaling, name + ' rank/scaling')
        for actual, (p, value) in zip((layer.weight, layer.weight_a, layer.weight_b), values):
            require(actual is p and torch.equal(p.detach().cpu(), value), name + ' base/A/B changed')
    require(scheduler.state_dict() == scheduler_state, 'scheduler changed')
    require(lrs == [group['lr'] for group in optimizer.param_groups], 'learning rates changed')
    require(len(optimizer.param_groups) == len(groups_before), 'optimizer param groups count')
    for actual, (group, params, options) in zip(optimizer.param_groups, groups_before):
        require(actual is group and len(group['params']) == len(params)
                and all(a is b for a, b in zip(group['params'], params))
                and {k: v for k, v in group.items() if k != 'params'} == options,
                'optimizer param group changed')
    require(set(optimizer.state) == set(states) - changed, 'optimizer state keys')
    for p, (state, versions) in states.items():
        if p in changed:
            continue
        require(optimizer.state[p] is state and state.keys() == versions.keys(), 'untouched state object')
        for k, v in state.items():
            require(((id(v), v._version) if torch.is_tensor(v) else v) == versions[k],
                    'untouched optimizer state modified')
    require(torch.equal(rng, torch.get_rng_state()), 'CPU RNG')
    if cuda_rng:
        require(all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())), 'CUDA RNG')
    require(random.getstate() == python_rng, 'Python RNG')
    now = np.random.get_state()
    require(now[0] == numpy_rng[0] and np.array_equal(now[1], numpy_rng[1])
            and now[2:] == numpy_rng[2:], 'NumPy RNG')
    new_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    require(count == new_count, 'trainable parameter count')
    summary = dict(step=step, num_target_modules=len(layers), reset_side='B' if side_manifest is None else 'matched',
                   reset_mode='legacy_all_B' if side_manifest is None else 'matched_manifest',
                   reset_sides={side: list(sides.values()).count(side) for side in ('A', 'B', 'none')},
                   manifest_path=str(Path(side_manifest).resolve()) if side_manifest is not None else None,
                   manifest_sha256=manifest_hash,
                   optimizer_state_reset_count=reset_count, parameter_objects_preserved=True,
                   scheduler_preserved=True, learning_rates_preserved=True,
                   base_weights_unchanged=True, lora_factors_unchanged=True,
                   rank_unchanged=True, scaling_unchanged=True, rng_unchanged=True,
                   A_optimizer_state_preserved=all(side != 'A' for side in sides.values()),
                   B_optimizer_state_removed=all(layer.weight_b not in optimizer.state for _, layer in layers),
                   unselected_optimizer_states_preserved=True,
                   old_total_lora_params=count, new_total_lora_params=new_count)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Reset-only step {step}: cleared {reset_count} selected optimizer states', flush=True)
    return summary
