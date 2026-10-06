"""B-factor Adam-state reset control; no gradient or geometric operations."""
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
from .official import LinearWithLoRA


def reset_only_step(model, target_names, optimizer, scheduler, step, output_dir):
    layers = [(name, model.get_submodule(name)) for name in target_names]
    if not layers or len(set(target_names)) != len(layers):
        raise ValueError('Reset-only requires nonempty unique target modules')
    if any(type(layer) is not LinearWithLoRA for _, layer in layers):
        raise ValueError('Reset-only requires ordinary Dense LinearWithLoRA')
    destination = Path(output_dir) / 'reset_only' / f'step_{step}'
    destination.mkdir(parents=True, exist_ok=False)
    parameters = [(p, tuple(p.shape), p.requires_grad, p._version) for p in model.parameters()]
    count = sum(p.numel() for p, _, trainable, _ in parameters if trainable)
    snapshots = [(name, layer, layer.lora_rank, layer.lora_scaler,
                  [(p, p.detach().cpu().clone()) for p in
                   (layer.weight, layer.weight_a, layer.weight_b)]) for name, layer in layers]
    scheduler_state = copy.deepcopy(scheduler.state_dict())
    lrs = [group['lr'] for group in optimizer.param_groups]
    states = {p: (state, {k: (id(v), v._version) if torch.is_tensor(v) else copy.deepcopy(v)
                         for k, v in state.items()}) for p, state in optimizer.state.items()}
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    changed = {layer.weight_b for _, layer in layers}
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
                and p._version == version, 'parameter identity/shape/value/trainability')
    for name, layer, rank, scaling, values in snapshots:
        require(layer.lora_rank == rank and layer.lora_scaler == scaling, name + ' rank/scaling')
        for actual, (p, value) in zip((layer.weight, layer.weight_a, layer.weight_b), values):
            require(actual is p and torch.equal(p.detach().cpu(), value), name + ' base/A/B changed')
    require(scheduler.state_dict() == scheduler_state, 'scheduler changed')
    require(lrs == [group['lr'] for group in optimizer.param_groups], 'learning rates changed')
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
    summary = dict(step=step, num_target_modules=len(layers), reset_side='B',
                   optimizer_state_reset_count=reset_count, parameter_objects_preserved=True,
                   scheduler_preserved=True, learning_rates_preserved=True,
                   base_weights_unchanged=True, lora_factors_unchanged=True,
                   rank_unchanged=True, scaling_unchanged=True, rng_unchanged=True,
                   A_optimizer_state_preserved=True, B_optimizer_state_removed=True,
                   old_total_lora_params=count, new_total_lora_params=new_count)
    (destination / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(f'Reset-only step {step}: cleared {reset_count} B optimizer states', flush=True)
    return summary
