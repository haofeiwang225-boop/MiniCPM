"""Adapter reconstruction and plain HF merged export, without benchmark edits."""
from dataclasses import asdict
import torch
from torch import nn
from .official import LinearWithRaLoRA, LoRAConfig, dense_factors, freeze_base
from .rank_expansion import ExpandedLoRA


def save_adapter(model, names, path, base_model):
    layers = {}
    for name in names:
        module = model.get_submodule(name)
        base = module.base_layer if isinstance(module, ExpandedLoRA) else module
        layers[name] = dict(config=asdict(base.lora_config), n_split=base.n_split,
                            mini_rank=base.mini_lora_rank, scaling=base.lora_scaler,
                            forward_method=base.forward_method,
                            dynamic_scaling=base.dynamic_scaling, rank_stablize=base.rank_stablize,
                            state={k: v.detach().cpu() for k, v in module.state_dict().items()
                                   if 'weight_a' in k or 'weight_b' in k or k in ('add_a', 'add_b')})
    torch.save(dict(format_version=1, base_model=str(base_model), layers=layers), path)


def load_adapter(model, path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    for name, spec in payload['layers'].items():
        old = model.get_submodule(name)
        layer = LinearWithRaLoRA(LoRAConfig(**spec['config']),
                                forward_method=spec['forward_method'],
                                ralora_dynamic_scaling=spec['dynamic_scaling'],
                                ralora_rank_stablize=spec['rank_stablize'])
        layer.weight = old.weight
        layer.bias = old.bias
        layer.dynamic_init(spec['config']['lora_rank'], spec['mini_rank'], spec['n_split'])
        layer.lora_scaler = spec['scaling']
        layer.to(device=old.weight.device)
        state = spec['state']
        if 'add_a' in state:
            layer = ExpandedLoRA(layer, state['add_a'], state['add_b'])
        missing, unexpected = layer.load_state_dict(state, strict=False)
        if unexpected or any('weight_a' in k or 'weight_b' in k or k in ('add_a', 'add_b') for k in missing):
            raise ValueError('Incomplete adapter checkpoint')
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model, child, layer)
    freeze_base(model)
    for module in model.modules():
        if isinstance(module, ExpandedLoRA):
            module.add_a.requires_grad_(True)
            module.add_b.requires_grad_(True)
    return list(payload['layers'])


def merge_for_export(model, names):
    """Replace custom layers by ordinary Linear; original model config stays valid."""
    for name in names:
        layer = model.get_submodule(name)
        base = layer.base_layer if isinstance(layer, ExpandedLoRA) else layer
        a, b = dense_factors(base)
        delta = base.lora_scaler * (b @ a)
        if isinstance(layer, ExpandedLoRA):
            delta = delta + layer.add_b.detach() @ layer.add_a.detach()
        linear = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
                           device=base.weight.device, dtype=base.weight.dtype)
        with torch.no_grad():
            linear.weight.copy_(base.weight + delta.to(base.weight.dtype))
            if base.bias is not None:
                linear.bias.copy_(base.bias)
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model, child, linear)
    return model
