"""Frozen, verbatim reuse of the validated v3 loader (no training loader changes).
Source: minicpm_rank1_safe_probe_v3.py
Source SHA256: 9a52d4cc56c008eb919cb2b20e2128f1abe7d519da10672cea87117cd5803611
Only the eight loading/verification functions are copied, unchanged, because
that local v3 script is untracked. No synthetic probe or training logic is copied.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import torch
from torch import nn

def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for buf in iter(lambda: f.read(1 << 20), b''):
            h.update(buf)
    return h.hexdigest()


def must(ok, what):
    if not bool(ok):
        raise AssertionError(what)


def tensor_exact(a, b):
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)


def load_original_lora_class(repo_root):
    """Use the repository's REAL lora.py, without its broad package __init__.

    This is an isolated test launcher only. It does not modify repository code
    or stub out any LoRA computation, and it needs no DeepSpeed/bitsandbytes.
    """
    source = Path(repo_root) / 'MyTransformers_RaLoRA_official' / 'common' / 'lora_modules' / 'lora.py'
    must(source.is_file(), f'Cannot find original LoRA implementation: {source}')
    module_name = '_rank1_probe_original_minicpm_lora'
    if module_name in sys.modules:
        impl = sys.modules[module_name]
    else:
        info = importlib.util.spec_from_file_location(module_name, source)
        must(info is not None and info.loader is not None, 'Cannot load original lora.py')
        impl = importlib.util.module_from_spec(info)
        sys.modules[module_name] = impl
        info.loader.exec_module(impl)
    return impl.LinearWithLoRA, impl.LoRAConfig


def make_original_layer(old, config, lora_cls):
    """Same construction as adaptive_lora.dense_init.make_dense_layer."""
    layer = lora_cls(config)
    layer.lora_config = config
    layer.weight, layer.bias = old.weight, old.bias
    layer.init_lora_weights()
    layer.to(device=old.weight.device)
    return layer


def restore_dense_adapter_on_full_model(model, payload, repo_root):
    """Rebuild adapter factors directly with original LoRA class; no official.py.

    Mirrors checkpoint.load_adapter's Dense branch without importing unrelated
    variant registries. Fail on anything other than supported plain Dense LoRA.
    """
    lora_cls, config_cls = load_original_lora_class(repo_root)
    for name, saved in payload['layers'].items():
        must(saved.get('structure') == 'dense', f'Non-Dense adapter not supported: {name}')
        state = saved['state']
        must('add_a' not in state and 'add_b' not in state,
             f'Adapter already expanded: {name}')
        must(set(state) == {'weight_a', 'weight_b'},
             f'Unexpected checkpoint factor keys in {name}: {sorted(state)}')
        current = model.get_submodule(name)
        must(isinstance(current, nn.Linear), f'Expected linear target at {name}')
        config = config_cls(**saved['config'])
        must(tuple(state['weight_a'].shape) == (config.lora_rank, config.in_features)
             and tuple(state['weight_b'].shape) == (config.out_features, config.lora_rank),
             f'Invalid adapter shape at {name}')
        layer = make_original_layer(current, config, lora_cls)
        layer.lora_scaler = float(saved['scaling'])
        with torch.no_grad():
            layer.weight_a.copy_(state['weight_a'].to(layer.weight_a.device, layer.weight_a.dtype))
            layer.weight_b.copy_(state['weight_b'].to(layer.weight_b.device, layer.weight_b.dtype))
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model, child, layer)
    return list(payload['layers'])


def verify_disk_weights(parameters, model_dir):
    """Compare EVERY base tensor with its real local safetensors value."""
    from safetensors import safe_open
    folder = Path(model_dir)
    index = folder / 'model.safetensors.index.json'
    if index.exists():
        mapping = json.loads(index.read_text())['weight_map']
    else:
        with safe_open(folder / 'model.safetensors', framework='pt') as handle:
            mapping = {key: 'model.safetensors' for key in handle.keys()}
    must(set(parameters) == set(mapping), 'Loaded base parameter keys differ from safetensors index')
    checked = 0
    for filename in sorted(set(mapping.values())):
        with safe_open(folder / filename, framework='pt', device='cpu') as handle:
            expected = {key for key, value in mapping.items() if value == filename}
            must(set(handle.keys()) == expected, 'Shard keys differ from index: '+filename)
            for key in sorted(expected):
                source = handle.get_tensor(key)
                actual = parameters[key].detach().cpu()
                must(tensor_exact(actual, source.to(actual.dtype)), 'Base weight differs from checkpoint: '+key)
                must(torch.isfinite(actual).all(), 'Nonfinite base weight: '+key)
                checked += 1
    return checked


def load_probe_model(model_dir, dtype):
    """Probe-local standard HF loader; do not use the training loader's 4.x workaround."""
    from transformers import AutoModelForCausalLM
    model, info = AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=True,
        dtype=dtype, attn_implementation='eager', use_safetensors=True,
        output_loading_info=True)
    for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'):
        must(key in info and not info[key], f'Incomplete real checkpoint loading: {key}={info.get(key)}')
    must(not any(p.is_meta for p in model.parameters()), 'Unmaterialized model parameters')
    count = verify_disk_weights(dict(model.named_parameters()), model_dir)
    return model, dict(loading_info={k:list(v) for k,v in info.items()}, base_tensors_verified=count,
                       all_base_weights_match_safetensors=True,
                       interface='AutoModelForCausalLM.from_pretrained(path), no state_dict')
