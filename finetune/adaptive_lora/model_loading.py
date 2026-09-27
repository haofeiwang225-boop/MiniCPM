"""Read local HF checkpoints without assuming optional safetensors metadata."""
import json
from pathlib import Path
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM


def load_training_model(path, dtype, model_family):
    kwargs = dict(trust_remote_code=True, local_files_only=True,
                  torch_dtype=dtype, attn_implementation='eager')
    if model_family != 'minicpm5':
        return AutoModelForCausalLM.from_pretrained(path, **kwargs)
    # transformers 4.46 dereferences metadata.get('format') even when metadata=None.
    # Use its public state_dict interface; original tensors/files stay untouched.
    folder = Path(path)
    index = folder/'model.safetensors.index.json'
    if index.exists():
        names = sorted(set(json.loads(index.read_text())['weight_map'].values()))
    else:
        names = ['model.safetensors']
    state = {}
    for name in names:
        shard = load_file(str(folder/name), device='cpu')
        if state.keys() & shard.keys():
            raise ValueError('Duplicate checkpoint tensor names')
        state.update(shard)
    model, info = AutoModelForCausalLM.from_pretrained(path, state_dict=state,
                                                     output_loading_info=True, **kwargs)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Checkpoint was not loaded exactly: {info}')
    return model
