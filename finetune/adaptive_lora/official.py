"""Thin integration of local official layers, allocation, and initialization."""
from pathlib import Path
import sys
from types import SimpleNamespace
import torch

OFFICIAL = Path(__file__).resolve().parents[2] / 'MyTransformers_RaLoRA_official'
sys.path.insert(0, str(OFFICIAL))
from common.lora_modules.lora import LinearWithLoRA, LoRAConfig
from common.lora_modules.ralora import LinearWithRaLoRA, ralora_reinit
from common.lora_modules.lora_set_up import LoRAManager
from common.lora_modules.grad_utils import get_record_gradient_post_accumulate_hook


def official_args(args):
    return SimpleNamespace(
        lora_rank=args.lora_rank, lora_scaler=args.lora_alpha,
        lora_dropout=args.lora_dropout, run_lora_in_fp32=True,
        weight_a_init_method=args.weight_a_init_method, weight_b_init_method=None,
        device=args.device, batch_size_per_gpu=args.batch_size, global_rank=0, world_size=1,
        ralora_features_func=None, ralora_min_rank=args.ralora_min_rank,
        ralora_max_rank=args.ralora_max_rank, ralora_erank_max_power=args.ralora_erank_max_power,
        ralora_svd_threshold=0, ralora_cumulative_variance_threshold=0,
        ralora_allocate_by_erank=False, output_path=str(args.output_dir),
        experiment_name='initial_probe',
    )


def attach_ralora(model, args):
    options = official_args(args)
    targets = [(name, module) for name, module in model.named_modules()
               if isinstance(module, torch.nn.Linear) and name.rsplit('.', 1)[-1] in args.target_modules]
    if not targets:
        raise ValueError('No requested Linear targets found')
    matched = {name.rsplit('.', 1)[-1] for name, _ in targets}
    if matched != set(args.target_modules):
        raise ValueError(f'Unmatched target modules: {set(args.target_modules)-matched}')
    for name, module in targets:
        variant = dict(forward_method=args.ralora_forward_method,
                       ralora_dynamic_scaling=args.ralora_dynamic_scaling,
                       ralora_rank_stablize=args.ralora_rank_stablize)
        layer = LoRAManager.create_lora_layer(module, LinearWithRaLoRA, variant, options)
        parent, _, child = name.rpartition('.')
        setattr(model.get_submodule(parent) if parent else model, child, layer)
    return [name for name, _ in targets]


def first_probe(model, batches, args):
    # Original algorithm unchanged. Guard the frozen weights against accidental update.
    before = {n: m.weight.detach().cpu().clone() for n, m in model.named_modules()
              if isinstance(m, LinearWithRaLoRA)}
    ralora_reinit(model, batches, official_args(args), iters=len(batches))
    for name, old in before.items():
        layer = model.get_submodule(name)
        if not torch.equal(layer.weight.detach().cpu(), old):
            raise RuntimeError(f'Initial probe modified base weight: {name}')
        # Official dynamic_init constructs A/B on CPU; preserve FP32 adapter dtype.
        layer.to(device=args.device)
    freeze_base(model)


def freeze_base(model):
    for param in model.parameters():
        param.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LinearWithLoRA) and module.has_lora_weights:
            for name in ('weight_a', 'weight_b'):
                value = getattr(module, name)
                if isinstance(value, torch.nn.ParameterList):
                    for p in value:
                        p.requires_grad_(True)
                else:
                    value.requires_grad_(True)


def blocks(layer):
    if isinstance(layer.weight_a, torch.nn.ParameterList) or layer.weight_a.ndim == 3:
        return list(zip(layer.weight_a, layer.weight_b))
    return [(layer.weight_a, layer.weight_b)]


def dense_factors(layer):
    # Ordinary Dense LoRA must not pass through block_diag, even with one pair.
    if type(layer) is LinearWithLoRA:
        return layer.weight_a.detach().float(), layer.weight_b.detach().float()
    pairs = blocks(layer)
    return torch.block_diag(*[a.detach().float() for a, _ in pairs]), torch.block_diag(*[b.detach().float() for _, b in pairs])
