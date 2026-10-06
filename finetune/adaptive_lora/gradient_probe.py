"""Second probe: one target weight at a time, restoring all training state."""
import torch
from .official import get_record_gradient_post_accumulate_hook


def probe_weight(model, layer, batches, return_losses=False):
    params = list(model.parameters())
    flags = [p.requires_grad for p in params]
    grads = [p.grad for p in params]
    modes = [(m, m.training) for m in model.modules()]
    weight = layer.weight
    if hasattr(weight, 'grad_stored') or hasattr(weight, 'iters'):
        raise ValueError('Stale initial-probe gradient state')
    original = weight.detach().cpu().clone()
    hook = None
    batch_losses = []
    devices = sorted({p.device.index for p in params if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()  # deterministic calibration: disable dropout, keep adapters active
            for p in params:
                p.requires_grad_(False)
                p.grad = None
            weight.requires_grad_(True)
            hook = weight.register_post_accumulate_grad_hook(get_record_gradient_post_accumulate_hook())
            for batch in batches:
                loss = model(**{k: v.to(weight.device) for k, v in batch.items()}).loss
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite calibration loss')
                batch_losses.append(float(loss.detach()))
                loss.backward()
                # Preserve FP32 accumulation across batches even for BF16 base weights.
                if hasattr(weight, "grad_stored"):
                    weight.grad_stored = weight.grad_stored.float()
            if not hasattr(weight, 'iters') or weight.iters != len(batches):
                raise RuntimeError('Target did not receive a gradient on every batch')
            gradient = weight.grad_stored.float() / weight.iters
            if not torch.equal(weight.detach().cpu(), original):
                raise RuntimeError('Probe modified base weight')
            return (gradient, batch_losses) if return_losses else gradient
    finally:
        if hook is not None:
            hook.remove()
        for name in ('grad_stored', 'iters'):
            if hasattr(weight, name):
                delattr(weight, name)
        for p, flag, grad in zip(params, flags, grads):
            p.requires_grad_(flag)
            p.grad = grad
        for module, mode in modes:
            module.training = mode


def evaluate_calibration_loss(model, batches):
    """Eval/no-grad diagnostic, restoring per-module modes and CPU/CUDA RNG."""
    if not batches:
        raise ValueError('Calibration loss requires at least one batch')
    modes = [(module, module.training) for module in model.modules()]
    params = list(model.parameters())
    device = params[0].device
    devices = sorted({p.device.index for p in params if p.is_cuda})
    losses = []
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            model.eval()
            for batch in batches:
                loss = model(**{k:v.to(device) for k,v in batch.items()}).loss
                if loss.numel() != 1 or not torch.isfinite(loss).all():
                    raise RuntimeError('Nonfinite or nonscalar calibration diagnostic loss')
                losses.append(float(loss.detach()))
    finally:
        for module, mode in modes:
            module.training = mode
    return losses, sum(losses)/len(losses)
