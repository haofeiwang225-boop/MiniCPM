"""GID-RSC transaction primitives. V1 geometry is imported, never redefined."""
from contextlib import contextmanager
import math
import random
import numpy as np
import torch
from .direction_correction import relative_A_projection_change, relative_B_projection_change
from .subspace_matching import orth


@contextmanager
def preserved_training_state(model):
    """Freeze stochastic forward behavior; restore grad objects, flags, modes and RNG."""
    params = list(model.parameters())
    states = [(p.requires_grad, p.grad) for p in params]
    modes = [(m, m.training) for m in model.modules()]
    buffers = [(b, b.detach().clone()) for b in model.buffers()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    devices = sorted({p.device.index for p in params if p.is_cuda})
    try:
        with torch.random.fork_rng(devices=devices):
            try:
                model.eval()
                yield
            finally:
                for p, (flag, grad) in zip(params, states):
                    p.requires_grad_(flag)
                    p.grad = grad
                for module, mode in modes:
                    module.training = mode
                with torch.no_grad():
                    for buffer, value in buffers:
                        if not torch.equal(buffer, value):
                            buffer.copy_(value)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def validate_samples(batches):
    if not batches:
        raise ValueError('RSC requires nonempty per-sample batches')
    for batch in batches:
        if not batch or any(not torch.is_tensor(v) or v.ndim == 0 or v.shape[0] != 1 for v in batch.values()):
            raise ValueError('RSC loss semantics require batch_size=1 for every sample')


def sample_loss(model, batch):
    device = next(model.parameters()).device
    loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
    if loss.numel() != 1 or not torch.isfinite(loss).all():
        raise ArithmeticError('Nonfinite/nonscalar RSC sample loss')
    return loss


def mean_sample_loss(model, batches):
    validate_samples(batches)
    with preserved_training_state(model), torch.no_grad():
        losses = [float(sample_loss(model, b)) for b in batches]
    return dict(sample_losses=losses, mean_loss=math.fsum(losses)/len(losses))


def factor_directional_derivatives(model, candidates, batches):
    """Exact autograd factor derivatives, averaged over samples, all at PRE.

    candidates contain only one selected factor per layer. Scaling is already
    in autograd via the unchanged forward s*BA; do not multiply by s again.
    """
    validate_samples(batches)
    if not candidates:
        return {}
    params = [c['parameter'] for c in candidates]
    if len(set(params)) != len(params):
        raise ValueError('Duplicate selected factor')
    values = {c['module']: [] for c in candidates}
    with preserved_training_state(model), torch.enable_grad():
        for p in model.parameters():
            p.requires_grad_(False)
        for p in params:
            p.requires_grad_(True)
        for batch in batches:
            gradients = torch.autograd.grad(sample_loss(model, batch), params)
            for c, g in zip(candidates, gradients):
                # Use the actually representable target in the parameter's dtype.
                delta = c['target'].to(c['parameter']).double() - c['parameter'].detach().double()
                value = float((g.detach().double()*delta).sum())
                if not math.isfinite(value):
                    raise ArithmeticError('Nonfinite RSC directional derivative')
                values[c['module']].append(value)
    return {name: dict(per_sample=ds, mean=math.fsum(ds)/len(ds)) for name, ds in values.items()}


def passes_direction_check(proposal, selection, proposal_margin=0., selection_margin=0.):
    if any(not math.isfinite(x) or x < 0 for x in (proposal_margin, selection_margin)):
        raise ValueError('Descent margins must be finite and nonnegative')
    return (math.isfinite(proposal) and math.isfinite(selection)
            and proposal < -proposal_margin and selection < -selection_margin)


def trial_factors(candidates, strength, tolerance):
    """Factor-space chord interpolation, NOT another Grassmann geodesic."""
    if not 0 < strength <= 1 or not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError('Invalid RSC trial strength/trust tolerance')
    values, checks = [], []
    for c in candidates:
        layer, p = c['layer'], c['parameter']
        old = p.detach()
        new = (old.double() + strength*(c['target'].to(p).double()-old.double())).to(p.dtype)
        a = new if c['side'] == 'A' else layer.weight_a.detach()
        b = new if c['side'] == 'B' else layer.weight_b.detach()
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        check = dict(module=c['module'], side=c['side'], finite=finite, feasible=False)
        if finite:
            ar, br = orth(a.T).shape[1], orth(b).shape[1]
            fn = relative_A_projection_change if c['side'] == 'A' else relative_B_projection_change
            relative, change, denominator = fn(layer.weight_a.detach(), layer.weight_b.detach(), new)
            feasible = ar == br == layer.lora_rank and math.isfinite(relative) and relative <= tolerance
            check.update(rank_A=ar, rank_B=br, relative_update_change=relative,
                         absolute_scaled_delta_change=abs(layer.lora_scaler)*change,
                         denominator=denominator, feasible=feasible, changed=not torch.equal(new, old))
        check['reason'] = None if check['feasible'] else 'nonfinite_rank_or_trust_region_failure'
        checks.append(check)
        values.append(new.detach().clone())
    return values, checks


@contextmanager
def temporary_factors(candidates, values):
    """Always roll back trial copies, including partial-copy/forward failures."""
    originals = [(c['parameter'], c['parameter'].detach().clone()) for c in candidates]
    try:
        with torch.no_grad():
            for c, value in zip(candidates, values):
                c['parameter'].copy_(value)
        yield
    finally:
        with torch.no_grad():
            for p, old in originals:
                p.copy_(old)


def classify_loss_change(before, after, min_improvement=0.):
    """Compare observed losses exactly; quantized equality is not improvement."""
    if not all(math.isfinite(x) for x in (before, after, min_improvement)) or min_improvement < 0:
        raise ValueError('Losses and nonnegative improvement margin must be finite')
    improvement = before - after
    status = ('numerically_equal' if after == before else
              'loss_increase' if after > before else 'strict_improvement')
    return dict(status=status, improvement=improvement,
                accepted=status == 'strict_improvement' and improvement >= min_improvement,
                min_improvement=min_improvement)


def joint_loss_search(model, candidates, proposal, selection, strengths, tolerance,
                      min_improvement=0.):
    if not strengths or len(strengths) > 8 or any(not math.isfinite(s) or not 0 < s <= 1 for s in strengths) or any(a <= b for a, b in zip(strengths, strengths[1:])):
        raise ValueError('Use 1..8 strictly decreasing finite trial strengths in (0,1]')
    if not math.isfinite(min_improvement) or min_improvement < 0:
        raise ValueError('Loss improvement threshold must be finite and nonnegative')
    before = dict(proposal=mean_sample_loss(model, proposal), selection=mean_sample_loss(model, selection))
    attempts = []
    if not candidates:
        return None, dict(before=before, after=before, attempts=attempts, accepted_strength=None,
                          reason='no_direction_consistent_candidates')
    for strength in strengths:
        values, checks = trial_factors(candidates, strength, tolerance)
        attempt = dict(strength=strength, interpolation='factor_linear_not_grassmann', checks=checks,
                       accepted=False)
        attempts.append(attempt)
        if not all(c['feasible'] for c in checks) or not any(c.get('changed') for c in checks):
            attempt['reason'] = 'invalid_or_zero_trial'
            continue
        with temporary_factors(candidates, values):
            after = dict(proposal=mean_sample_loss(model, proposal), selection=mean_sample_loss(model, selection))
        attempt['losses'] = after
        changes = {k: classify_loss_change(before[k]['mean_loss'], after[k]['mean_loss'],
                                          min_improvement) for k in before}
        accepted = all(change['accepted'] for change in changes.values())
        rejected = [change['status'] for change in changes.values() if not change['accepted']]
        reason = ('accepted' if accepted else 'loss_increase' if 'loss_increase' in rejected else
                  'numerically_equal' if 'numerically_equal' in rejected else 'below_min_improvement')
        attempt.update(accepted=accepted, reason=reason, loss_changes=changes)
        if accepted:
            return values, dict(before=before, after=after, attempts=attempts,
                                accepted_strength=strength, reason='accepted')
    return None, dict(before=before, after=before, attempts=attempts, accepted_strength=None,
                      reason='no_joint_trial_accepted')
