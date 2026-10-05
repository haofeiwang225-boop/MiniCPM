"""Ambient BA tangent residual, without a dense identity or projector."""
import torch


def residual_gradient(g, qa, qb):
    left, right = qb.T @ g, g @ qa
    return g - qb @ left - right @ qa.T + qb @ (left @ qa) @ qa.T


def residual_energy_ratio(g, residual, eps=1e-30):
    return float((residual.double().square().sum() / (g.double().square().sum() + eps)).clamp(0, 1))


def block_tangent_residual(g, blocks):
    """Diagnostic for the actual restricted MELoRA tangent; includes off-block G."""
    from .subspace_matching import orth
    result = g.clone()
    row = col = 0
    for a, b in blocks:
        m, n = b.shape[0], a.shape[1]
        result[row:row+m, col:col+n] = residual_gradient(
            g[row:row+m, col:col+n], orth(a.T), orth(b))
        row += m
        col += n
    return result
