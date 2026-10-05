"""Dense BA tangent diagnostics and same-width spectral oracle (energy, not GID)."""
import warnings
import torch


def tangent_metrics(g, residual, singular_values, rank, coverage_A, coverage_B,
                    eps=1e-30, tolerance=2e-5):
    # Double precision reductions keep the check meaningful for small gradients.
    energy = float(g.double().square().sum())
    if energy == 0:
        return dict(tangent_coverage=1., tangent_coverage_from_residual=1.,
                    residual_energy_ratio=0., oracle_tangent_coverage=1.,
                    direction_efficiency=1., coverage_A_oracle=1., coverage_B_oracle=1.,
                    efficiency_A=1., efficiency_B=1., decomposition_error=0.,
                    coverage_consistency_error=0., oracle_excess=0., k_oracle=0,
                    valid_singular_values=0, zero_gradient=True)
    tangent = g - residual
    ct = float(tangent.double().square().sum()) / (energy + eps)
    rho = float(residual.double().square().sum()) / (energy + eps)
    consistency = abs(ct - (1-rho))
    if consistency > tolerance:
        raise ArithmeticError(f'Tangent decomposition failed: C_T+rho={ct+rho}')
    s = singular_values.double()
    if not torch.isfinite(s).all() or (s < 0).any():
        raise ValueError('Invalid singular spectrum')
    e = s.square()
    total = float(e.sum())
    valid = int((s > max(g.shape)*torch.finfo(singular_values.dtype).eps*s.max()).sum())
    k = min(2*rank, valid)
    oracle = float(e[:k].sum()) / (total+eps)
    side_oracle = float(e[:min(rank, valid)].sum()) / (total+eps)
    efficiency = ct/(oracle+eps)
    excess = max(0., efficiency-1.)
    if excess > tolerance:
        warnings.warn(f'Tangent coverage exceeds same-rank oracle: {efficiency=}')
    return dict(tangent_coverage=max(0., min(1., ct)),
                tangent_coverage_from_residual=1-rho,
                residual_energy_ratio=max(0., min(1., rho)),
                oracle_tangent_coverage=oracle,
                direction_efficiency=max(0., min(1+tolerance, efficiency)),
                coverage_A_oracle=side_oracle, coverage_B_oracle=side_oracle,
                efficiency_A=coverage_A/(side_oracle+eps),
                efficiency_B=coverage_B/(side_oracle+eps),
                decomposition_error=abs(energy-float(tangent.double().square().sum())-
                                        float(residual.double().square().sum()))/(energy+eps),
                coverage_consistency_error=consistency, oracle_excess=excess,
                k_oracle=k, valid_singular_values=valid, zero_gradient=False)


def residual_energy_rank(singular_values, target=0.90):
    """Smallest k explaining target fraction of residual squared singular values."""
    if not 0 < target <= 1:
        raise ValueError('Residual energy target must be in (0, 1]')
    s = singular_values.double()
    if not torch.isfinite(s).all() or (s < 0).any():
        raise ValueError('Invalid residual spectrum')
    e = s.sort(descending=True).values.square()
    total = e.sum()
    if total == 0:
        return 0
    return min(len(e), int(torch.searchsorted(e.cumsum(0), total*target))+1)


def frobenius_metrics(g, residual, residual_ratio, eps=1e-30, tolerance=2e-5):
    """Absolute norms of averaged G, alongside relative sanity checks."""
    import math
    gradient_norm = float(torch.linalg.vector_norm(g.double()))
    residual_norm = float(torch.linalg.vector_norm(residual.double()))
    tangent_norm = float(torch.linalg.vector_norm((g-residual).double()))
    error = abs(residual_norm/(gradient_norm+eps)-math.sqrt(residual_ratio))
    energy_error = abs(gradient_norm**2-tangent_norm**2-residual_norm**2)
    relative_error = energy_error/(gradient_norm**2+eps)
    if error > tolerance or relative_error > tolerance:
        raise ArithmeticError('Frobenius norm consistency check failed')
    return dict(gradient_fro_norm=gradient_norm, residual_fro_norm=residual_norm,
                tangent_fro_norm=tangent_norm, residual_norm_consistency_error=error,
                energy_decomposition_error=energy_error,
                relative_energy_decomposition_error=relative_error)
