"""Same singular-value L1 entropy definition as official compute_effective_rank."""
import torch


def gid_from_singular_values(s, eps=1e-10):
    if not torch.isfinite(s).all() or (s < 0).any():
        raise ValueError('Singular values must be finite and nonnegative')
    if s.sum() == 0:
        return 0.0  # residual has no directions; avoids the upstream zero/zero
    p = s / s.sum()
    return max(1.0, float(torch.exp(-(p * torch.log(p + eps)).sum())))


def svd_with_gid(g):
    u, s, vh = torch.linalg.svd(g.float(), full_matrices=False)
    return u, s, vh, gid_from_singular_values(s)
