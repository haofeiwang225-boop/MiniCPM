"""Rank-aware bases and singular-energy-weighted coverage (float32)."""
import torch


def orth(matrix):
    matrix = matrix.detach().float()
    if matrix.ndim != 2 or not torch.isfinite(matrix).all():
        raise ValueError('Expected a finite matrix')
    if matrix.shape[1] == 0:
        return matrix
    # Unpivoted QR alone includes arbitrary directions for zero/rank-deficient B.
    # SVD of the small R factor removes these directions without a full-width SVD.
    q, r = torch.linalg.qr(matrix, mode='reduced')
    u, s, _ = torch.linalg.svd(r, full_matrices=False)
    tol = max(matrix.shape) * torch.finfo(matrix.dtype).eps * s.max()
    return q @ u[:, s > tol]


def coverage(q, directions, singular_values):
    energy = singular_values.square()
    if energy.sum() == 0:
        return 1.0  # no nonzero gradient directions to miss
    projection = (q.T @ directions).square().sum(dim=0).clamp(0, 1)
    return float(((energy / energy.sum()) * projection).sum().clamp(0, 1))


def match_subspaces(a, b, u, s, vh):
    qa, qb = orth(a.T), orth(b)
    return qa, qb, coverage(qa, vh.T, s), coverage(qb, u, s)
