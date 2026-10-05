"""Replaceable Euclidean baseline interface; no manifold optimization."""
import torch


def _blend_factor(factor, target_basis, strength):
    # Procrustes chooses a nearby orientation, unlike a fixed B=U replacement.
    # Unlike pure projection, it can move an exactly orthogonal subspace.
    if target_basis.shape[1] == 0:
        return factor
    u, _, vh = torch.linalg.svd(target_basis.T @ factor, full_matrices=False)
    target = target_basis @ (u @ vh)
    norms = factor.norm(dim=0, keepdim=True)
    return (1-strength)*factor + strength*target*norms


def apply_direction_correction(a, b, u, vh, mismatch_side, method='none', strength=0.05):
    if method not in ('none', 'blend') or not 0 <= strength <= 1:
        raise ValueError('Invalid direction correction settings')
    if method == 'none':
        return a, b
    rank = a.shape[0]
    if mismatch_side in ('A', 'both'):
        a = _blend_factor(a.T, vh.T[:, :rank], strength).T
    if mismatch_side in ('B', 'both'):
        b = _blend_factor(b, u[:, :rank], strength)
    return a, b


def compute_B_target_given_A(g, a, rank):
    """Left singular directions not already covered by A's row space."""
    from .subspace_matching import orth
    qa = orth(a.T)
    h = g - (g @ qa) @ qa.T
    u, s, _ = torch.linalg.svd(h.float(), full_matrices=False)
    valid = int((s > max(h.shape)*torch.finfo(s.dtype).eps*s.max()).sum())
    return u[:, :min(rank, valid)]


def compute_A_target_given_B(g, b, rank):
    """Right singular directions not already covered by B's column space."""
    from .subspace_matching import orth
    qb = orth(b)
    h = g - qb @ (qb.T @ g)
    _, s, vh = torch.linalg.svd(h.float(), full_matrices=False)
    valid = int((s > max(h.shape)*torch.finfo(s.dtype).eps*s.max()).sum())
    return vh[:min(rank, valid)].T
