"""Legacy blend and fixed-rank Grassmann correction with an update trust region."""
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


def _canonical_basis(q):
    """Coordinate-pivot basis of a subspace, used only at the geodesic cut locus.

    At principal angle pi/2 the shortest path is not unique. This deterministic
    ambient-coordinate convention removes dependence on the supplied basis.
    No ambient square projector is formed.
    """
    columns = []
    for _ in range(q.shape[1]):
        residual = q if not columns else q - torch.stack(columns, 1) @ (torch.stack(columns, 1).T @ q)
        norms = residual.square().sum(1)
        pivot = int(torch.nonzero(norms >= norms.max()*(1-1e-10))[0])
        v = residual @ residual[pivot]
        v = v / v.norm()
        columns.append(v)
    return torch.stack(columns, 1)


def grassmann_geodesic(q0, q1, eta):
    """Principal-angle subspace path; output basis orientation is immaterial."""
    if q0.shape != q1.shape or q0.ndim != 2 or not 0 <= eta <= 1:
        raise ValueError('Geodesic requires equal-size bases and eta in [0,1]')
    rank = q0.shape[1]
    if rank == 0 or not torch.isfinite(q0).all() or not torch.isfinite(q1).all():
        raise ValueError('Empty/nonfinite geodesic basis')
    # Reorthogonalize in float64 to avoid acos/sin cancellation near zero angle.
    x0, x1 = q0.double(), q1.double()
    eye = torch.eye(rank, device=q0.device, dtype=torch.float64)
    for q in (x0, x1):
        if int(torch.linalg.matrix_rank(q)) != rank:
            raise ValueError('Geodesic basis is numerically rank deficient')
    x0 = torch.linalg.qr(x0, mode='reduced')[0]
    x1 = torch.linalg.qr(x1, mode='reduced')[0]
    for q in (x0, x1):
        if not torch.allclose(q.T @ q, eye, atol=1e-10, rtol=1e-10):
            raise ArithmeticError('Geodesic reorthogonalization failed')
    if eta == 0:
        return x0.to(q0.dtype)
    if eta == 1:
        return x1.to(q0.dtype)
    u, cosine, vh = torch.linalg.svd(x0.T @ x1, full_matrices=False)
    cosine = cosine.clamp(0, 1)
    x, y = x0 @ u, x1 @ vh.T
    cut = cosine < 1e-10
    if cut.any():
        x[:, cut] = _canonical_basis(x[:, cut])
        y[:, cut] = _canonical_basis(y[:, cut])
        cosine[cut] = 0
    theta = torch.acos(cosine)
    sine = torch.sin(theta)
    moving = sine > 1e-7
    q = x.clone()
    p = (y[:, moving] - x[:, moving]*cosine[moving]) / sine[moving]
    q[:, moving] = x[:, moving]*torch.cos(eta*theta[moving]) + p*torch.sin(eta*theta[moving])
    if not torch.isfinite(q).all() or int(torch.linalg.matrix_rank(q)) != rank:
        raise ArithmeticError('Geodesic lost numerical rank')
    q = torch.linalg.qr(q, mode='reduced')[0]
    if not torch.allclose(q.T @ q, eye, atol=1e-10, rtol=1e-10):
        raise ArithmeticError('Geodesic is not orthonormal')
    return q.to(q0.dtype)


def factor_product_fro_norm(b, a):
    """||BA||_F from r x r Gram matrices; no dense BA allocation."""
    b, a = b.double(), a.double()
    squared = torch.sum((b.T @ b) * (a @ a.T).T)
    if not torch.isfinite(squared):
        raise ArithmeticError('Nonfinite factor product norm')
    return float(squared.clamp_min(0).sqrt())


def relative_B_projection_change(a, b, new_b, eps=1e-12):
    denominator = factor_product_fro_norm(b, a) + eps
    change = factor_product_fro_norm(new_b.double()-b.double(), a)
    return change/denominator, change, denominator


def relative_A_projection_change(a, b, new_a, eps=1e-12):
    denominator = factor_product_fro_norm(b, a) + eps
    change = factor_product_fro_norm(b, new_a.double()-a.double())
    return change/denominator, change, denominator


def _eta_grid(start, decay, minimum):
    if not 0 < minimum <= start <= 1 or not 0 < decay < 1:
        raise ValueError('Invalid eta search interval/decay')
    values = []
    eta = start
    while eta >= minimum*(1-1e-12):
        values.append(max(eta, minimum))
        eta *= decay
        if len(values) > 10000:
            raise ValueError('Eta grid is too large')
    return values


def _tangent_coverage(g, a, b):
    from .subspace_matching import orth
    from .residual_gradient import residual_gradient
    qa, qb = orth(a.T), orth(b)
    residual = residual_gradient(g, qa, qb)
    energy = float(g.double().square().sum())
    coverage = 1-float(residual.double().square().sum())/(energy+1e-30) if energy else 1.
    return coverage, qa.shape[1], qb.shape[1]


def _evaluate_side(side, g, a, b, target, coverage_before, etas, tolerance, min_gain):
    from .subspace_matching import orth
    rank = a.shape[0]
    result = dict(feasible=False, target_rank=target.shape[1], best_eta=None,
                  gain=None, relative_update_change=None, candidate_coverage=None,
                  skip_reason='target_rank_deficient' if target.shape[1] != rank else 'no_feasible_eta',
                  candidates=[])
    if target.shape[1] != rank:
        return result, None
    current = orth(a.T if side == 'A' else b)
    best = None
    for eta in etas:
        q = grassmann_geodesic(current, target, eta)
        new_a = (a @ q) @ q.T if side == 'A' else a
        new_b = q @ (q.T @ b) if side == 'B' else b
        if not torch.isfinite(new_a).all() or not torch.isfinite(new_b).all():
            result['candidates'].append(dict(eta=eta, feasible=False, reason='nonfinite_factor'))
            continue
        coverage, ar, br = _tangent_coverage(g, new_a, new_b)
        relative, change, denominator = (relative_A_projection_change(a,b,new_a) if side == 'A'
                                         else relative_B_projection_change(a,b,new_b))
        gain = coverage-coverage_before
        feasible = ar == rank and br == rank and relative <= tolerance and gain >= min_gain and gain >= 0
        candidate = dict(eta=eta, feasible=feasible, gain=gain, tangent_coverage=coverage,
                         relative_update_change=relative, update_change_norm=change,
                         denominator=denominator, rank_A=ar, rank_B=br)
        result['candidates'].append(candidate)
        # Exact deterministic ordering: maximum gain, minimum perturbation, smaller eta.
        key = (gain, -relative, -eta)
        if feasible and (best is None or key > best[0]):
            best = (key, candidate, (new_a.clone(), new_b.clone()))
    if best is not None:
        c = best[1]
        result.update(feasible=True, best_eta=c['eta'], gain=c['gain'],
                      relative_update_change=c['relative_update_change'],
                      update_change_norm=c['update_change_norm'], denominator=c['denominator'],
                      candidate_coverage=c['tangent_coverage'], skip_reason=None)
        return result, best[2]
    return result, None


def evaluate_A_candidate(g, a, b, coverage_before, etas, tolerance, min_gain):
    target = compute_A_target_given_B(g, b, a.shape[0])
    return _evaluate_side('A', g, a, b, target, coverage_before, etas, tolerance, min_gain)


def evaluate_B_candidate(g, a, b, coverage_before, etas, tolerance, min_gain):
    target = compute_B_target_given_A(g, a, a.shape[0])
    return _evaluate_side('B', g, a, b, target, coverage_before, etas, tolerance, min_gain)


def plan_geometric_direction_correction(g, a, b, oracle_coverage, scaling,
        residual_threshold=.1, function_tolerance=.05, min_gain=1e-3,
        eta_start=1., eta_decay=.5, eta_min=.03125, diagnostic_mismatch_side='none'):
    """Pure fixed-rank planner. Does not mutate factors, optimizer, or random state."""
    import math
    if not all(math.isfinite(x) for x in (oracle_coverage, scaling, residual_threshold,
                                         function_tolerance, min_gain)):
        raise ValueError('Nonfinite correction settings')
    if not 0 <= residual_threshold <= 1 or function_tolerance < 0 or min_gain <= 0:
        raise ValueError('Invalid correction thresholds')
    etas = _eta_grid(eta_start, eta_decay, eta_min)
    coverage, ar, br = _tangent_coverage(g, a, b)
    rank = a.shape[0]
    dg, cg = max(0., oracle_coverage-coverage), max(0., 1-oracle_coverage)
    row = dict(rank=rank, rank_before=rank, rank_after=rank, tangent_coverage_before=coverage,
        tangent_coverage_after=coverage, oracle_tangent_coverage=oracle_coverage,
        direction_gap=dg, capacity_gap=cg, residual_energy_ratio=max(0.,1-coverage),
        diagnostic_mismatch_side=diagnostic_mismatch_side, applied_side='none', applied_eta=None,
        applied_gain=0., applied_relative_update_change=0., absolute_scaled_delta_change=0.,
        correction_applied=False, correction_skip_reason=None,
        update_norm_denominator=factor_product_fro_norm(b,a)+1e-12,
        old_scaling=float(scaling), new_scaling=float(scaling),
        old_total_lora_params=a.numel()+b.numel(), new_total_lora_params=a.numel()+b.numel())
    for side in ('A','B'):
        row.update({side+'_candidate_feasible':False, side+'_target_rank':None,
                    side+'_best_eta':None, side+'_gain':None, side+'_relative_update_change':None})
    if ar != rank or br != rank:
        row['correction_skip_reason']='rank_deficient_factor'
    elif cg >= dg:
        row['correction_skip_reason']='deferred_capacity_no_expansion_in_innovation3'
    elif 1-coverage <= residual_threshold:
        row['correction_skip_reason']='residual_below_threshold'
    elif row['update_norm_denominator'] <= 2e-12:
        row['correction_skip_reason']='near_zero_lora_update'
    if row['correction_skip_reason']:
        return row, None
    candidates = []
    for side, evaluate in (('A',evaluate_A_candidate),('B',evaluate_B_candidate)):
        info, factors = evaluate(g,a,b,coverage,etas,function_tolerance,min_gain)
        for key in ('target_rank','best_eta','gain','relative_update_change'):
            row[side+'_'+key] = info[key]
        row[side+'_candidate_feasible'] = info['feasible']
        row[side+'_search'] = info
        if info['feasible']:
            candidates.append((info['gain'], -info['relative_update_change'], side == 'A', side, info, factors))
    if not candidates:
        row['correction_skip_reason']='no_feasible_candidate'
        return row, None
    _,_,_,side,info,factors = max(candidates, key=lambda c:c[:3])
    row.update(applied_side=side, applied_eta=info['best_eta'], applied_gain=info['gain'],
               applied_relative_update_change=info['relative_update_change'],
               absolute_scaled_delta_change=abs(scaling)*info['update_change_norm'],
               tangent_coverage_after=info['candidate_coverage'], correction_applied=True)
    return row, factors


def apply_geometric_correction(layer, factors, plan, optimizer):
    """In-place mutation of exactly one factor; reset only its optimizer state."""
    side = plan['applied_side']
    if side == 'none':
        return dict(optimizer_state_reset_side='none', optimizer_state_reset_count=0)
    if side not in ('A','B') or not plan['correction_applied'] or factors is None:
        raise ValueError('Invalid geometric plan')
    parameter = layer.weight_a if side == 'A' else layer.weight_b
    new_value = factors[0] if side == 'A' else factors[1]
    if new_value.shape != parameter.shape:
        raise ValueError('Geometric correction cannot resize parameters')
    from .subspace_matching import orth
    candidate_basis = orth(new_value.T if side == 'A' else new_value)
    if not torch.isfinite(new_value).all() or candidate_basis.shape[1] != layer.lora_rank:
        raise ValueError('Correction must retain configured numerical rank')
    if layer.lora_scaler != plan['old_scaling']:
        raise ValueError('Scaling changed since geometric planning')
    with torch.no_grad():
        parameter.copy_(new_value)
    existed = parameter in optimizer.state
    optimizer.state.pop(parameter, None)
    return dict(optimizer_state_reset_side=side, optimizer_state_reset_count=int(existed))
