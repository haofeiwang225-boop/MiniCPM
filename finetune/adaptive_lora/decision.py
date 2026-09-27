"""Decision and allocation policies, independent of training/optimizer code."""
import math


def allocate_rank(gid_residual, current_rank, max_rank, step_cap):
    return min(max(0, math.ceil(gid_residual)), step_cap, max(0, max_rank-current_rank))


def make_adaptive_decision(coverage_A, coverage_B, residual_energy_ratio,
                           gid_full, gid_residual, current_rank, max_rank,
                           alignment_threshold=0.6, energy_threshold=0.1,
                           step_cap=4, min_residual_gid=1.0):
    values = (coverage_A, coverage_B, residual_energy_ratio, gid_full, gid_residual)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite diagnostics')
    if not 0 <= alignment_threshold <= 1 or not 0 <= energy_threshold <= 1:
        raise ValueError('Thresholds must be in [0,1]')
    if current_rank < 0 or max_rank < 1 or step_cap < 1:
        raise ValueError('Invalid rank limits')
    low_a, low_b = coverage_A < alignment_threshold, coverage_B < alignment_threshold
    side = 'both' if low_a and low_b else 'A' if low_a else 'B' if low_b else 'none'
    decision, delta = 'keep', 0
    if gid_full == 0:
        side = 'none'
    elif side != 'none':
        decision = 'direction_mismatch'
    elif residual_energy_ratio > energy_threshold and gid_residual >= min_residual_gid:
        delta = allocate_rank(gid_residual, current_rank, max_rank, step_cap)
        if delta:
            decision = 'expand'
    return dict(decision=decision, mismatch_side=side, delta_rank=delta,
                new_rank=current_rank+delta)
