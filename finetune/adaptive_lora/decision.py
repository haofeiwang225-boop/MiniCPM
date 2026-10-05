"""Oracle-relative policy. Individual coverage diagnoses side only."""
import math


def allocate_rank(residual_rank, current_rank, max_rank, step_cap):
    if int(residual_rank) != residual_rank or residual_rank < 0:
        raise ValueError('Rank allocation requires an integer energy-threshold rank')
    return min(int(residual_rank), step_cap, max(0, max_rank-current_rank))


def make_adaptive_decision(tangent_coverage, oracle_tangent_coverage,
                           direction_efficiency, coverage_A, coverage_B,
                           residual_energy_ratio, gid_residual, current_rank, max_rank,
                           residual_rank, efficiency_A, efficiency_B,
                           efficiency_threshold=0.8, oracle_threshold=0.9,
                           energy_threshold=0.1, step_cap=4, side_gap=0.15):
    values = (tangent_coverage, oracle_tangent_coverage, direction_efficiency,
              coverage_A, coverage_B, residual_energy_ratio, gid_residual,
              efficiency_A, efficiency_B)
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite diagnostics')
    if any(not 0 <= x <= 1 for x in (efficiency_threshold, oracle_threshold,
                                    energy_threshold, side_gap)):
        raise ValueError('Thresholds must be in [0,1]')
    if current_rank < 0 or max_rank < 1 or step_cap < 1:
        raise ValueError('Invalid rank limits')
    decision, side, delta = 'keep', 'none', 0
    if residual_energy_ratio <= energy_threshold:
        reason = 'residual_already_small'
    elif direction_efficiency < efficiency_threshold:
        decision = 'direction_mismatch'
        reason = ('direction_gap' if oracle_tangent_coverage >= oracle_threshold
                  else 'mixed_direction_and_capacity_direction_first')
        # Relative oracle utilization gap; complementary coverage is NOT penalized
        # when the joint direction efficiency is already sufficient.
        if efficiency_A + side_gap < efficiency_B:
            side = 'A'
        elif efficiency_B + side_gap < efficiency_A:
            side = 'B'
        else:
            side = 'both'
    elif oracle_tangent_coverage < oracle_threshold:
        delta = allocate_rank(residual_rank, current_rank, max_rank, step_cap)
        decision = 'expand' if delta else 'keep'
        reason = 'capacity_gap' if delta else 'rank_budget_or_residual_exhausted'
    else:
        reason = 'near_oracle_residual_above_target_deferred'
    return dict(decision=decision, mismatch_side=side, delta_rank=delta,
                new_rank=current_rank+delta, decision_reason=reason)
