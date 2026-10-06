import json
import statistics
from collections import Counter
from pathlib import Path


def write_diagnostics(path, rows, old_params, new_params):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with (path / 'modules.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    counts = Counter(r['decision'] for r in rows)
    summary = {
        'average_coverage_A': sum(r['coverage_A'] for r in rows)/len(rows),
        'average_coverage_B': sum(r['coverage_B'] for r in rows)/len(rows),
        'average_residual_energy': sum(r['residual_energy_ratio'] for r in rows)/len(rows),
        **{'num_'+key: counts[key] for key in ('keep', 'direction_mismatch', 'expand')},
        'old_total_lora_params': old_params, 'new_total_lora_params': new_params,
        'old_rank_distribution': dict(Counter(r['old_rank'] for r in rows)),
        'new_rank_distribution': dict(Counter(r['applied_new_rank'] for r in rows)),
        'rank_convention': 'dense factor width r + appended_dense_rank' if all(r.get('structure') == 'dense' for r in rows) else 'legacy block algebraic width',
    }
    for key in ('coverage_A', 'coverage_B', 'tangent_coverage', 'oracle_tangent_coverage',
                'direction_efficiency', 'efficiency_A', 'efficiency_B',
                'residual_energy_ratio', 'gid_full', 'gid_residual', 'residual_rank_k90',
                'gradient_fro_norm', 'tangent_fro_norm', 'residual_fro_norm'):
        summary['mean_'+key] = sum(r[key] for r in rows)/len(rows)
    summary['max_coverage_consistency_error'] = max(r['coverage_consistency_error'] for r in rows)
    summary['max_oracle_excess'] = max(r['oracle_excess'] for r in rows)
    summary['all_base_weights_unchanged'] = all(r['base_weight_unchanged'] for r in rows)
    summary['diagnostic_only'] = all(r['diagnostic_only'] for r in rows)
    summary['num_modules'] = len(rows)
    summary['median_gradient_fro_norm'] = statistics.median(r['gradient_fro_norm'] for r in rows)
    for key in ('residual_norm_consistency_error', 'energy_decomposition_error',
                'relative_energy_decomposition_error'):
        summary['max_'+key] = max(r[key] for r in rows)
    ge = sum(r['gradient_fro_norm']**2 for r in rows)
    re = sum(r['residual_fro_norm']**2 for r in rows)
    summary.update(global_gradient_energy=ge, global_residual_energy=re,
                   global_residual_energy_ratio=re/(ge+1e-30))
    # Same eval-mode model/data loss at every layer; use one pass, not 84 repeats.
    for key in ('probe_batch_losses', 'mean_probe_loss', 'num_calibration_batches',
                'calibration_indices_path', 'calibration_indices_sha256'):
        summary[key] = rows[0][key]
    summary['probe_loss_source_module'] = rows[0]['module']
    summary['max_probe_batch_loss_difference_across_modules'] = max(
        abs(a-b) for r in rows for a,b in zip(r['probe_batch_losses'], rows[0]['probe_batch_losses']))
    (path / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (path / 'rank_pattern.json').write_text(json.dumps({r['module']: r['applied_new_rank'] for r in rows}, indent=2)+'\n')
    return summary


def write_geometric_diagnostics(path, rows, summary):
    """Separate event format; innovation-1 modules/summary are unchanged."""
    path = Path(path)
    path.mkdir(parents=True,exist_ok=False)
    with (path/'modules.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
    summary.update(num_modules=len(rows),
        applied_sides=dict(Counter(r['applied_side'] for r in rows)),
        skip_reasons=dict(Counter(r['correction_skip_reason'] for r in rows if r['correction_skip_reason'])),
        optimizer_state_reset_count=sum(r['optimizer_state_reset_count'] for r in rows),
        mean_tangent_coverage_before=sum(r['tangent_coverage_before'] for r in rows)/len(rows),
        mean_tangent_coverage_after=sum(r['tangent_coverage_after'] for r in rows)/len(rows),
        max_relative_update_change=max(r['applied_relative_update_change'] for r in rows),
        all_base_weights_unchanged=all(r['base_weight_unchanged'] for r in rows))
    import math
    energy = math.fsum(float(r['gradient_fro_norm'])**2 for r in rows)
    before = math.fsum(float(r['gradient_fro_norm'])**2 * float(r['tangent_coverage_before']) for r in rows)
    after = math.fsum(float(r['gradient_fro_norm'])**2 * float(r['tangent_coverage_after']) for r in rows)
    ct_before = before/(energy+1e-30) if energy else 1.0
    ct_after = after/(energy+1e-30) if energy else 1.0
    summary.update(global_gradient_energy=energy,
                   global_tangent_energy_before=before, global_tangent_energy_after=after,
                   global_tangent_coverage_before=ct_before, global_tangent_coverage_after=ct_after,
                   global_tangent_coverage_gain=ct_after-ct_before)
    (path/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary
