import json
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
        'new_rank_distribution': dict(Counter(r['new_rank'] for r in rows)),
        'rank_convention': 'total algebraic factor width: mini_rank*n_split + appended_dense_rank',
    }
    (path / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    (path / 'rank_pattern.json').write_text(json.dumps({r['module']: r['new_rank'] for r in rows}, indent=2)+'\n')
    return summary
