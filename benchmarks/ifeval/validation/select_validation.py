"""Select a fixed diagnostic subset from saved generations; never load a model."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate import aggregate, load_evaluator, verify_vendor

QUOTAS = {
    'unfinished_thinking': 20,
    'truncated_final': 15,
    'completed_failure': 10,
    'thinking_contamination': 3,
    'stable_correct': 2,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'outputs/full/results.jsonl')
    parser.add_argument('--data', type=Path, default=Path('/home/hongchang/whf/datasets/MiniCPM5_eval/IFEval/ifeval_input_data.jsonl'))
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'validation/diagnostic50')
    parser.add_argument('--nltk_data', default='/home/hongchang/whf/nltk_data')
    args = parser.parse_args()
    source_bytes, data_bytes = args.source.read_bytes(), args.data.read_bytes()
    rows = [json.loads(line) for line in source_bytes.splitlines() if line.strip()]
    docs = [json.loads(line) for line in data_bytes.splitlines() if line.strip()]
    by_key = {row['key']: row for row in docs}
    assert len({row['key'] for row in rows}) == len(rows) == len(by_key) == len(docs)
    evaluator = load_evaluator(args.nltk_data)
    from langdetect import DetectorFactory
    DetectorFactory.seed = 42
    random.seed(42)
    candidates = {group: [] for group in QUOTAS}
    for row in rows:
        doc = by_key[row['key']]
        for field in ('prompt', 'instruction_id_list', 'kwargs'):
            assert row[field] == doc[field], (row['key'], field)
        finished = row['thinking_finished']
        response = row['scored_response']
        final = response.rsplit('</think>', 1)[-1].strip() if finished and '</think>' in response else ''
        scores = evaluator.process_results(doc, [final])
        if not finished:
            group = 'unfinished_thinking'
        elif not row['ended_with_eos']:
            group = 'truncated_final'
        elif not scores['prompt_level_strict_acc']:
            group = 'completed_failure'
        elif not row['scores']['prompt_level_strict_acc']:
            group = 'thinking_contamination'
        else:
            group = 'stable_correct'
        if group in ('unfinished_thinking', 'truncated_final'):
            assert row['length_limit_reached'] and row['generated_tokens'] == 1280
        failed = [inst for inst, correct in zip(doc['instruction_id_list'], scores['inst_level_strict_acc']) if not correct]
        candidates[group].append({
            'key': row['key'], 'group': group, 'prompt': doc['prompt'],
            'instruction_id_list': doc['instruction_id_list'], 'kwargs': doc['kwargs'],
            'generated_tokens': row['generated_tokens'], 'ended_with_eos': row['ended_with_eos'],
            'thinking_finished': finished, 'length_limit_reached': row['length_limit_reached'],
            'raw_scores': row['scores'], 'final_only_scores': scores,
            'failed_final_instructions': failed, 'final_response': final,
        })
    selected = []
    for group, quota in QUOTAS.items():
        pool = list(candidates[group])
        assert len(pool) >= quota, (group, len(pool), quota)
        seen_instructions, seen_failures, seen_lengths = set(), set(), set()
        for _ in range(quota):
            # Cover failure types, then instruction types, then length bands.
            # Stable key tie-break; never select based on future experiment scores.
            def priority(row):
                return (len(set(row['failed_final_instructions']) - seen_failures),
                        len(set(row['instruction_id_list']) - seen_instructions),
                        row['generated_tokens'] // 256 not in seen_lengths,
                        -row['key'])
            chosen = max(pool, key=priority)
            pool.remove(chosen)
            selected.append(chosen)
            seen_failures.update(chosen['failed_final_instructions'])
            seen_instructions.update(chosen['instruction_id_list'])
            seen_lengths.add(chosen['generated_tokens'] // 256)
    selected_keys = {row['key'] for row in selected}
    assert len(selected_keys) == 50
    selected_by_key = {row['key']: row for row in selected}
    # Preserve the dataset's original order, not the grouping/selection order.
    selected_docs = [doc for doc in docs if doc['key'] in selected_keys]
    selected = [selected_by_key[doc['key']] for doc in selected_docs]
    manifest = {
        'purpose': 'Failure-enriched diagnostic subset; not an unbiased full-set accuracy estimate.',
        'source': str(args.source.resolve()), 'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'data': str(args.data.resolve()), 'data_sha256': hashlib.sha256(data_bytes).hexdigest(),
        'selector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'evaluator': verify_vendor(), 'seed': 42, 'langdetect_seed': 42,
        'response_policy': 'After the last </think> if thinking_finished, otherwise empty; upstream scorer unchanged.',
        'population_counts': {group: len(pool) for group, pool in candidates.items()},
        'selected_counts': dict(Counter(row['group'] for row in selected)),
        'selected_keys': [row['key'] for row in selected],
        'keys_by_group': {group: [row['key'] for row in selected if row['group'] == group] for group in QUOTAS},
        'instruction_types_covered': sorted({inst for row in selected for inst in row['instruction_id_list']}),
        'baseline_raw_metrics': aggregate([row['raw_scores'] for row in selected], evaluator),
        'baseline_final_only_metrics': aggregate([row['final_only_scores'] for row in selected], evaluator),
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, records in [('data.jsonl', selected_docs), ('selection.jsonl', selected)]:
        with (args.output_dir / name).open('x', encoding='utf-8') as file:
            for row in records:
                file.write(json.dumps(row, ensure_ascii=False) + '\n')
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    assert hashlib.sha256(args.source.read_bytes()).hexdigest() == manifest['source_sha256']
    print(json.dumps({key: manifest[key] for key in ('population_counts', 'selected_counts', 'keys_by_group', 'instruction_types_covered', 'baseline_final_only_metrics')}, indent=2))


if __name__ == '__main__':
    main()
