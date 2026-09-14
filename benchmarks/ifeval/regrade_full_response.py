"""Re-evaluate saved raw outputs verbatim; no model/tokenizer/GPU imports."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random

import yaml

from evaluate import METRICS, aggregate, load_evaluator, verify_vendor

ROOT = Path(__file__).resolve().parent


def read_records(path):
    if path.suffix == '.jsonl':
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(rows, dict):
            rows = rows.get('results', rows.get('samples', rows.get('data')))
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f'Unsupported/empty result structure: {path}')
    return rows


def find_source(directory):
    candidates = sorted(set(directory.glob('*result*.jsonl')) | set(directory.glob('*result*.json')))
    if len(candidates) != 1:
        raise ValueError(f'Expected one result JSON/JSONL file, found: {candidates}')
    return candidates[0]


def raw_field(rows):
    candidates = [name for name in ('model_output', 'raw_model_output', 'raw_output')
                  if all(isinstance(row.get(name), str) for row in rows)]
    if not candidates:
        raise ValueError('No raw output field found; refusing to use scored_response as raw output')
    if any(any(row[name] != row[candidates[0]] for row in rows) for name in candidates[1:]):
        raise ValueError('Ambiguous raw output fields')
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', type=Path, default=ROOT / 'outputs/sampling_8192_final')
    parser.add_argument('--output_dir', type=Path, default=ROOT / 'outputs/sampling_8192_full_response_regrade')
    parser.add_argument('--nltk_data', default='/home/hongchang/whf/nltk_data')
    args = parser.parse_args()
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    source = find_source(args.input_dir)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    rows = read_records(source)
    field = raw_field(rows)
    config_path = args.input_dir / 'run_config.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if config.get('response_processing') != 'final_only':
        raise ValueError('Expected final_only source results')
    provenance = verify_vendor()
    if config['evaluator']['files'] != provenance['files'] or config['evaluator']['commit'] != provenance['commit']:
        raise ValueError('Scorer differs from the original run')
    for name in ('nltk', 'langdetect', 'immutabledict'):
        if importlib.metadata.version(name) != config['versions'][name]:
            raise ValueError(f'Scoring dependency version differs: {name}')
    dataset_path = Path(config['data_path']).expanduser()
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if dataset_hash != config['data_sha256']:
        raise ValueError('Dataset hash differs from the original run')
    docs = {row['key']: row for row in read_records(dataset_path)}
    if len({row['key'] for row in rows}) != len(rows):
        raise ValueError('Duplicate sample keys')
    for row in rows:
        for name in ('prompt', 'instruction_id_list', 'kwargs'):
            if row[name] != docs[row['key']][name]:
                raise ValueError(f'Metadata mismatch: {row["key"]}, {name}')
    evaluator = load_evaluator(args.nltk_data or config.get('nltk_data'))
    from langdetect import DetectorFactory
    random.seed(config['seed'])
    DetectorFactory.seed = config.get('langdetect_seed', config['seed'])
    # Never reset the RNG per item: preserve the original sequential scorer setup.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    run_config = dict(config)
    run_config.update(output_dir=str(args.output_dir), response_processing='raw_model_output_verbatim', regrade={
        'source': str(source.resolve()), 'source_sha256': source_hash,
        'source_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'raw_field': field, 'num_samples': len(rows),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'nltk_data': args.nltk_data, 'evaluator': provenance,
        'note': 'No thinking extraction, whitespace stripping, special-token removal, or model generation.',
    })
    (args.output_dir / 'run_config.yaml').write_text(yaml.safe_dump(run_config, allow_unicode=True), encoding='utf-8')
    print(f'Source: {source}; records={len(rows)}; raw field={field}; fields={sorted(rows[0])}', flush=True)
    scores, original_scores = [], []
    transitions = {'true_to_false': 0, 'false_to_true': 0, 'compared_samples': 0}
    with (args.output_dir / 'results.jsonl').open('x', encoding='utf-8') as output:
        for index, row in enumerate(rows):
            raw = row[field]
            score = evaluator.process_results(docs[row['key']], [raw])
            original = row.get('scores')
            if original is not None:
                original_scores.append(original)
                before, after = original['prompt_level_strict_acc'], score['prompt_level_strict_acc']
                transitions['true_to_false'] += bool(before and not after)
                transitions['false_to_true'] += bool(not before and after)
                transitions['compared_samples'] += 1
            scores.append(score)
            record = {name: row[name] for name in ('key', 'prompt', 'instruction_id_list', 'kwargs')}
            record.update(raw_model_output=raw, scored_response=raw,
                          full_response_scores=score, original_final_only_scores=original)
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
            output.flush()
            print(f'[{index + 1}/{len(rows)}] key={row["key"]}', flush=True)
    full = aggregate(scores, evaluator)
    if len(original_scores) == len(rows):
        final = aggregate(original_scores, evaluator)
    else:
        old_summary = json.loads((args.input_dir / 'summary.json').read_text())
        if old_summary['total'] != len(rows):
            raise ValueError('Original summary sample count differs')
        final = {name: old_summary[name] for name in METRICS}
    summary = {'total': len(rows), 'complete': True, 'final_only_metrics': final,
               'full_response_metrics': full,
               'delta_full_minus_final': {name: full[name] - final[name] for name in METRICS},
               'prompt_strict_transitions': transitions}
    if hashlib.sha256(source.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Source changed during re-evaluation')
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
