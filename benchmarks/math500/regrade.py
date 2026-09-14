"""Regrade saved generations without loading a model or modifying source results."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import yaml

try:
    from .answer_parser import parse_answer
    from .grader import grade_answer
except ImportError:
    from answer_parser import parse_answer
    from grader import grade_answer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Original results.jsonl')
    parser.add_argument('--output_dir', type=Path, required=True, help='New, nonexistent directory')
    parser.add_argument('--workers', type=int, default=4, help='Concurrent isolated grading workers')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('workers must be positive')
    source_bytes = args.input.read_bytes()
    rows = [json.loads(line) for line in source_bytes.decode('utf-8').splitlines() if line.strip()]
    if not rows or len({r['question_id'] for r in rows}) != len(rows):
        raise ValueError('Input must contain nonempty results with unique question IDs')
    config_path = args.input.parent / 'run_config.yaml'
    config = yaml.safe_load(config_path.read_text(encoding='utf-8')) if config_path.exists() else {}
    if any('enable_thinking' not in r for r in rows) and 'enable_thinking' not in config:
        raise ValueError('Thinking mode is required in each row or run_config.yaml')
    old_summary_path = args.input.parent / 'summary.json'
    old_summary = json.loads(old_summary_path.read_text()) if old_summary_path.exists() else {}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config['regrade'] = {
        'source': str(args.input.resolve()),
        'source_sha256': hashlib.sha256(source_bytes).hexdigest(),
        'code_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                        for name in ('answer_parser.py', 'grader.py', 'regrade.py')},
    }
    (args.output_dir / 'run_config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    def regrade_row(old):
        extraction = parse_answer(
            old['model_output'], problem=old['problem'],
            thinking_expected=old.get('enable_thinking', config.get('enable_thinking')))
        grading = grade_answer(extraction['answer'], old['gold_answer'], problem=old['problem'])
        return dict(old, parsed_answer=extraction['answer'], parse_reason=extraction['reason'],
                    grading=grading, correct=grading['correct'])

    results, changes = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor, \
            (args.output_dir / 'comparisons.jsonl').open('x', encoding='utf-8') as comparisons, \
            (args.output_dir / 'results.jsonl').open('x', encoding='utf-8') as output, \
            (args.output_dir / 'failures.jsonl').open('x', encoding='utf-8') as failures, \
            (args.output_dir / 'changes.jsonl').open('x', encoding='utf-8') as change_file:
        for index, (old, result) in enumerate(zip(rows, executor.map(regrade_row, rows))):
            comparison = {'question_id': old['question_id'], 'gold_answer': old['gold_answer'],
                          'before': {k: old.get(k) for k in ('correct', 'parsed_answer', 'parse_reason', 'grading')},
                          'after': {k: result[k] for k in ('correct', 'parsed_answer', 'parse_reason', 'grading')}}
            comparisons.write(json.dumps(comparison, ensure_ascii=False) + '\n')
            comparisons.flush()
            output.write(json.dumps(result, ensure_ascii=False) + '\n')
            output.flush()
            if not result['correct']:
                failures.write(json.dumps(result, ensure_ascii=False) + '\n')
                failures.flush()
            if old['correct'] != result['correct'] or old['parsed_answer'] != result['parsed_answer']:
                change = {'question_id': old['question_id'], 'gold_answer': old['gold_answer'],
                          'before': {k: old.get(k) for k in ('correct', 'parsed_answer', 'parse_reason', 'grading')},
                          'after': {k: result[k] for k in ('correct', 'parsed_answer', 'parse_reason', 'grading')}}
                changes.append(change)
                change_file.write(json.dumps(change, ensure_ascii=False) + '\n')
                change_file.flush()
            results.append(result)
            print(f'[{index + 1}/{len(rows)}] question_id={old["question_id"]} '
                  f'{old["correct"]} -> {result["correct"]}', flush=True)
    total = len(results)
    correct = sum(r['correct'] for r in results)
    summary = dict(old_summary)
    summary.update({
        'total': total, 'correct': correct, 'accuracy': correct / total,
        'regrade_complete': True,
        'false_to_true_count': sum(not old['correct'] and new['correct'] for old, new in zip(rows, results)),
        'true_to_false_count': sum(old['correct'] and not new['correct'] for old, new in zip(rows, results)),
        'original_correct': sum(r['correct'] for r in rows),
        'original_accuracy': sum(r['correct'] for r in rows) / total,
        'false_to_true': [c['question_id'] for c in changes if not c['before']['correct'] and c['after']['correct']],
        'true_to_false': [c['question_id'] for c in changes if c['before']['correct'] and not c['after']['correct']],
        'failed_question_ids': [r['question_id'] for r in results if not r['correct']],
        'parse_reasons': dict(Counter(r['parse_reason'] for r in results)),
        'grading_reasons': dict(Counter(r['grading']['reason'] for r in results)),
        'unfinished_thinking': sum(r['parse_reason'] == 'unfinished_thinking' for r in results),
    })
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    if hashlib.sha256(args.input.read_bytes()).hexdigest() != config['regrade']['source_sha256']:
        raise RuntimeError('Source results changed during regrading; inspect the run before using it')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
