"""Compare complete experiments on exactly the same MATH-500 subset."""

import argparse
import json
from pathlib import Path

import yaml


def compare(directories):
    experiments = []
    reference = None
    for directory in directories:
        config = yaml.safe_load((directory / 'run_config.yaml').read_text())
        summary = json.loads((directory / 'summary.json').read_text())
        rows = [json.loads(line) for line in (directory / 'results.jsonl').read_text().splitlines() if line.strip()]
        if not summary['complete'] or len(rows) != len(config['question_ids']):
            raise ValueError(f'Experiment incomplete: {directory}')
        if {r['question_id'] for r in rows} != set(config['question_ids']):
            raise ValueError(f'Question IDs differ from configuration: {directory}')
        common = {key: config[key] for key in (
            'model', 'samples_sha256', 'seed', 'question_ids', 'previously_truncated_ids',
            'previously_correct_ids', 'enable_thinking', 'code_sha256')}
        runtime = json.loads((directory / 'runtime.json').read_text())
        common['runtime'] = {k: runtime[k] for k in ('torch', 'transformers', 'dtype', 'device')}
        if reference is not None and common != reference:
            raise ValueError(f'Different data, model, seed, code or runtime: {directory}')
        reference = common
        experiments.append({'directory': str(directory), 'config': config, 'summary': summary, 'rows': rows})
    comparison = {'diagnostic_only': True, 'experiments': [
        {'experiment': e['config']['experiment'], 'generation': e['config']['generation'],
         'prompt_suffix': e['config']['prompt_suffix'], 'summary': e['summary']} for e in experiments],
        'questions': {}}
    for qid in reference['question_ids']:
        comparison['questions'][qid] = {}
        for e in experiments:
            row = next(r for r in e['rows'] if r['question_id'] == qid)
            comparison['questions'][qid][e['config']['experiment']] = {
                k: row[k] for k in ('correct', 'ended_with_eos', 'parse_reason', 'generated_tokens', 'stratum')}
    return comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories', type=Path, nargs='+')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = compare(args.directories)
    print('| Experiment | Accuracy | EOS | Unfinished | Avg tokens | Formerly truncated: EOS + correct | Controls: correct |')
    print('|---|---:|---:|---:|---:|---:|---:|')
    for experiment in result['experiments']:
        s = experiment['summary']
        bad = s['strata']['previously_truncated']
        good = s['strata']['previously_correct']
        print(f'| {experiment["experiment"]} | {s["correct"]}/{s["total"]} ({s["accuracy"]:.1%}) '
              f'| {s["ended_with_eos"]} | {s["unfinished_thinking"]} | {s["average_generated_tokens"]:.1f} '
              f'| {bad["ended_and_correct"]}/{bad["total"]} | {good["correct"]}/{good["total"]} |')
    if args.output:
        with args.output.open('x', encoding='utf-8') as output:
            json.dump(result, output, indent=2, ensure_ascii=False)
            output.write('\n')


if __name__ == '__main__':
    main()
