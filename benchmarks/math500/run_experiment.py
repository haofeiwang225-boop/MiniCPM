"""Run a controlled MATH-500 subset experiment, separate from the main evaluator."""

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import torch
import transformers
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from .evaluate import load_samples
    from .prompt import build_messages
    from .answer_parser import parse_answer
    from .grader import grade_answer
except ImportError:
    from evaluate import load_samples
    from prompt import build_messages
    from answer_parser import parse_answer
    from grader import grade_answer


def experiment_messages(problem, suffix):
    messages = build_messages(problem)
    if suffix:
        messages[-1]['content'] += '\n\n' + suffix
    return messages


def metrics(rows):
    count = len(rows)
    return {
        'total': count, 'correct': sum(r['correct'] for r in rows),
        'accuracy': sum(r['correct'] for r in rows) / count if count else None,
        'ended_with_eos': sum(r['ended_with_eos'] for r in rows),
        'ended_and_correct': sum(r['ended_with_eos'] and r['correct'] for r in rows),
        'unfinished_thinking': sum(r['parse_reason'] == 'unfinished_thinking' for r in rows),
        'length_limit_reached': sum(r['length_limit_reached'] for r in rows),
        'average_generated_tokens': sum(r['generated_tokens'] for r in rows) / count if count else None,
        'generation_seconds': sum(r['generation_seconds'] for r in rows),
        'parse_reasons': dict(Counter(r['parse_reason'] for r in rows)),
    }


def save_summary(directory, rows, config):
    summary = dict(metrics(rows), experiment=config['experiment'],
                   expected_total=len(config['question_ids']),
                   complete=len(rows) == len(config['question_ids']), diagnostic_only=True)
    summary['strata'] = {name: metrics([r for r in rows if r['stratum'] == name])
                         for name in ('previously_truncated', 'previously_correct')}
    temporary = directory / 'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    temporary.replace(directory / 'summary.json')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if args.output_dir:
        config['output_dir'] = str(args.output_dir)
    ids = config['question_ids']
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 for i in ids):
        parser.error('question_ids must be distinct nonnegative integers')
    if config['enable_thinking'] is not True:
        parser.error('These experiments require enable_thinking=true')
    generation = config['generation']
    if type(generation['max_new_tokens']) is not int or generation['max_new_tokens'] <= 0:
        parser.error('max_new_tokens must be a positive integer')
    strata_ids = config['previously_truncated_ids'] + config['previously_correct_ids']
    if set(strata_ids) != set(ids) or len(strata_ids) != len(ids):
        parser.error('The two strata must partition question_ids')
    samples = load_samples(Path(config['data_path']), question_ids=ids)
    root = Path(__file__).parent
    config['code_sha256'] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ('run_experiment.py', 'evaluate.py', 'prompt.py', 'answer_parser.py', 'grader.py')}
    config['samples_sha256'] = hashlib.sha256(
        json.dumps(samples, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    directory = Path(config['output_dir'])
    rows = []
    if args.resume:
        if yaml.safe_load((directory / 'run_config.yaml').read_text()) != config:
            parser.error('Resume requires identical configuration, samples and code')
        rows = [json.loads(line) for line in (directory / 'results.jsonl').read_text().splitlines() if line.strip()]
        done_ids = [r['question_id'] for r in rows]
        if len(set(done_ids)) != len(done_ids) or not set(done_ids) <= set(ids):
            parser.error('Invalid completed question IDs')
    else:
        directory.mkdir(parents=True, exist_ok=False)
        (directory / 'run_config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
        (directory / 'results.jsonl').touch(exist_ok=False)
    done = {r['question_id'] for r in rows}
    save_summary(directory, rows, config)
    if len(done) == len(ids):
        print('Experiment already complete.')
        return
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    model = AutoModelForCausalLM.from_pretrained(
        config['model'], torch_dtype='auto', device_map='auto')
    model.eval()
    effective = deepcopy(model.generation_config)
    unused = effective.update(**generation)
    if unused:
        raise ValueError(f'Unrecognized generation parameters: {unused}')
    effective.validate()
    runtime = {'generation': effective.to_dict(), 'torch': torch.__version__,
               'transformers': transformers.__version__, 'dtype': str(model.dtype),
               'device': str(model.device),
               'max_position_embeddings': model.config.max_position_embeddings}
    runtime_path = directory / 'runtime.json'
    if args.resume and runtime_path.exists() and json.loads(runtime_path.read_text()) != runtime:
        raise ValueError('Runtime changed; use a new output directory')
    runtime_path.write_text(json.dumps(runtime, indent=2) + '\n', encoding='utf-8')
    eos_ids = effective.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    print(f'Actual generation overrides: {generation}', flush=True)
    with (directory / 'results.jsonl').open('a', encoding='utf-8') as output:
        for sample in samples:
            qid = sample['question_id']
            if qid in done:
                continue
            # Reset per question to avoid earlier outputs changing later RNG states.
            seed = config['seed'] + qid
            set_seed(seed)
            inputs = tokenizer.apply_chat_template(
                experiment_messages(sample['problem'], config['prompt_suffix']),
                tokenize=True, add_generation_prompt=True, enable_thinking=True,
                return_dict=True, return_tensors='pt',
            ).to(model.device)
            prompt_tokens = inputs['input_ids'].shape[-1]
            if prompt_tokens + effective.max_new_tokens > model.config.max_position_embeddings:
                raise ValueError('Input plus generation budget exceeds context window')
            print(f'[{len(rows) + 1}/{len(ids)}] q{qid} prompt_tokens={prompt_tokens}', flush=True)
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                generated = model.generate(**inputs, generation_config=effective)[0][prompt_tokens:]
            if model.device.type == 'cuda':
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            text = tokenizer.decode(generated, skip_special_tokens=False)
            extraction = parse_answer(text, thinking_expected=True, problem=sample['problem'])
            grading = grade_answer(extraction['answer'], sample['answer'], problem=sample['problem'])
            row = {
                'question_id': qid, 'problem': sample['problem'], 'gold_answer': sample['answer'],
                'stratum': 'previously_truncated' if qid in config['previously_truncated_ids'] else 'previously_correct',
                'enable_thinking': True, 'seed': seed,
                'prompt': tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False),
                'prompt_tokens': prompt_tokens, 'model_output': text,
                'parsed_answer': extraction['answer'], 'parse_reason': extraction['reason'],
                'correct': grading['correct'], 'grading': grading,
                'generated_tokens': generated.shape[-1],
                'length_limit_reached': generated.shape[-1] >= effective.max_new_tokens,
                'ended_with_eos': bool(generated.numel() and generated[-1].item() in (eos_ids or [])),
                'generation_seconds': seconds,
            }
            output.write(json.dumps(row, ensure_ascii=False) + '\n')
            output.flush()
            rows.append(row)
            save_summary(directory, rows, config)
            print(f'q{qid}: correct={row["correct"]} tokens={row["generated_tokens"]} '
                  f'eos={row["ended_with_eos"]} parse={row["parse_reason"]}', flush=True)
    print(json.dumps(save_summary(directory, rows, config), indent=2))


if __name__ == '__main__':
    main()
