"""Diagnose truncated thinking by continuing saved text with an extra token budget.

This reconstructs tokens from saved text and resets RNG per question. It is not
an exact replay of a single uninterrupted generation with a larger budget.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

try:
    from .prompt import build_messages
    from .answer_parser import parse_answer
    from .grader import grade_answer
except ImportError:
    from prompt import build_messages
    from answer_parser import parse_answer
    from grader import grade_answer


def summarize(rows, expected):
    return {
        'total': len(rows), 'expected_total': expected, 'complete': len(rows) == expected,
        'ended_with_eos': sum(r['ended_with_eos'] for r in rows),
        'thinking_closed': sum(r['thinking_closed'] for r in rows),
        'correct': sum(r['correct'] for r in rows),
        'ended_and_correct': sum(r['ended_with_eos'] and r['correct'] for r in rows),
        'still_unfinished_thinking': sum(r['parse_reason'] == 'unfinished_thinking' for r in rows),
        'extra_limit_reached': sum(r['extra_limit_reached'] for r in rows),
        'outcomes': dict(Counter(r['outcome'] for r in rows)),
        'diagnostic_only': True,
    }


def write_summary(output_dir, rows, expected):
    summary = summarize(rows, expected)
    temporary = output_dir / 'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    temporary.replace(output_dir / 'summary.json')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--extra_tokens', type=int, default=8192)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.extra_tokens <= 0:
        parser.error('extra_tokens must be positive')
    source = args.input.read_bytes()
    source_rows = [json.loads(line) for line in source.decode('utf-8').splitlines() if line.strip()]
    config = yaml.safe_load((args.input.parent / 'run_config.yaml').read_text())
    selected = [r for r in source_rows if r['parse_reason'] == 'unfinished_thinking']
    if not selected:
        parser.error('No unfinished_thinking samples found')
    for row in selected:
        if (row['ended_with_eos'] or '</think>' in row['model_output']
                or row['generated_tokens'] != config['generation']['max_new_tokens']
                or not row.get('enable_thinking', config.get('enable_thinking', False))):
            parser.error(f'Question {row["question_id"]} is not a budget-truncated thinking sample')
    if len({r['question_id'] for r in selected}) != len(selected):
        parser.error('Duplicate question IDs')
    metadata = {
        'source': str(args.input.resolve()), 'source_sha256': hashlib.sha256(source).hexdigest(),
        'model': config['model'], 'enable_thinking': True,
        'generation': dict(config['generation'], max_new_tokens=args.extra_tokens),
        'original_generation': config['generation'],
        'seed': config['seed'], 'seed_policy': 'seed + question_id, reset per question',
        'question_ids': [r['question_id'] for r in selected],
        'diagnostic_only': True, 'prefix_reconstructed_from_text': True,
    }
    metadata_path = args.output_dir / 'run_config.yaml'
    results_path = args.output_dir / 'results.jsonl'
    results = []
    if args.resume:
        if yaml.safe_load(metadata_path.read_text()) != metadata:
            parser.error('Resume metadata does not match source/configuration')
        results = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
        completed = [r['question_id'] for r in results]
        if len(set(completed)) != len(completed) or not set(completed) <= set(metadata['question_ids']):
            parser.error('Invalid completed question IDs')
    else:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        metadata_path.write_text(yaml.safe_dump(metadata), encoding='utf-8')
        results_path.touch(exist_ok=False)
    write_summary(args.output_dir, results, len(selected))
    done = {r['question_id'] for r in results}
    pending = [r for r in selected if r['question_id'] not in done]
    if not pending:
        print('All selected samples have already completed.')
        return
    tokenizer = AutoTokenizer.from_pretrained(config['model'])
    model = AutoModelForCausalLM.from_pretrained(
        config['model'], torch_dtype='auto', device_map='auto')
    model.eval()
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    with results_path.open('a', encoding='utf-8') as output_file:
        for old in pending:
            prompt_inputs = tokenizer.apply_chat_template(
                build_messages(old['problem']), tokenize=True, add_generation_prompt=True,
                enable_thinking=True, return_dict=True, return_tensors='pt')
            prefix_ids = tokenizer.encode(old['model_output'], add_special_tokens=False, return_tensors='pt')
            prefix_text = tokenizer.decode(prefix_ids[0], skip_special_tokens=False,
                                           clean_up_tokenization_spaces=False)
            if prefix_text != old['model_output']:
                raise ValueError(f'Cannot reconstruct saved text exactly: question {old["question_id"]}')
            input_ids = torch.cat([prompt_inputs['input_ids'], prefix_ids], dim=-1).to(model.device)
            if input_ids.shape[-1] + args.extra_tokens > model.config.max_position_embeddings:
                raise ValueError('Continuation would exceed model context length')
            attention_mask = torch.ones_like(input_ids)
            set_seed(config['seed'] + old['question_id'])
            print(f'[{len(results) + 1}/{len(selected)}] question_id={old["question_id"]} '
                  f'prefix_tokens={prefix_ids.shape[-1]} extra_budget={args.extra_tokens}', flush=True)
            with torch.inference_mode():
                generated = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                           **metadata['generation'])
            extra_ids = generated[0][input_ids.shape[-1]:]
            combined = torch.cat([prefix_ids[0], extra_ids.cpu()])
            model_output = tokenizer.decode(combined, skip_special_tokens=False,
                                            clean_up_tokenization_spaces=False)
            extraction = parse_answer(model_output, thinking_expected=True, problem=old['problem'])
            grading = grade_answer(extraction['answer'], old['gold_answer'], problem=old['problem'])
            ended = bool(extra_ids.numel() and extra_ids[-1].item() in (eos_ids or []))
            if not ended:
                outcome = 'still_unfinished_thinking' if extraction['reason'] == 'unfinished_thinking' else 'final_text_without_eos'
            elif extraction['answer'] is None:
                outcome = 'ended_without_final_answer'
            else:
                outcome = 'ended_and_correct' if grading['correct'] else 'ended_but_not_graded_correct'
            result = {
                'question_id': old['question_id'], 'problem': old['problem'], 'gold_answer': old['gold_answer'],
                'original_model_output': old['model_output'], 'model_output': model_output,
                'continuation_output': tokenizer.decode(extra_ids, skip_special_tokens=False,
                                                        clean_up_tokenization_spaces=False),
                'original_generated_tokens': old['generated_tokens'],
                'reconstructed_prefix_tokens': prefix_ids.shape[-1],
                'extra_generated_tokens': extra_ids.shape[-1],
                'total_generated_tokens': prefix_ids.shape[-1] + extra_ids.shape[-1],
                'ended_with_eos': ended, 'thinking_closed': '</think>' in model_output,
                'extra_limit_reached': extra_ids.shape[-1] >= args.extra_tokens,
                'parsed_answer': extraction['answer'], 'parse_reason': extraction['reason'],
                'correct': grading['correct'], 'grading': grading, 'outcome': outcome,
                'enable_thinking': True,
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            output_file.flush()
            results.append(result)
            write_summary(args.output_dir, results, len(selected))
            print(f'question_id={old["question_id"]} extra_tokens={extra_ids.shape[-1]} '
                  f'outcome={outcome}', flush=True)
    print(json.dumps(summarize(results, len(selected)), indent=2))


if __name__ == '__main__':
    main()
