"""Greedy debug copy of evaluate.py; only sampling mode and default config differ.

读取问题和标准答案 -> 仅用问题生成 -> 提取答案 -> 数学等价判分 -> 保存汇总。
"""

import argparse
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


def load_samples(data_path, limit=None):
    """仅保留 problem、answer；以零起始文件行号作为 question_id。"""
    samples = []
    with data_path.open(encoding='utf-8') as data_file:
        for index, line in enumerate(data_file):
            if not line.strip():
                continue
            row = json.loads(line)
            sample = {
                'question_id': index,
                'problem': row['problem'],
                'answer': row['answer'],
            }
            del row
            for key in ('problem', 'answer'):
                if not isinstance(sample[key], str) or not sample[key].strip():
                    raise ValueError(f'Invalid {key} at line {index + 1}')
            samples.append(sample)
            if limit is not None and len(samples) >= limit:
                break
    if not samples:
        raise ValueError('Test data is empty')
    return samples


def main():
    # 1. 命令行参数和配置，与 MMLU-Pro 脚本保持一致。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', type=Path, default=Path(__file__).with_name('debug_greedy.yaml'))
    parser.add_argument('--max_samples', type=int)
    parser.add_argument('--output_dir', type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    for key in ('max_samples', 'output_dir'):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value

    limit = config['max_samples']
    if limit is not None and (type(limit) is not int or limit <= 0):
        parser.error('max_samples must be a positive integer or null in YAML')
    required_generation = {
        'do_sample': False, 'temperature': 1.0,
        'top_p': 0.95, 'max_new_tokens': 8192,
    }
    if config['generation'] != required_generation:
        parser.error(f'This experiment requires generation={required_generation}')

    # 2. 只读取本地数据，不下载数据；答案不进入 prompt。
    samples = load_samples(Path(config['data_path']).expanduser(), limit)
    output_dir = Path(config['output_dir']).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'results.jsonl'

    # 3. 防止覆盖已有结果，逐题保存。
    with output_path.open('x', encoding='utf-8') as output_file:
        (output_dir / 'run_config.yaml').write_text(
            yaml.safe_dump(config), encoding='utf-8')
        set_seed(config['seed'])
        tokenizer = AutoTokenizer.from_pretrained(config['model'])
        model = AutoModelForCausalLM.from_pretrained(
            config['model'], torch_dtype='auto', device_map='auto')
        model.eval()
        eos_ids = model.generation_config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        results = []

        # 4. 官方 chat template 与固定生成配置。
        for index, sample in enumerate(samples):
            messages = build_messages(sample['problem'])
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=True, return_dict=True, return_tensors='pt',
            ).to(model.device)
            print(
                f'[{index + 1}/{len(samples)}] '
                f'prompt_tokens={inputs["input_ids"].shape[-1]}', flush=True)
            with torch.inference_mode():
                outputs = model.generate(**inputs, **config['generation'])
            generated = outputs[0][inputs['input_ids'].shape[-1]:]
            model_output = tokenizer.decode(generated, skip_special_tokens=False)

            # 5. 复用现有 parser 和 grader；未提取到答案时计为错误。
            extraction = parse_answer(model_output, thinking_expected=True)
            parsed_answer = extraction['answer']
            grading = grade_answer(parsed_answer, sample['answer'])
            correct = grading['correct']
            result = {
                'question_id': sample['question_id'],
                'problem': sample['problem'],
                'gold_answer': sample['answer'],
                'model_output': model_output,
                'parsed_answer': parsed_answer,
                'correct': bool(correct),
                'parse_reason': extraction['reason'],
                'grading': grading,
                'generated_tokens': generated.shape[-1],
                'ended_with_eos': bool(
                    generated.numel() and generated[-1].item() in (eos_ids or [])),
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            output_file.flush()
            results.append(result)
            print(
                f'[{index + 1}/{len(samples)}] '
                f'question_id={result["question_id"]} correct={correct}', flush=True)

    # 6. 保存整体准确率，包含未解析出答案的样本。
    total = len(results)
    correct = sum(row['correct'] for row in results)
    summary = {'total': total, 'correct': correct, 'accuracy': correct / total}
    (output_dir / 'summary.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(
        f'Total samples: {total}\nCorrect: {correct}\n'
        f'Accuracy: {summary["accuracy"]:.6f}')
    print(f'Results: {output_path.resolve()}')


if __name__ == '__main__':
    main()
