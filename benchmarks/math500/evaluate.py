"""Evaluate local MATH-500 using MiniCPM5 Transformers inference.

读取问题和标准答案 -> 仅用问题生成 -> 提取答案 -> 数学等价判分 -> 保存汇总。
"""

# 标准库与第三方依赖：argparse 用于命令行参数解析，json/yaml 用于配置与结果读写，
# torch 与 transformers 用于加载模型并执行推理。
import argparse
from collections import Counter
import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

# 兼容包内导入与直接运行两种方式：作为模块时用相对导入，作为脚本时退化为绝对导入。
try:
    from .prompt import build_messages
    from .answer_parser import parse_answer
    from .grader import grade_answer
except ImportError:
    from prompt import build_messages
    from answer_parser import parse_answer
    from grader import grade_answer


def load_samples(data_path, limit=None, question_ids=None):
    """仅保留 problem、answer；以零起始文件行号作为 question_id。

    负责从 JSONL 数据文件中加载测试样本，支持按 limit 截断与按 question_ids 精确筛选，
    并对每条数据的字段类型与非空做基本校验。
    """
    samples = []
    selected_ids = set(question_ids) if question_ids is not None else None
    with data_path.open(encoding='utf-8') as data_file:
        for index, line in enumerate(data_file):
            if selected_ids is not None and index not in selected_ids:
                continue
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
            if selected_ids is not None and len(samples) == len(selected_ids):
                break
            if selected_ids is None and limit is not None and len(samples) >= limit:
                break
    if selected_ids is not None:
        missing = selected_ids - {sample['question_id'] for sample in samples}
        if missing:
            raise ValueError(f'question_ids not found: {sorted(missing)}')
        if limit is not None:
            samples = samples[:limit]
    if not samples:
        raise ValueError('Test data is empty')
    return samples


def save_summary(results, output_dir, config, expected_total):
    """Record first-attempt accuracy and completion failures separately.

    汇总当前结果集并原子写入 summary.json：统计准确率、未完成 thinking、长度超限、
    未以 eos 结束等各类失败原因计数，便于实时追踪评测进度。
    """
    total = len(results)                                  # 当前已评测样本数
    correct = sum(row['correct'] for row in results)      # 其中判为正确的样本数
    summary = {
        'total': total,                                   # 已评测样本总数
        'correct': correct,                               # 正确样本数
        'accuracy': correct / total if total else 0.0,    # 准确率 = correct / total
        'expected_total': expected_total,                 # 期望评测的总样本数（来自数据加载）
        'complete': total == expected_total,              # 是否已完成全部期望样本的评测
        'model': config['model'],                         # 使用的模型名称或路径
        'generation': config['generation'],               # 生成配置（do_sample/temperature/top_p/max_new_tokens 等）
        'enable_thinking': config.get('enable_thinking', True),  # 是否启用 thinking 模式
        'seed': config['seed'],                           # 随机种子，用于复现
        'unfinished_thinking': sum(                       # thinking 阶段未完成的样本数
            row['parse_reason'] == 'unfinished_thinking' for row in results),
        'length_limit_reached': sum(row['length_limit_reached'] for row in results),  # 触达 max_new_tokens 长度上限的样本数
        'no_eos': sum(not row['ended_with_eos'] for row in results),  # 生成未以 eos 结束的样本数
        'parse_reasons': dict(Counter(row['parse_reason'] for row in results)),      # 答案解析原因 -> 计数
        'grading_reasons': dict(Counter(row['grading']['reason'] for row in results)),  # 判分原因 -> 计数
        'failed_question_ids': [row['question_id'] for row in results if not row['correct']],  # 错误样本的 question_id 列表
    }
    temporary = output_dir / 'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    temporary.replace(output_dir / 'summary.json')
    return summary


def main():
    """评测主流程：解析配置 -> 加载数据 -> 加载模型 -> 逐题生成与判分 -> 保存汇总。

    步骤说明：
      1. 命令行参数与 YAML 配置合并，校验生成参数是否符合实验要求；
      2. 仅读取本地 JSONL 数据，答案不进入 prompt；
      3. 以 'x' 模式打开结果文件防止覆盖已有结果，逐题即时落盘；
      4. 使用官方 chat template 与 YAML 生成配置进行推理；
      5. 复用现有 parser 与 grader，未提取到答案计为错误；
      6. 最终保存整体准确率，包含未解析出答案的样本。
    """
    # 1. 命令行参数和配置，与 MMLU-Pro 脚本保持一致。
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--max_samples', type=int)
    parser.add_argument('--output_dir', type=Path) #命令行配置
    parser.add_argument(
        '--enable_thinking', type=str.lower, choices=('true', 'false'), default='true',
        help='Enable thinking (default: true)')
    parser.add_argument(
        '--question_ids', type=int, nargs='+',
        help='Select zero-based question IDs; max_samples applies after selection')

    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    config['enable_thinking'] = args.enable_thinking == 'true'
    config['question_ids'] = args.question_ids
    if args.question_ids is not None and any(index < 0 for index in args.question_ids):
        parser.error('question_ids must be nonnegative integers')
    for key in ('max_samples', 'output_dir'):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value

    limit = config['max_samples']
    if limit is not None and (type(limit) is not int or limit <= 0):
        parser.error('max_samples must be a positive integer or null in YAML')
    if not isinstance(config.get('generation'), dict):
        parser.error('generation must be a mapping in YAML')
    max_new_tokens = config['generation'].get('max_new_tokens')
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        parser.error('generation.max_new_tokens must be a positive integer')
    prompt_suffix = config.get('prompt_suffix') or ''
    if not isinstance(prompt_suffix, str):
        parser.error('prompt_suffix must be a string or null')

    # 2. 只读取本地数据，不下载数据；答案不进入 prompt。
    samples = load_samples(
        Path(config['data_path']).expanduser(), limit, args.question_ids)
    output_dir = Path(config['output_dir']).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'results.jsonl'

    # 3. 防止覆盖已有结果，逐题保存。
    with output_path.open('x', encoding='utf-8') as output_file, \
            (output_dir / 'failures.jsonl').open('x', encoding='utf-8') as failure_file:
        (output_dir / 'run_config.yaml').write_text(
            yaml.safe_dump(config), encoding='utf-8')
        print(
            'model.generate kwargs: ' + json.dumps(config['generation']), flush=True)
        set_seed(config['seed'])
        tokenizer = AutoTokenizer.from_pretrained(config['model'])#tokenizer 负责把人类文字转换成模型能理解的数字 token，以及把模型输出的 token 转回文字
        model = AutoModelForCausalLM.from_pretrained(
            config['model'], torch_dtype='auto', device_map='auto')
        model.eval()
        eos_ids = model.generation_config.eos_token_id #eos 是停止的符号
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        results = []
        save_summary(results, output_dir, config, len(samples))

        # 4. 官方 chat template 与 YAML 生成配置；空 suffix 不改变原始 prompt。
        for index, sample in enumerate(samples):
            messages = build_messages(sample['problem'])
            if prompt_suffix:
                messages[-1]['content'] += '\n\n' + prompt_suffix
            inputs = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=config['enable_thinking'], return_dict=True, return_tensors='pt',
            ).to(model.device)
            print(
                f'[{index + 1}/{len(samples)}] '
                f'prompt_tokens={inputs["input_ids"].shape[-1]}', flush=True)
            with torch.inference_mode():
                outputs = model.generate(**inputs, **config['generation'])
            generated = outputs[0][inputs['input_ids'].shape[-1]:]
            model_output = tokenizer.decode(generated, skip_special_tokens=False)

            # 5. 复用现有 parser 和 grader；未提取到答案时计为错误。
            extraction = parse_answer(
                model_output, thinking_expected=config['enable_thinking'], problem=sample['problem'])
            parsed_answer = extraction['answer']
            grading = grade_answer(parsed_answer, sample['answer'], problem=sample['problem'])
            correct = grading['correct']
            result = {
                'question_id': sample['question_id'],
                'enable_thinking': config['enable_thinking'],
                'problem': sample['problem'],
                'gold_answer': sample['answer'],
                'model_output': model_output,
                'parsed_answer': parsed_answer,
                'correct': bool(correct),
                'parse_reason': extraction['reason'],
                'grading': grading,
                'generated_tokens': generated.shape[-1],
                'length_limit_reached': generated.shape[-1] >= config['generation']['max_new_tokens'],
                'ended_with_eos': bool(
                    generated.numel() and generated[-1].item() in (eos_ids or [])),
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            output_file.flush()
            if not correct:
                failure_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                failure_file.flush()
            results.append(result)
            save_summary(results, output_dir, config, len(samples))
            print(
                f'[{index + 1}/{len(samples)}] '
                f'question_id={result["question_id"]} correct={correct} '
                f'generated_tokens={result["generated_tokens"]} '
                f'ended_with_eos={result["ended_with_eos"]} '
                f'parse_reason={result["parse_reason"]}', flush=True)

    # 6. 保存整体准确率，包含未解析出答案的样本。
    total = len(results)
    correct = sum(row['correct'] for row in results)
    summary = save_summary(results, output_dir, config, len(samples))
    print(
        f'Total samples: {total}\nCorrect: {correct}\n'
        f'Accuracy: {summary["accuracy"]:.6f}')
    print(f'Results: {output_path.resolve()}')


if __name__ == '__main__':
    # 脚本直接运行入口：执行评测主流程。
    main()
