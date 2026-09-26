"""Evaluate local MATH-500 through a local OpenAI-compatible endpoint."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
import yaml

try:
    from .prompt import build_messages
    from .answer_parser import parse_answer
    from .grader import grade_answer
except ImportError:
    from prompt import build_messages
    from answer_parser import parse_answer
    from grader import grade_answer


def build_request(config, problem):
    if config.get('enable_thinking') is not True:
        raise ValueError('Author protocol requires enable_thinking=true')
    if config.get('generation') != {'temperature': 1.0, 'top_p': 0.95, 'n': 1}:
        raise ValueError('generation must contain only temperature=1, top_p=0.95, n=1')
    if config.get('prompt_suffix'):
        raise ValueError('Custom prompt_suffix is not permitted by this protocol')
    return {'model': config['served_model_name'], 'messages': build_messages(problem),
            **config['generation'], 'chat_template_kwargs': {'enable_thinking': True}}


def endpoint_json(url, payload=None):
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode('utf-8')
    request = Request(url, data=body, headers={'Content-Type': 'application/json'})
    # Transport timeout only; no token/time budget is sent to the generation API.
    with urlopen(request, timeout=3600) as response:
        return json.load(response)


def score_response(raw, problem, gold):
    finished = '</think>' in raw
    final = raw.rsplit('</think>', 1)[1] if finished else ''
    extraction = (parse_answer(final, thinking_expected=False, problem=problem) if finished
                  else {'answer': None, 'reason': 'unfinished_thinking'})
    grading = grade_answer(extraction['answer'], gold, problem=problem)
    return final, finished, extraction, grading


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
        'generation': config['generation'],               # 实际 API 采样参数
        'enable_thinking': config.get('enable_thinking', True),  # 是否启用 thinking 模式
        'seed': config.get('seed'),                           # 随机种子，用于复现
        'unfinished_thinking': sum(                       # thinking 阶段未完成的样本数
            row['parse_reason'] == 'unfinished_thinking' for row in results),
        'length_limit_reached': sum(row['length_limit_reached'] for row in results),  # endpoint 报告 finish_reason=length 的样本数
        'no_eos': sum(row['ended_with_eos'] is False for row in results),  # 生成未以 eos 结束的样本数
        'eos_unknown': sum(row['ended_with_eos'] is None for row in results),
        'finish_reasons': dict(Counter(row['finish_reason'] for row in results)),
        'parse_reasons': dict(Counter(row['parse_reason'] for row in results)),      # 答案解析原因 -> 计数
        'grading_reasons': dict(Counter(row['grading']['reason'] for row in results)),  # 判分原因 -> 计数
        'failed_question_ids': [row['question_id'] for row in results if not row['correct']],  # 错误样本的 question_id 列表
    }
    temporary = output_dir / 'summary.json.tmp'
    temporary.write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    temporary.replace(output_dir / 'summary.json')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.yaml'))
    parser.add_argument('--max_samples', type=int)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--question_ids', type=int, nargs='+')
    parser.add_argument('--enable_thinking', choices=['true'], default='true')
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    for key in ('max_samples', 'output_dir'):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    config['question_ids'] = args.question_ids
    if args.question_ids is not None and any(i < 0 for i in args.question_ids):
        parser.error('question_ids must be nonnegative')
    limit = config['max_samples']
    if limit is not None and (type(limit) is not int or limit <= 0):
        parser.error('max_samples must be positive or null')
    build_request(config, 'validation')
    if config['context_length'] != 131072:
        parser.error('context_length must be 131072')
    samples = load_samples(Path(config['data_path']).expanduser(), limit, args.question_ids)
    output_dir = Path(config['output_dir']).expanduser()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output directory: {output_dir}')
    base_url = config['base_url'].rstrip('/')
    models = endpoint_json(base_url + '/models')
    served = next((m for m in models['data'] if m['id'] == config['served_model_name']), None)
    if served is None or served.get('max_model_len') != 131072:
        raise RuntimeError('Endpoint must report requested model with max_model_len=131072')
    config['endpoint_model_metadata'] = served
    config['frozen_files_sha256'] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('grader.py', 'answer_parser.py')}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'run_config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    output_path = output_dir / 'results.jsonl'
    results = []
    with output_path.open('x', encoding='utf-8') as output_file, \
            (output_dir / 'failures.jsonl').open('x', encoding='utf-8') as failure_file, \
            (output_dir / 'requests.jsonl').open('x', encoding='utf-8') as request_file:
        save_summary(results, output_dir, config, len(samples))
        for index, sample in enumerate(samples):
            payload = build_request(config, sample['problem'])
            print('Actual endpoint request: ' + json.dumps(payload, ensure_ascii=False), flush=True)
            request_file.write(json.dumps({'question_id': sample['question_id'], 'request': payload}, ensure_ascii=False) + '\n')
            request_file.flush()
            # One request, n=1; no automatic retries or regeneration.
            response = endpoint_json(base_url + '/chat/completions', payload)
            if len(response.get('choices', [])) != 1:
                raise RuntimeError('Expected exactly one generated choice')
            choice = response['choices'][0]
            message = choice['message']
            raw = message.get('content') or ''
            # The dedicated server runs without a reasoning parser, preserving </think>.
            # A server that strips the delimiter must not be silently treated as finished.
            final, finished, extraction, grading = score_response(raw, sample['problem'], sample['answer'])
            finish_reason = choice.get('finish_reason')
            result = {
                'question_id': sample['question_id'], 'enable_thinking': True,
                'problem': sample['problem'], 'gold_answer': sample['answer'],
                'prompt': payload['messages'][0]['content'], 'request': payload,
                'model_output': raw, 'endpoint_response': response,
                'final_response': final, 'thinking_finished': finished,
                'parsed_answer': extraction['answer'], 'correct': bool(grading['correct']),
                'score': int(bool(grading['correct'])), 'parse_reason': extraction['reason'],
                'grading': grading, 'generated_tokens': response.get('usage', {}).get('completion_tokens'),
                'length_limit_reached': finish_reason == 'length',
                # OpenAI finish_reason does not prove which token caused stopping.
                'ended_with_eos': None, 'finish_reason': finish_reason,
            }
            output_file.write(json.dumps(result, ensure_ascii=False) + '\n')
            output_file.flush()
            if not result['correct']:
                failure_file.write(json.dumps(result, ensure_ascii=False) + '\n')
                failure_file.flush()
            results.append(result)
            save_summary(results, output_dir, config, len(samples))
            print(f"[{index + 1}/{len(samples)}] question_id={sample['question_id']} "
                  f"correct={result['correct']} tokens={result['generated_tokens']} "
                  f"finish_reason={finish_reason} parse_reason={extraction['reason']}", flush=True)
    summary = save_summary(results, output_dir, config, len(samples))
    print(f"Total samples: {summary['total']}\nCorrect: {summary['correct']}\nAccuracy: {summary['accuracy']:.6f}")
    print(f'Results: {output_path.resolve()}')


if __name__ == '__main__':
    main()
