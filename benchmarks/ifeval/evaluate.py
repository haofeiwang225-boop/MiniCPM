import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys

import yaml

ROOT = Path(__file__).resolve().parent

METRICS = (
    'prompt_level_strict_acc',
    'inst_level_strict_acc',
    'prompt_level_loose_acc',
    'inst_level_loose_acc',
)


def load_samples(path, limit):
    if limit is not None and (type(limit) is not int or limit <= 0):
        raise ValueError('max_samples must be positive or null')

    rows = []
    seen = set()

    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue

        row = json.loads(line)

        if row['key'] in seen or not isinstance(row['prompt'], str):
            raise ValueError('Invalid prompt or duplicate key')

        if (
            not row['instruction_id_list']
            or len(row['instruction_id_list']) != len(row['kwargs'])
        ):
            raise ValueError(
                'Instruction IDs and kwargs must have matching nonzero lengths'
            )

        seen.add(row['key'])
        rows.append(row)

        if limit is not None and len(rows) >= limit:
            break

    if not rows:
        raise ValueError('Empty IFEval dataset')

    return rows


def verify_vendor():
    provenance = json.loads(
        (ROOT / 'vendor/provenance.json').read_text(encoding='utf-8')
    )

    for name, digest in provenance['files'].items():
        path = ROOT / 'vendor/lm_eval/tasks/ifeval' / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest()

        if actual != digest:
            raise RuntimeError(f'Vendored harness file changed: {name}')

    return provenance


def load_evaluator(nltk_data=None):
    verify_vendor()

    missing = [
        name
        for name in ('nltk', 'langdetect', 'immutabledict', 'packaging')
        if importlib.util.find_spec(name) is None
    ]

    if missing:
        raise RuntimeError(
            'Missing offline dependencies: '
            + ', '.join(missing)
            + '. Install from local wheels; no download was attempted.'
        )

    import nltk
    from packaging.version import Version

    if Version(importlib.metadata.version('nltk')) < Version('3.9.1'):
        raise RuntimeError('Local nltk>=3.9.1 is required')

    if nltk_data:
        nltk.data.path.insert(
            0,
            str(Path(nltk_data).expanduser()),
        )

    try:
        nltk.data.find('tokenizers/punkt_tab')

        for name in (
            'collocations.tab',
            'sent_starters.txt',
            'abbrev_types.txt',
            'ortho_context.tab',
        ):
            nltk.data.find(
                'tokenizers/punkt_tab/english/' + name
            )

    except LookupError as error:
        raise RuntimeError(
            'Local NLTK punkt_tab resource is missing. '
            'Supply nltk_data in YAML or NLTK_DATA; '
            'automatic download is disabled by this precheck.'
        ) from error

    if any(
        name == 'lm_eval' or name.startswith('lm_eval.')
        for name in sys.modules
    ):
        raise RuntimeError(
            'Run this script in a fresh process to use the pinned evaluator'
        )

    sys.path.insert(
        0,
        str(ROOT / 'vendor'),
    )

    return importlib.import_module(
        'lm_eval.tasks.ifeval.utils'
    )


def aggregate(scores, evaluator):
    return {
        name: (
            evaluator.agg_inst_level_acc(
                [score[name] for score in scores]
            )
            if name.startswith('inst_')
            else sum(score[name] for score in scores) / len(scores)
        )
        for name in METRICS
    }


def generation_status(token_ids, raw, eos_token_id, max_new_tokens):
    """Diagnostics only: never change the response passed to the scorer."""
    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else (eos_token_id or [])
    return {
        'ended_with_eos': bool(token_ids and token_ids[-1] in eos_ids),
        'length_limit_reached': len(token_ids) >= max_new_tokens,
        # The initial <think> is in the chat prompt, outside generated tokens.
        'thinking_finished': '</think>' in raw and raw.rfind('</think>') > raw.rfind('<think>'),
    }


def final_response(response, thinking_finished):
    """Match the diagnostic baseline: no answer is mined from unfinished thinking."""
    if not thinking_finished or '</think>' not in response:
        return ''
    return response.rsplit('</think>', 1)[-1].strip()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--config',
        type=Path,
        default=ROOT / 'config.yaml',
    )

    parser.add_argument(
        '--max_samples',
        type=int,
    )

    parser.add_argument(
        '--output_dir',
        type=Path,
    )

    parser.add_argument(
        '--check_only',
        action='store_true',
    )

    args = parser.parse_args()

    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'

    config = yaml.safe_load(
        args.config.read_text(encoding='utf-8')
    )

    if args.max_samples is not None:
        config['max_samples'] = args.max_samples

    if args.output_dir is not None:
        config['output_dir'] = str(args.output_dir)

    generation = config['generation']
    if config['num_fewshot'] != 0:
        parser.error('This protocol requires 0-shot')
    if (type(generation.get('do_sample')) is not bool
            or type(generation.get('max_new_tokens')) is not int
            or generation['max_new_tokens'] <= 0
            or generation.get('num_beams', 1) != 1
            or generation.get('num_return_sequences', 1) != 1):
        parser.error('Require do_sample boolean, positive max_new_tokens, one beam and one sequence')
    if config['enable_thinking'] is not True or config['response_processing'] != 'final_only':
        parser.error('This protocol requires thinking=True and final_only scoring')
    system_prompt = config.get('system_prompt')
    if system_prompt is not None and not isinstance(system_prompt, str):
        parser.error('system_prompt must be a string or null')

    model_path = Path(
        config['model']
    ).expanduser()

    if not model_path.is_dir():
        parser.error(
            'model must be an existing local directory'
        )

    data_path = Path(
        config['data_path']
    ).expanduser()

    samples = load_samples(
        data_path,
        config['max_samples'],
    )

    evaluator = load_evaluator(
        config.get('nltk_data')
    )

    from lm_eval.tasks.ifeval.instructions_registry import (
        INSTRUCTION_DICT,
    )

    unknown = {
        item
        for row in samples
        for item in row['instruction_id_list']
    } - INSTRUCTION_DICT.keys()

    if unknown:
        raise ValueError(
            f'Unknown instruction IDs: {sorted(unknown)}'
        )

    if args.check_only:
        print(
            f'Offline preflight passed: '
            f'{len(samples)} samples; '
            f'pinned upstream scorer import succeeded.'
        )
        return

    import torch

    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        set_seed,
    )

    output_dir = Path(
        config['output_dir']
    ).expanduser()

    output_dir.mkdir(
        parents=True,
        exist_ok=False,
    )

    set_seed(config['seed'])
    random.seed(config['seed'])
    # langdetect uses its own random.Random, independent of random.seed().
    from langdetect import DetectorFactory
    DetectorFactory.seed = config['seed']
    config['langdetect_seed'] = DetectorFactory.seed

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype='auto',
        device_map='auto',
        local_files_only=True,
    )

    model.eval()

    protocol_generation_config = dict(
        config['generation']
    )

    actual_generate_kwargs = dict(config['generation'])
    actual_generate_kwargs.setdefault('num_beams', 1)
    actual_generate_kwargs.setdefault('num_return_sequences', 1)

    effective_generation_config = (
        model.generation_config.to_dict()
    )

    effective_generation_config.update(
        actual_generate_kwargs
    )

    config.update(
        {
            'protocol_generation_config':
                protocol_generation_config,
            'actual_generate_kwargs':
                actual_generate_kwargs,
            'effective_generation_config':
                effective_generation_config,
            'num_generations_per_prompt':
                1,
            'num_samples':
                len(samples),
            'data_sha256':
                hashlib.sha256(
                    data_path.read_bytes()
                ).hexdigest(),
            'chat_template':
                tokenizer.chat_template,
            'evaluator':
                verify_vendor(),
            'versions':
                {
                    name:
                        importlib.metadata.version(name)
                    for name in (
                        'torch',
                        'transformers',
                        'nltk',
                        'langdetect',
                        'immutabledict',
                    )
                },
        }
    )

    (
        output_dir / 'run_config.yaml'
    ).write_text(
        yaml.safe_dump(
            config,
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding='utf-8',
    )

    print(
        'Actual generate kwargs:',
        json.dumps(
            actual_generate_kwargs,
            ensure_ascii=False,
        ),
        flush=True,
    )

    scores = []
    status_counts = {name: 0 for name in ('ended_with_eos', 'length_limit_reached',
                                         'unfinished_thinking', 'final_started_but_truncated')}
    total_generated_tokens = 0

    results_path = (
        output_dir / 'results.jsonl'
    )

    with results_path.open(
        'x',
        encoding='utf-8',
    ) as result_file:

        for index, doc in enumerate(samples):

            messages = [
                {
                    'role': 'user',
                    'content': doc['prompt'],
                }
            ]
            if system_prompt:
                messages.insert(0, {'role': 'system', 'content': system_prompt})

            rendered = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )

            inputs = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
                return_dict=True,
                return_tensors='pt',
            ).to(model.device)

            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    **actual_generate_kwargs,
                )

            tokens = outputs[0][
                inputs['input_ids'].shape[-1]:
            ]

            raw = tokenizer.decode(
                tokens,
                skip_special_tokens=False,
            )

            response = tokenizer.decode(
                tokens,
                skip_special_tokens=True,
            )

            status = generation_status(
                tokens.tolist(), raw,
                effective_generation_config.get('eos_token_id'),
                actual_generate_kwargs['max_new_tokens'],
            )

            response = final_response(response, status['thinking_finished'])
            status['unfinished_thinking'] = not status['thinking_finished']
            status['final_started_but_truncated'] = (
                status['thinking_finished'] and status['length_limit_reached']
                and not status['ended_with_eos'])

            score = evaluator.process_results(
                doc,
                [response],
            )

            scores.append(score)
            for name in status_counts:
                status_counts[name] += int(status[name])
            total_generated_tokens += len(tokens)

            row = dict(
                doc,
                rendered_prompt=rendered,
                model_output=raw,
                scored_response=response,
                enable_thinking=True,
                generated_tokens=len(tokens),
                scores=score,
                **status,
            )

            result_file.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                )
                + '\n'
            )

            result_file.flush()

            summary = dict(
                total=len(scores),
                expected_total=len(samples),
                complete=len(scores) == len(samples),
                **status_counts,
                avg_generated_tokens=total_generated_tokens / len(scores),
                **aggregate(
                    scores,
                    evaluator,
                ),
            )

            temporary = (
                output_dir / 'summary.json.tmp'
            )

            temporary.write_text(
                json.dumps(
                    summary,
                    indent=2,
                    ensure_ascii=False,
                )
                + '\n',
                encoding='utf-8',
            )

            temporary.replace(
                output_dir / 'summary.json'
            )

            print(
                f'[{index + 1}/{len(samples)}] '
                f'key={doc["key"]} '
                f'tokens={len(tokens)} '
                f'ended_with_eos={status["ended_with_eos"]} '
                f'length_limit_reached={status["length_limit_reached"]} '
                f'thinking_finished={status["thinking_finished"]} '
                f'scores={score}',
                flush=True,
            )

    print(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == '__main__':
    main()
