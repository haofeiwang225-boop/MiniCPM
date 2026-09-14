"""Run one MMLU-Pro test row; preserve the supplied run_gpt4o.py prompt."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def form_options(options):
    # Verbatim formatting from the dataset's run_gpt4o.py.
    option_str = 'Options are:\n'
    opts = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    for opt, o in zip(options, opts):
        option_str += f'({o}): {opt}' + '\n'
    return option_str


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', default='openbmb/MiniCPM5-2B')
    parser.add_argument('--data-dir', type=Path, default=Path(
        '/home/hongchang/whf/datasets/MiniCPM5_eval/MMLU-Pro/data'))
    args = parser.parse_args()

    test = pq.read_table(args.data_dir / 'test-00000-of-00001.parquet')
    sample = test.slice(0, 1).to_pylist()[0]
    validation = pq.read_table(
        args.data_dir / 'validation-00000-of-00001.parquet').to_pylist()
    print('=== DATA SCHEMA ===', flush=True)
    print(test.schema, flush=True)
    print('=== SAMPLE ===', flush=True)
    print(json.dumps(sample, ensure_ascii=False, indent=2), flush=True)

    # Preserve the official dataset example's same-category few-shot prompt.
    prefix = ''
    for d in validation:
        if d['category'] == sample['category']:
            prefix += 'Q: ' + d['question'] + '\n' + form_options(d['options']) + '\n' + d['cot_content'] + '\n\n'
    query = prefix + 'Q: ' + sample['question'] + '\n' + form_options(sample['options']) + '\n'
    messages = [
        {'role': 'system', 'content': 'You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`.'},
        {'role': 'user', 'content': query},
    ]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else 'cpu',
    ).eval()
    inputs = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        enable_thinking=True, return_dict=True, return_tensors='pt',
    ).to(model.device)
    print('=== RENDERED PROMPT ===', flush=True)
    print(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False), flush=True)
    print('=== GENERATION CONFIG ===', flush=True)
    print(model.generation_config, flush=True)
    # Official Transformers cookbook limit; sampling defaults come from the checkpoint.
    with torch.inference_mode():
        outputs = model.generate(**inputs, max_new_tokens=1024)
    generated = outputs[0][inputs['input_ids'].shape[-1]:]
    print('=== FULL GENERATED TOKENS (think opener is in prompt) ===', flush=True)
    print(tokenizer.decode(generated, skip_special_tokens=False), flush=True)
    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    print('=== STOP STATUS ===', flush=True)
    print(json.dumps({'generated_tokens': len(generated),
                      'ended_with_eos': generated[-1].item() in (eos_ids or []),
                      'max_new_tokens': 1024}), flush=True)


if __name__ == '__main__':
    main()
