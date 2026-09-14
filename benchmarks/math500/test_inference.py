"""Compare two generation limits on the same MATH-500 problem and prompt."""

import json
from copy import deepcopy
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


MODEL_PATH = '/home/hongchang/whf/models/MiniCPM5-2B'
DATA_PATH = '/home/hongchang/whf/datasets/MiniCPM5_eval/MATH-500/test.jsonl'
SEED = 42


def main():
    with open(DATA_PATH, encoding='utf-8') as data_file:
        problem = json.loads(next(data_file))['problem']

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype='auto', device_map='auto',
    )
    model.eval()

    messages = [{'role': 'user', 'content': problem}]
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=True,
        return_dict=True,
        return_tensors='pt',
    ).to(model.device)
    prompt = tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
    generation_config = model.generation_config.to_dict()
    print('=== PROBLEM ===\n' + problem, flush=True)
    print('=== PROMPT ===\n' + prompt, flush=True)
    print('=== GENERATION CONFIG ===', flush=True)
    print(json.dumps(generation_config, ensure_ascii=False, indent=2), flush=True)

    eos_ids = model.generation_config.eos_token_id
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    for max_new_tokens in (1024, 2048):
        # Reset the sampling RNG: the length limit is the only experiment variable.
        set_seed(SEED)
        run_config = deepcopy(model.generation_config)
        run_config.update(
            do_sample=True, temperature=1.0, top_p=0.95,
            max_new_tokens=max_new_tokens,
        )
        with torch.inference_mode():
            outputs = model.generate(**inputs, generation_config=run_config)
        generated = outputs[0][inputs['input_ids'].shape[-1]:]
        model_output = tokenizer.decode(generated, skip_special_tokens=False)
        record = {
            'model': MODEL_PATH,
            'data_path': DATA_PATH,
            'problem': problem,
            'prompt': prompt,
            'enable_thinking': True,
            'seed': SEED,
            'generation_config': run_config.to_dict(),
            'model_output': model_output,
            'prompt_tokens': inputs['input_ids'].shape[-1],
            'generated_tokens': generated.shape[-1],
            'ended_with_eos': bool(generated.numel() and generated[-1].item() in (eos_ids or [])),
        }
        output_path = Path(__file__).with_name(
            f'test_inference_output_{max_new_tokens}.json')
        output_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
        )
        # The opening <think> belongs to the prompt; print all generated tokens.
        print(f'=== FULL MODEL OUTPUT ({max_new_tokens}) ===\n' + model_output, flush=True)
        print(f'Saved: {output_path}', flush=True)


if __name__ == '__main__':
    main()
