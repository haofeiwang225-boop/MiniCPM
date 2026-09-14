"""Local Transformers generation with unmodified LiveCodeBench scoring."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from prepare_data import DATASET_REPO, DATASET_REVISION, SHARDS, index_data

HERE = Path(__file__).resolve().parent
LCB_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
SNAPSHOT = (Path.home() / ".cache/huggingface/hub"
            / "datasets--livecodebench--code_generation_lite/snapshots" / DATASET_REVISION)


def get_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lcb_repo", type=Path,
                        default=Path("/home/hongchang/whf/code_benchmarks/LiveCodeBench"))
    parser.add_argument("--model", default="/home/hongchang/whf/models/MiniCPM5-2B")
    parser.add_argument("--data_dir", type=Path, default=SNAPSHOT)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--release_version", choices=["release_v6"], default="release_v6")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--enable_thinking", choices=["true", "false"], default="true")
    # Defaults from lcb_runner/runner/parser.py and vllm_runner.py.
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--stop", nargs="*", default=["###"],
                        help="Official stop strings; --stop without values disables them.")
    parser.add_argument("--num_process_evaluate", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check_only", action="store_true",
                        help="Verify data/imports/prompt without loading model weights.")
    args = parser.parse_args()
    args.enable_thinking = args.enable_thinking == "true"
    if args.n < 1 or args.max_tokens < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("n, max_tokens, and max_samples must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be nonnegative and 0 < top_p <= 1")
    for key in ("lcb_repo", "data_dir", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def final_response(raw, enable_thinking=True):
    if not enable_thinking:
        return raw, None
    finished = "</think>" in raw
    return (raw.rsplit("</think>", 1)[1] if finished else ""), finished


def dump(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    args = get_args()
    commit = subprocess.check_output(
        ["git", "-C", str(args.lcb_repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != LCB_COMMIT:
        raise RuntimeError(f"Unexpected LiveCodeBench commit: {commit}")
    # Imports in the upstream prompts package use repository-relative file paths.
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.lcb_repo))
    os.chdir(args.lcb_repo)
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.prompts.code_generation import get_generic_question_template_answer
    from lcb_runner.utils.extraction_utils import extract_code
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.evaluation import codegen_metrics
    from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, set_seed
    import torch

    # Refuse modified upstream scoring/prompt/data code, including staged changes.
    changed = subprocess.check_output([
        "git", "diff", "HEAD", "--name-only", "--", "lcb_runner",
    ], cwd=args.lcb_repo, text=True).strip()
    if changed:
        raise RuntimeError(f"Upstream code has local modifications:\n{changed}")
    index, counts = index_data(args.data_dir)
    print(f"release_v6 actual count: {len(index)}; shards: {counts}", flush=True)
    selected = index[:args.max_samples] if args.max_samples is not None else index
    if not args.check_only and args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {args.output_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    generation_config = GenerationConfig.from_pretrained(args.model, local_files_only=True)
    kwargs = dict(do_sample=args.temperature > 0, max_new_tokens=args.max_tokens,
                  num_beams=1, num_return_sequences=1, repetition_penalty=1.0)
    if args.temperature > 0:
        kwargs.update(temperature=args.temperature, top_p=args.top_p, top_k=0)
    else:
        # Neutralize sampling defaults inherited from the model configuration.
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 0
    if args.stop:
        kwargs["stop_strings"] = args.stop
    generation_config.update(**kwargs)
    config = dict(
        model=args.model, lcb_repo=str(args.lcb_repo), lcb_commit=commit,
        dataset=DATASET_REPO, dataset_revision=DATASET_REVISION,
        data_dir=str(args.data_dir), dataset_sha256=SHARDS, shard_counts=counts,
        release_version=args.release_version, dataset_size=len(index), num_samples=len(selected),
        n=args.n, seed=args.seed, enable_thinking=args.enable_thinking, few_shot=0, system_prompt=None,
        generation_args=kwargs, effective_generation_config=generation_config.to_dict(),
        num_process_evaluate=args.num_process_evaluate, timeout=args.timeout,
        scorer="lcb_runner.evaluation.codegen_metrics",
        parser="lcb_runner.utils.extraction_utils.extract_code: standard fenced-code branch",
        evaluator_sha256={str(p.relative_to(args.lcb_repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((args.lcb_repo / "lcb_runner/evaluation").glob("*.py"))},
    )
    print(json.dumps({"generation_args": kwargs, "n": args.n}, indent=2), flush=True)
    model = None
    if not args.check_only:
        args.output_dir.mkdir(parents=True, exist_ok=False)
        dump(args.output_dir / "run_config.json", config)
        set_seed(args.seed)
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", device_map="auto", local_files_only=True)
        model.eval()
    all_results = {}
    for position, (task_id, shard, offset) in enumerate(selected):
        with (args.data_dir / shard).open("rb") as stream:
            stream.seek(offset)
            question = CodeGenerationProblem(**json.loads(stream.readline()))
        prompt = get_generic_question_template_answer(question)
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=args.enable_thinking,
            return_dict=True, return_tensors="pt")
        rendered = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
        if args.check_only:
            print(f"Verified official prompt/chat template: task={task_id}, tokens={inputs['input_ids'].shape[-1]}")
            return
        inputs = inputs.to(model.device)
        prompt_length = inputs["input_ids"].shape[-1]
        candidates = []
        for sample_id in range(args.n):
            with torch.inference_mode():
                output = model.generate(**inputs, generation_config=generation_config,
                                        tokenizer=tokenizer, **kwargs)
            tokens = output[0, prompt_length:]
            raw = tokenizer.decode(tokens, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
            final, finished = final_response(raw, args.enable_thinking)
            code = extract_code(final, LMStyle.OpenAIChat)
            eos_ids = generation_config.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
            record = dict(question_id=task_id, sample_id=sample_id, prompt=prompt,
                          rendered_prompt=rendered, raw_response=raw, final_response=final,
                          extracted_code=code, generated_tokens=len(tokens),
                          enable_thinking=args.enable_thinking,
                          thinking_finished=finished,
                          unfinished_thinking=args.enable_thinking and not finished,
                          ended_with_eos=bool(len(tokens) and tokens[-1].item() in eos_ids),
                          length_limit_reached=len(tokens) >= args.max_tokens,
                          generation_args=kwargs)
            candidates.append(record)
            # Save before code execution so an interrupted evaluation loses no raw output.
            with (args.output_dir / "generations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Evaluate per question to bound memory; both checking and final aggregation
        # use upstream functions unchanged, with all n candidates for every question.
        metrics = codegen_metrics(
            [question.get_evaluation_sample()], [[r["extracted_code"] for r in candidates]],
            num_process_evaluate=args.num_process_evaluate, timeout=args.timeout)
        all_results[position] = metrics[1][0]
        with (args.output_dir / "results.jsonl").open("a", encoding="utf-8") as stream:
            for sample_id, record in enumerate(candidates):
                record["official_evaluation_result"] = metrics[1][0][sample_id]
                record["official_evaluation_metadata"] = metrics[2][0][sample_id]
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{position + 1}/{len(selected)}] {task_id}: {metrics[0]['pass@1']}", flush=True)
        del question, metrics, candidates, output, inputs
        gc.collect()
    # Same k list as the original codegen_metrics signature (no custom pass@k).
    import inspect
    k_list = inspect.signature(codegen_metrics).parameters["k_list"].default
    summary = compute_metrics_from_results(all_results, k_list=k_list)
    summary.update(num_tasks=len(selected), samples_per_task=args.n,
                   task_index_to_id={i: item[0] for i, item in enumerate(selected)})
    dump(args.output_dir / "summary.json", summary)
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("detail", "task_index_to_id")}, indent=2))


if __name__ == "__main__":
    main()
