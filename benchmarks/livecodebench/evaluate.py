"""Local Transformers generation with unmodified LiveCodeBench scoring."""

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

# 从数据准备脚本导入数据集仓库地址、版本号、分片哈希和索引构建函数
from prepare_data import DATASET_REPO, DATASET_REVISION, SHARDS, index_data

# 当前脚本所在目录
HERE = Path(__file__).resolve().parent
# 锁定上游 LiveCodeBench 的 commit，确保评测逻辑版本一致
LCB_COMMIT = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
# 数据集在 HuggingFace 本地缓存中的快照路径
SNAPSHOT = (Path.home() / ".cache/huggingface/hub"
            / "datasets--livecodebench--code_generation_lite/snapshots" / DATASET_REVISION)


def get_args():
    """解析命令行参数，包括仓库路径、模型路径、生成配置和评测配置。"""
    parser = argparse.ArgumentParser(description=__doc__)
    # --- 路径与数据集参数 ---
    parser.add_argument("--lcb_repo", type=Path,
                        default=Path("/home/hongchang/whf/code_benchmarks/LiveCodeBench"))
    parser.add_argument("--model", default="/home/hongchang/whf/models/MiniCPM5-2B")
    parser.add_argument("--data_dir", type=Path, default=SNAPSHOT)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--release_version", choices=["release_v6"], default="release_v6")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument("--enable_thinking", choices=["true", "false"], default="true")
    # --- 生成参数，默认值与 lcb_runner/runner/parser.py 和 vllm_runner.py 保持一致 ---
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--max_tokens", type=int, default=2000)
    parser.add_argument("--stop", nargs="*", default=["###"],
                        help="Official stop strings; --stop without values disables them.")
    # --- 评测参数 ---
    parser.add_argument("--num_process_evaluate", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check_only", action="store_true",
                        help="Verify data/imports/prompt without loading model weights.")
    args = parser.parse_args()
    # 将字符串 "true"/"false" 转为布尔值
    args.enable_thinking = args.enable_thinking == "true"
    # 校验关键参数的合法范围，避免无效配置导致评测异常
    if args.n < 1 or args.max_tokens < 1 or (args.max_samples is not None and args.max_samples < 1):
        parser.error("n, max_tokens, and max_samples must be positive")
    if args.temperature < 0 or not 0 < args.top_p <= 1:
        parser.error("temperature must be nonnegative and 0 < top_p <= 1")
    # 将路径参数统一解析为绝对路径，避免后续因工作目录切换导致路径失效
    for key in ("lcb_repo", "data_dir", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    return args


def final_response(raw, enable_thinking=True):
    """从模型原始输出中提取最终回答。

    思考模式下，模型输出包含 <think>...</think> 思考过程和最终回答，
    需要剥离思考部分只保留最终回答；非思考模式直接返回原始输出。
    """
    # 非思考模式：原始输出即为最终回答，无需处理
    if not enable_thinking:
        return raw, None
    # 思考模式：检查思考过程是否完整结束（存在闭合的 </think> 标签）
    finished = "</think>" in raw
    # 若思考结束，取最后一个 </think> 之后的内容作为最终回答；否则返回空串
    return (raw.rsplit("</think>", 1)[1] if finished else ""), finished


def dump(path, value):
    """以独占创建模式写入 JSON 文件，避免覆盖已有结果。"""
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)


def main():
    """主流程：校验环境 -> 加载模型 -> 遍历问题生成代码 -> 评测并汇总指标。"""
    args = get_args()
    # 校验上游 LiveCodeBench 仓库版本，确保评测逻辑与预期一致
    commit = subprocess.check_output(
        ["git", "-C", str(args.lcb_repo), "rev-parse", "HEAD"], text=True).strip()
    if commit != LCB_COMMIT:
        raise RuntimeError(f"Unexpected LiveCodeBench commit: {commit}")
    # 上游 prompts 包中使用仓库相对路径加载文件，因此需要将仓库加入 sys.path 并切换工作目录
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(args.lcb_repo))
    os.chdir(args.lcb_repo)
    # 导入上游 LiveCodeBench 模块和 Transformers 模型加载工具
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.prompts.code_generation import get_generic_question_template_answer
    from lcb_runner.utils.extraction_utils import extract_code
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.evaluation import codegen_metrics
    from lcb_runner.evaluation.pass_k_utils import compute_metrics_from_results
    from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, set_seed
    import torch

    # 拒绝任何对上游评测/提示/数据代码的本地修改，保证评测结果可复现
    changed = subprocess.check_output([
        "git", "diff", "HEAD", "--name-only", "--", "lcb_runner",
    ], cwd=args.lcb_repo, text=True).strip()
    if changed:
        raise RuntimeError(f"Upstream code has local modifications:\n{changed}")
    # 构建数据索引并按 max_samples 截取子集
    index, counts = index_data(args.data_dir)
    print(f"release_v6 actual count: {len(index)}; shards: {counts}", flush=True)
    selected = index[:args.max_samples] if args.max_samples is not None else index
    # 非校验模式下，拒绝覆盖已有输出目录，防止丢失历史结果
    if not args.check_only and args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing directory: {args.output_dir}")
    # 使用 local_files_only=True 避免联网下载，确保使用本地缓存的模型文件
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    generation_config = GenerationConfig.from_pretrained(args.model, local_files_only=True)
    # 构建生成参数：temperature>0 时采样，否则贪心解码
    kwargs = dict(do_sample=args.temperature > 0, max_new_tokens=args.max_tokens,
                  num_beams=1, num_return_sequences=1, repetition_penalty=1.0)
    if args.temperature > 0:
        kwargs.update(temperature=args.temperature, top_p=args.top_p, top_k=0)
    else:
        # 贪心解码时中和模型配置中继承的采样默认值，避免影响确定性输出
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 0
    # 设置停止字符串，用于在生成时提前终止
    if args.stop:
        kwargs["stop_strings"] = args.stop
    generation_config.update(**kwargs)
    # 构建完整运行配置字典，记录所有影响评测结果的参数，用于结果复现
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
        # 记算评测模块每个文件的 SHA256，用于追踪评测代码版本
        evaluator_sha256={str(p.relative_to(args.lcb_repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((args.lcb_repo / "lcb_runner/evaluation").glob("*.py"))},
    )
    print(json.dumps({"generation_args": kwargs, "n": args.n}, indent=2), flush=True)
    model = None
    if not args.check_only:
        # 创建输出目录并保存运行配置
        args.output_dir.mkdir(parents=True, exist_ok=False)
        dump(args.output_dir / "run_config.json", config)
        # 设置随机种子保证可复现性
        set_seed(args.seed)
        # 加载模型到自动选择的设备上，使用自动精度避免显存溢出
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype="auto", device_map="auto", local_files_only=True)
        model.eval()
    # 存储所有问题的评测结果，用于最终汇总
    all_results = {}
    # 遍历选中的每个问题，进行代码生成和评测
    for position, (task_id, shard, offset) in enumerate(selected):
        # 通过分片文件和字节偏移量定位并读取单个问题数据，避免全量加载
        with (args.data_dir / shard).open("rb") as stream:
            stream.seek(offset)
            question = CodeGenerationProblem(**json.loads(stream.readline()))
        # 使用上游官方提示模板构建问题提示
        prompt = get_generic_question_template_answer(question)
        messages = [{"role": "user", "content": prompt}]
        # 应用 chat template，enable_thinking 控制是否启用思考模式
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=args.enable_thinking,
            return_dict=True, return_tensors="pt")
        # 解码渲染后的提示文本，用于后续记录和调试
        rendered = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False,
                                    clean_up_tokenization_spaces=False)
        # 校验模式：仅验证提示和模板是否正确，不加载模型
        if args.check_only:
            print(f"Verified official prompt/chat template: task={task_id}, tokens={inputs['input_ids'].shape[-1]}")
            return
        inputs = inputs.to(model.device)
        # 记算提示长度，用于从生成结果中截取新生成的 token
        prompt_length = inputs["input_ids"].shape[-1]
        candidates = []
        # 对每个问题生成 n 个候选解答，用于计算 pass@k 指标
        for sample_id in range(args.n):
            with torch.inference_mode():
                output = model.generate(**inputs, generation_config=generation_config,
                                        tokenizer=tokenizer, **kwargs)
            # 截取新生成的 token（去掉输入提示部分）
            tokens = output[0, prompt_length:]
            raw = tokenizer.decode(tokens, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
            # 从原始输出中提取最终回答和思考完成状态
            final, finished = final_response(raw, args.enable_thinking)
            # 使用上游代码提取逻辑从回答中提取代码块
            code = extract_code(final, LMStyle.OpenAIChat)
            # 收集所有可能的 eos token id，用于判断生成是否正常结束
            eos_ids = generation_config.eos_token_id
            eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
            # 构建完整的候选记录，包含生成和诊断信息
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
            # 在代码执行前保存原始输出，确保评测中断时不丢失生成结果
            with (args.output_dir / "generations.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        # 逐问题评测以控制内存占用；使用上游评测函数，传入所有 n 个候选解答
        metrics = codegen_metrics(
            [question.get_evaluation_sample()], [[r["extracted_code"] for r in candidates]],
            num_process_evaluate=args.num_process_evaluate, timeout=args.timeout)
        all_results[position] = metrics[1][0]
        # 将评测结果写入结果文件，包含每个候选的官方评测结果和元数据
        with (args.output_dir / "results.jsonl").open("a", encoding="utf-8") as stream:
            for sample_id, record in enumerate(candidates):
                record["official_evaluation_result"] = metrics[1][0][sample_id]# 这里的 metrics[1][0] 是官方评测结果，
                record["official_evaluation_metadata"] = metrics[2][0][sample_id]# metrics[2][0] 是官方评测元数据
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"[{position + 1}/{len(selected)}] {task_id}: {metrics[0]['pass@1']}", flush=True)
        # 显式释放大对象并触发垃圾回收，避免长评测过程中内存累积
        del question, metrics, candidates, output, inputs
        gc.collect()
    # 从上游函数签名中提取默认的 k_list，保持与官方一致的 pass@k 计算口径
    import inspect
    k_list = inspect.signature(codegen_metrics).parameters["k_list"].default
    # 汇总所有问题的评测结果，计算 pass@k 等最终指标
    summary = compute_metrics_from_results(all_results, k_list=k_list)
    summary.update(num_tasks=len(selected), samples_per_task=args.n,
                   task_index_to_id={i: item[0] for i, item in enumerate(selected)})
    dump(args.output_dir / "summary.json", summary)
    # 打印汇总指标，排除 detail 和 task_index_to_id 等冗长字段
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ("detail", "task_index_to_id")}, indent=2))


if __name__ == "__main__":
    main()
