# 固定 50 题实验

在项目根目录运行，使用 minicpm5 环境：

```bash
python benchmarks/ifeval/evaluate.py --config benchmarks/ifeval/greedy_4096.yaml
python benchmarks/ifeval/evaluate.py --config benchmarks/ifeval/sampling_4096.yaml
```

两组均直接读取 `validation/diagnostic50/data.jsonl`，0-shot、thinking=True、
每题生成一次，预算 4096。sampling 使用 temperature=1.0、top_p=0.95。
不添加 system 或修改题目，不更改 vendor 内的评分器。

评分口径固定为 `final_only`，与诊断集基线一致：只有 thinking_finished 时才取
最后一个 `</think>` 后的文本并 strip；否则传空字符串给原版 evaluator。
完整生成文本仍保存在 model_output，实际评分文本保存在 scored_response。
旧 full 目录是完整输出评分口径，不应直接拿其旧分数与新实验比较。
默认 config.yaml 的评分口径也已切换为 final_only，生成参数未改变。

每题保存 ended_with_eos、length_limit_reached、thinking_finished、
unfinished_thinking、final_started_but_truncated。
summary.json 中对应结束状态字段是题数，avg_generated_tokens 是全部已完成题目的
平均生成 token 数（包含失败题），并保留四项 IFEval 指标。

final_started_but_truncated = thinking_finished 且达到 token 预算且未以 EOS 结束。
该字段表示进入了 final 阶段，不要求 final 正文非空。
EOS 恰好出现在预算末端时，ended_with_eos 和 length_limit_reached 同时为 true，
但 final_started_but_truncated 为 false。

配置、实际传参、合并模型默认配置、langdetect_seed 等记录在 run_config.yaml。
输出目录必须不存在，避免覆盖任何已有实验。
这 50 题是失败集中的诊断集，不能用于估计全量准确率或对标官方 86.7。

## 完整数据 8192-token 对照（先做单题 smoke）

A `sampling_8192_final.yaml` 使用原始 user prompt，无 system。
B `prompt_8192_final.yaml` 仅增加配置中逐字保存的独立 system message。
两组均指向原始 541 题数据，sampling temperature=1.0、top_p=0.95、
max_new_tokens=8192、单 beam、每题一次，不加 repetition_penalty。
B 是 system prompt 消融，未确认等同作者评测协议。

单题验证使用独立目录，不占用全量输出目录：

```bash
python benchmarks/ifeval/evaluate.py --config benchmarks/ifeval/sampling_8192_final.yaml --max_samples 1 --output_dir benchmarks/ifeval/outputs/smoke_sampling_8192_final
python benchmarks/ifeval/evaluate.py --config benchmarks/ifeval/prompt_8192_final.yaml --max_samples 1 --output_dir benchmarks/ifeval/outputs/smoke_prompt_8192_final
```

全量命令（本次不执行）：

```bash
python benchmarks/ifeval/evaluate.py --config benchmarks/ifeval/sampling_8192_final.yaml
python benchmarks/ifeval/evaluate.py --config benchmarks/ifeval/prompt_8192_final.yaml
```
