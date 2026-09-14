# MATH-500 生成对照实验

本目录是探索实验，不是 MiniCPM5 官方复现配置。每组从原始 problem 重新生成，不续写旧结果，不把旧结果和新结果混合计分。

## 实验组

| 配置 | max_new_tokens | do_sample | repetition_penalty | 用户消息 |
|---|---:|---|---:|---|
| baseline_8192.yaml | 8192 | true | 1.0 | 原始 problem |
| sampling_16384.yaml | 16384 | true | 1.0 | 原始 problem |
| penalty_16384.yaml | 16384 | true | 1.05 | 原始 problem |
| prompt_16384.yaml | 16384 | true | 1.0 | problem + 下方固定指令 |
| greedy_16384.yaml | 16384 | false | 1.0 | 原始 problem |

全部使用同一本地模型，`enable_thinking=true, temperature=1.0, top_p=0.95`。greedy 模式下 temperature/top_p 不参与采样。penalty 为 Transformers 的 `repetition_penalty`；1.05 是探索值，不等同于其他引擎的惩罚实现或官方设置。

prompt 组仅追加以下固定文本，不改写题目，不提供答案或 solution：

```text
Solve the problem concisely. Once you have established and checked the result, state your final answer in \boxed{...} and stop. Avoid repeating the same verification.
```

## 固定样本与可比性

所有组使用同样 12 题：

- 原先截断的 8 题：9、24、96、142、152、166、425、499。
- 原先正常答对的 4 题：59、84、367、458。

来源是 `outputs/full_16384/results.jsonl`（该历史运行实际预算为 8192）。固定包含 q9，其余 7 条从截断样本中以 Python Random(42) 抽取，再用同一个 RNG 抽取 4 条正常且正确的对照题。样本按原始数据顺序运行。

每题重置随机种子为 `42 + question_id`，避免前题生成长度改变后题采样状态。因此需重新跑 8192 基线，不能直接比较旧的单次全局 seed=42 结果。该子集刻意包含大量失败题，其准确率不代表完整 MATH-500，不能直接与官方 94.6 对比。

## 运行

在项目根目录执行：

```bash
conda activate minicpm5
python benchmarks/math500/run_experiment.py --config benchmarks/math500/experiments/baseline_8192.yaml
python benchmarks/math500/run_experiment.py --config benchmarks/math500/experiments/sampling_16384.yaml
python benchmarks/math500/run_experiment.py --config benchmarks/math500/experiments/penalty_16384.yaml
python benchmarks/math500/run_experiment.py --config benchmarks/math500/experiments/prompt_16384.yaml
python benchmarks/math500/run_experiment.py --config benchmarks/math500/experiments/greedy_16384.yaml
```

默认保存到 `outputs/experiments/<实验名>/`，已有目录不会覆盖。中断后给同一命令加 `--resume`，只跳过已完整保存的题；当前题会从头运行。配置、代码、数据和运行时改变后拒绝续跑。若文件有不完整 JSON 行，需先保留并检查该文件，不会自动删除。

可分终端启动。每个进程都会单独加载模型和 KV cache；单 GPU 显存不足会 OOM，共用 CPU 时会争抢算力。多 GPU 可用 `CUDA_VISIBLE_DEVICES` 分配不同 GPU；单设备建议先顺序运行。并行时不要将耗时差异归因于解码参数。

每组保存 `results.jsonl`、实时 `summary.json`、`run_config.yaml` 和 `runtime.json`。包含真实渲染 prompt、完整输出、实际合并后的 generation 配置、数据/代码哈希、每题种子、EOS、token 数、判分原因及耗时。

## 对比

```bash
python benchmarks/math500/compare_experiments.py \
  benchmarks/math500/outputs/experiments/baseline_8192 \
  benchmarks/math500/outputs/experiments/sampling_16384 \
  benchmarks/math500/outputs/experiments/penalty_16384 \
  benchmarks/math500/outputs/experiments/prompt_16384 \
  benchmarks/math500/outputs/experiments/greedy_16384 \
  --output benchmarks/math500/outputs/experiments/comparison.json
```

各组必须完整、使用相同样本与判分代码。优先看“原先截断题中，正常 EOS 且答对的数量”，同时看对照题是否退步、平均 token 是否降低。`sampling_16384` 对比 8192 用于检查预算；其他三组都应对比 `sampling_16384`，避免一次归因于多个变量。减少重复不代表提高正确率。固定测试集上的探索结果不能作为独立验证成绩。
