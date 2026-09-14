# 固定 50 题诊断集

基于 `outputs/full/results.jsonl` 的 541 次既有生成选择，不重新生成模型输出。

| 分组 | 原结果数量 | 选择数量 | 验证目的 |
|---|---:|---:|---|
| unfinished_thinking | 20 | 20 | 增加预算后能否结束 thinking |
| truncated_final | 30 | 15 | 最终回答能否完整生成；不保证截断前评分一定失败 |
| completed_failure | 56 | 10 | 已 EOS 结束，最终回答仍违反指令 |
| thinking_contamination | 232 | 3 | 最终回答正确、完整文本评分失败的对照 |
| stable_correct | 203 | 2 | 两种口径均正确，检查改动是否损伤已有能力 |

选择先覆盖失败指令类型，再覆盖指令类型和生成长度区间，同分按 key 升序。
输出保留原始数据顺序，共 50 个唯一 key、22 类指令；key 是数据集标识，不是行号。
每组实验必须使用同一份 `diagnostic50/data.jsonl`，不能在看过新实验分数后换题。

- `diagnostic50/data.jsonl`：从原数据逐条选出，prompt / kwargs / instruction IDs 完全不变。
- `diagnostic50/selection.jsonl`：逐题分组、原始分数、最终回答分数、失败指令及结束状态。
- `diagnostic50/manifest.json`：完整题号、配额、源数据/结果/代码哈希、基线分数。
- `select_validation.py`：可重复执行的离线选择脚本，要求写入新目录。

诊断评分仍使用原版 harness evaluator：thinking 完成才取 `</think>` 后文本，
否则用空回答。该口径是显式的诊断选择，未确认等同于作者协议。

本集合刻意集中失败样本，不能用于估计完整数据集准确率或对标官方 86.7。
旧生成在此集合的 final-only 指标：prompt strict 22%、instruction strict 46.94%、
prompt loose 26%、instruction loose 51.02%。

后续两种生成设置（尚未执行）：
1. 原始 prompt、thinking=True、greedy、max_new_tokens=4096。
2. 同样 prompt/thinking/预算，sampling=True、temperature=1.0、top_p=0.95。

应比较各组的 thinking 完成数、EOS 数、四项指标、旧错新对及旧对新错，
不能只报告这 50 题的总准确率。旧 1280-token 生成可直接作基线，不必再跑。

本次未修改 evaluate.py。它目前固定 1280 和 full_decoded_response，
正式跑以上实验前还需要明确放开预算/采样配置并支持 final-only 评分。
`--max_samples 50` 单独使用会取原数据前 50 题，并不会使用本诊断集；
实验 YAML 的 data_path 必须指定 `benchmarks/ifeval/validation/diagnostic50/data.jsonl`。
