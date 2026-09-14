# MiniCPM5-2B MMLU-Pro evaluator

使用本地 Parquet、Transformers 和模型自带 chat template；不调用 API，不下载 benchmark，不计算其他模型。此实现遵循本任务指定的 prompt 和生成配置；尚未验证其得分与官方报告一致。

## 环境与数据

依赖：`torch`、`transformers`、`accelerate`、`pyarrow`、`PyYAML`。模型部署版本要求参见 `../../docs/deployment/transformers.md`。本机已有环境可用 `conda activate minicpm5`。

默认数据目录为 `/home/hongchang/whf/datasets/MiniCPM5_eval/MMLU-Pro/data/`，测试文件为 `test-00000-of-00001.parquet`（12032 条），示例文件为 `validation-00000-of-00001.parquet`（70 条，每类别 5 条）。只读取本地文件，缺失时报错。`--data_path` 可指定目录或测试 Parquet 文件；few-shot 文件默认从测试文件同目录读取，也可在 YAML 指定训练/validation 文件。

## 运行

以下命令均在 MiniCPM 根目录执行。

```bash
conda activate minicpm5
python benchmarks/mmlu_pro/evaluate.py \
  --config benchmarks/mmlu_pro/config.yaml \
  --model openbmb/MiniCPM5-2B \
  --data_path /home/hongchang/whf/datasets/MiniCPM5_eval/MMLU-Pro/data/ \
  --output_dir benchmarks/mmlu_pro/outputs/full
```

100 条 debug，使用已有本地模型权重：

```bash
python benchmarks/mmlu_pro/evaluate.py \
  --model /home/hongchang/whf/models/MiniCPM5-2B \
  --max_samples 100 \
  --output_dir benchmarks/mmlu_pro/outputs/debug100
```

完整 test，使用已有本地模型权重：

```bash
python benchmarks/mmlu_pro/evaluate.py \
  --model /home/hongchang/whf/models/MiniCPM5-2B \
  --output_dir benchmarks/mmlu_pro/outputs/full_local
```

Hub 模型 ID 由 Transformers 加载；本地模型路径避免下载模型。脚本也支持进入此目录后执行 `python evaluate.py`；输出目录相对运行时工作目录。已有 `results.jsonl` 时拒绝覆盖，请为新运行指定新的输出目录。脚本不支持断点续跑。

## Prompt 与配置

系统提示固定为：

```text
You are an knowledge expert, you are supposed to answer the multi-choice question to derive your final answer as `The answer is ...`.
```

同类别示例按 validation 原始顺序选取，默认 5-shot；相同 question_id、相同题目以及重复示例被排除，不从 test 选示例。示例不足时明确报错。示例内容格式为：

```text
Q:
question

Options are:
(A): option
...

A:
Let's think step by step.
已有 cot_content 的推理正文
The answer is (X).
```

保留数据推理正文，只去除重复的 `A:`、思考前缀和末尾答案包装，再按上述格式渲染。最后追加当前测试题，止于 `Let's think step by step.`，不包含测试答案或测试 CoT。没有自行拼接模型特殊 token。

`config.yaml` 管理模型、路径、示例数、seed、样本上限及生成参数。默认 `do_sample=True, temperature=1.0, top_p=0.95, max_new_tokens=1024`，seed 为 42。`--max_samples` 选择文件中前 N 条，不是随机或分层抽样；省略则使用配置，默认完整 test。采样结果跨硬件或库版本不保证逐字一致。

## 输出与解析

每题立即写入 `results.jsonl`，字段为 `question_id, category, gold_answer, prediction, correct, model_output`。数据缺少 question_id 时使用零起始测试行号。

`model_output` 是所有新生成 token 的原始解码（`skip_special_tokens=False`），保留思考内容和终止 token。开头的 `<think>` 由官方 chat template 放在输入中，所以新生成文本不一定含此开头标签。

解析取最后一个独立的大写 A-J，不匹配英文单词内部的字母，支持 `The answer is (A)`、`The answer is A`、`Answer: A`。若有 `</think>`，仅从其后的最终回答解析；否则在生成文本中解析。无匹配时 prediction 为 null，计为错误。达到长度上限的输出仍保存并按同一规则解析；不会重试或自动增加长度。

运行结束打印总数、正确数及整体 Accuracy（正确数 / 总数，范围 0–1），同时写入 `summary.json` 和有效配置 `run_config.yaml`。异常中断时已完成行仍在 JSONL 中，但不输出完整评测汇总。

原有 `test_inference.py` 和 `test_inference_output.txt` 保留，供查看此前单样本推理。
