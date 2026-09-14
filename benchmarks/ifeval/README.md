# MiniCPM5-2B IFEval（离线）

使用本地数据和模型，复用 lm-evaluation-harness 的 IFEval 评分模块。
这是指定的 harness IFEval 规则加 MiniCPM thinking chat template，不声称复现官方 86.7 的未知协议。

当前两组 4096-token、固定 50 题实验请见 [EXPERIMENTS.md](EXPERIMENTS.md)。

## 固定协议

- 原始 `prompt` 作为唯一 user 消息，无 system、few-shot 或附加指令。
- `tokenizer.apply_chat_template(..., enable_thinking=True, add_generation_prompt=True)`。
- 每题一次生成，`do_sample=False, temperature=0.0, max_new_tokens=1280`。
- `model_output` 保存包含特殊 token 的完整生成文本；`scored_response` 是
  `tokenizer.decode(..., skip_special_tokens=True)` 的完整文本。
- 固定 `final_only`：沿用诊断基线，只在 thinking 完成时取最后一个 `</think>`
  后的文本并 strip；否则评分输入为空字符串。完整生成不删改，保存在 model_output。
- `scored_response` 保存实际送入上游 `process_results(doc, [response])` 的最终回答。
- prompt accuracy 使用均值；instruction accuracy 使用上游 `agg_inst_level_acc`，
  按所有指令计分，不平均每题的指令准确率。

`vendor/lm_eval/tasks/ifeval/` 中六个上游文件原样复制自本机已有仓库，
commit 为 `dd417662e5bda6a247489ec28d2ff46a45d1c42c`。
`vendor/provenance.json` 记录来源和 SHA256；每次运行校验文件未改。
空的包初始化文件仅用于隔离导入，未修改任何评分规则。

## 离线依赖

使用已有 `minicpm5` 环境。评分额外需要 `nltk>=3.9.1`、`langdetect`、
`immutabledict`、`packaging` 和本地 NLTK `punkt_tab` 资源。
程序先检查资源，再导入上游模块，防止上游自动下载。
缺依赖会直接失败，不回退到其他 grader。

如有本地 wheelhouse，可执行（替换为真实本地目录）：

```bash
python -m pip install --no-index --find-links /path/to/local/wheels -r benchmarks/ifeval/requirements.txt
```

本地 NLTK 目录应包含 `tokenizers/punkt_tab/english/` 及其数据文件。
通过 `NLTK_DATA=/path/to/nltk_data` 或 YAML 的 `nltk_data` 指定。不执行 `nltk.download()`。

## 只检查环境，不加载模型

在 MiniCPM 根目录：

```bash
conda activate minicpm5
python benchmarks/ifeval/evaluate.py --check_only
```

## 5 题 smoke test

```bash
python benchmarks/ifeval/evaluate.py \
  --config benchmarks/ifeval/config.yaml \
  --max_samples 5 \
  --output_dir benchmarks/ifeval/outputs/smoke5
```

命令中的 `--max_samples 5` 限制为前 5 题。输出目录必须不存在，避免覆盖旧结果。
数据中的 URL 仅作为 prompt 文本传给模型，不抓取页面。

## 输出

- `results.jsonl`：原始 prompt、key、instruction_id_list、kwargs、渲染后的
  chat prompt、完整 model_output、实际 scored_response、生成 token 数及四项逐题判分。
- `summary.json`：完成数及四项指标 `prompt_level_strict_acc`、
  `inst_level_strict_acc`、`prompt_level_loose_acc`、`inst_level_loose_acc`。
  分数范围为 0–1；instruction 逐题结果为 bool 列表。
- `run_config.yaml`：最终配置、实际 generation kwargs、合并模型默认值后的
  generation config、chat template、每题生成次数、数据和上游代码哈希、依赖版本。
  greedy 模式下，模型默认的 top_p 等采样设置不参与采样；可能出现 Transformers 提示。

本次仅做静态和本地输入检查，没有运行 5 题推理或全量推理。
