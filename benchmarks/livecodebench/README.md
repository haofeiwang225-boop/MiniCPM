# MiniCPM5-2B / official LiveCodeBench

This adapter imports the local official repository; it does not copy or modify
its evaluator, code parser, prompt builder, test cases, or pass@k implementation.

Pinned LiveCodeBench commit: `28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24`.
Pinned `livecodebench/code_generation_lite` data revision:
`0fe84c3912ea0c4d4a78037083943e8f0c4dd505`.

`release_v6` contains **1055** questions: 400 + 111 + 101 + 101 + 167 + 175
from `test.jsonl` through `test6.jsonl`. It is not the 175-question `v6` subset.
Every run verifies the six original LFS SHA256 hashes and counts the actual rows.
The default input is the existing Hugging Face snapshot, entirely offline.
Reading the unchanged JSONL files directly avoids datasets 5's removal of dataset
scripts. Each selected row is passed unchanged to official `CodeGenerationProblem`.

## Protocol and evidence

- `lcb_runner/runner/parser.py`: default n=10, temperature=0.2, top_p=0.95,
  max_tokens=2000, stop="###", evaluator timeout=6, evaluation workers=12.
- `lcb_runner/runner/vllm_runner.py`: default frequency/presence penalties are 0.
  The adapter uses Transformers sampling with top_k=0 (no top-k filter), neutral
  repetition penalty, and one sequence per call, repeated n times. Seed=0 is
  recorded. Transformers and vLLM are not expected to produce identical samples.
- `lcb_runner/prompts/code_generation.py:get_generic_question_template_answer`:
  official question, format instructions, starter code where present, and answer
  header. This exact string is the single user message. No system or few-shot
  messages are added. `apply_chat_template(enable_thinking=True)` renders it.
- `lcb_runner/utils/extraction_utils.py:extract_code`: use the standard fenced-code
  branch (`LMStyle.OpenAIChat` only selects that branch; no OpenAI API is used).
  Only the substring after the last `</think>` is submitted. Without that marker,
  the submitted response/code is empty. Raw decoding preserves special tokens.
- `lcb_runner/evaluation/compute_code_generation_metrics.py:codegen_metrics`:
  official code execution and test results; invoked per question to bound memory.
  Final results are aggregated by the unchanged official
  `pass_k_utils.compute_metrics_from_results`, using the original k list. With
  n=10 this reports pass@1, pass@5 and pass@10. No custom correctness rules.

The MiniCPM5 model card recommends temperature=1.0, top_p=0.95, but does not
establish its complete LiveCodeBench 69.1 protocol. The defaults here reproduce
the **LiveCodeBench repository defaults**, not a verified reconstruction of that
model-card score. In particular, a 2000-token budget or the `###` stop string can
end thinking early. Any changes via `--max_tokens`, `--temperature`, `--top_p`,
or `--stop` must be reported as experiment settings; do not label them official
MiniCPM5 settings. `--stop` with no following values disables stop strings.

## Run

From the MiniCPM root, in the `minicpm5` environment:

```bash
python benchmarks/livecodebench/evaluate.py --max_samples 5 \
  --output_dir benchmarks/livecodebench/outputs/smoke5

python benchmarks/livecodebench/evaluate.py \
  --output_dir benchmarks/livecodebench/outputs/release_v6_full
```

The smoke run also uses n=10 (50 generations for 5 questions). Existing output
directories are rejected. `--check_only` checks data, imports, and the first
rendered prompt without loading model weights. No model experiment was launched
during implementation.

Additional official import dependencies installed in minicpm5 are
`anthropic==0.42.0` and `pebble==5.2.2`. Anthropic is needed by upstream package
imports only; no API is called. The optional `prepare_data.py` downloader is not
needed on this machine because the official dataset is already cached.

## Output

- `run_config.json`: repository commit, evaluator hashes, dataset revision/hashes,
  actual counts, model, seed, generation kwargs and resolved generation config.
- `generations.jsonl`: each candidate saved before evaluation, including question
  ID, sample index, official user prompt, rendered prompt, raw/final responses,
  extracted code, generated tokens, thinking status, EOS/length status, kwargs.
- `results.jsonl`: the same candidate records plus official per-test results and
  evaluator error metadata. One row per candidate (10550 rows at full n=10).
- `summary.json`: official pass@k, details, and index-to-question-ID mapping.

Raw stop strings/EOS tokens remain in the audit response as emitted by Transformers;
the official code parser still only extracts fenced code from the final response.
