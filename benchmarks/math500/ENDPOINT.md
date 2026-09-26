# Author-email MATH-500 protocol

Start the local service in one terminal:

```bash
bash benchmarks/math500/serve_endpoint.sh
```

Run the evaluator from the MiniCPM root in another terminal:

```bash
conda run --no-capture-output -n minicpm5 python benchmarks/math500/evaluate.py
```

The default config evaluates all 500 original local test rows, once per row. It
does not read `solution` into a sample or send the gold answer to the endpoint.
Existing nonempty output directories are rejected. `--max_samples 1 --output_dir
benchmarks/math500/outputs/author_endpoint_smoke` runs one question only.

The request contains exactly `model`, a single user message, `temperature=1`,
`top_p=0.95`, `n=1`, and `chat_template_kwargs.enable_thinking=true`. The message is
the original problem, one newline, then:

```text
Please reason step by step, and put your final answer within \boxed{}.
```

No system message, demonstrations, manually inserted thinking tokens, top_k,
max_tokens, max_completion_tokens, max_new_tokens, reasoning budget, stop, or
stop_strings are sent. Requests are printed and saved verbatim in requests.jsonl.
There are no automatic retries or replacement samples.

The local vLLM service uses the existing MiniCPM5-2B model directory and
max-model-len=131072. `--generation-config vllm` prevents generation_config.json
from imposing an output cap. With no API output limit, this vLLM implementation
uses the remaining context capacity; context length still bounds prompt plus
output. GPU memory utilization and max-num-seqs are resource/concurrency controls,
not token budgets. No reasoning parser is enabled, so literal closing thinking
markers remain in content. The client verifies the advertised context length
before generating.

The unchanged parser is called only on content after the last `</think>`.
Without that marker the parsed answer is None, final_response is empty, and the
unchanged grader returns a failure. Server-separated reasoning is preserved in
endpoint_response, never used to recover an unfinished answer. All response JSON,
requests, usage, finish_reason, and frozen parser/grader hashes are saved.

`ended_with_eos` is null because OpenAI-compatible finish_reason=stop alone does
not identify the actual terminal token. Summary records eos_unknown separately;
length_limit_reached comes from finish_reason=length.

Old Transformers ablation YAMLs are not compatible with this strict entrypoint;
they remain untouched as historical experiment records. IFEval, LiveCodeBench,
the dataset, parser, and grader are not modified.
