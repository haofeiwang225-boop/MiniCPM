# Single-layer rank 4 → capacity 5 gain experiment

This is an independent short experiment, not a continuation of the historical
AdamW trajectory and not a change to the formal training pipeline.

## Fixed controls

- Base: `/home/hongchang/whf/models/MiniCPM5-2B` (real safetensors, every tensor verified).
- PRE: `finetune/adaptive_lora/outputs/metamath_pilot_geo1600_rebuilt_20261013_221250_seed42/correction_checkpoints/step_1600_pre.pt`.
- All 84 Dense q/v adapters are restored and compared exactly with PRE.
- Target: `model.layers.16.self_attn.q_proj`; optional CLI target layer 18.
  Both paths require an actual saved rank of 4, otherwise fail.
- A: old rank 4; B: old rank 4 + projected random rank-1 branch;
  C: old rank 4 + tangent-residual leading-right-singular-vector rank-1 branch.
- B/C have unit-norm FP32 added A, perpendicular to the old A row space,
  zero added B, and identical additive scaling 1. The old factor scaling stays
  unchanged. Capacity 5 does **not** mean numerical rank is 5 at insertion.
- C uses `R = G - Q_B(Q_B.T G) - (G Q_A)Q_A.T + Q_B(Q_B.T G Q_A)Q_A.T`,
  reusing the repository's residual and rank-aware orthogonal-basis functions.
  `G` is the equal-example mean gradient of the real masked LM loss with respect
  to the target frozen weight, on four training-source probe batches. No dev
  example participates in direction generation.
- All arms start from the same factors and use a **fresh AdamW**, same seed,
  data order, LR, weight decay and batch size 1. No scheduler, constant LR.
  **Dropout is disabled in all three arms** (`eval()` with autograd enabled),
  so an extra branch does not consume different dropout RNG and confound the
  comparison. This is a documented isolated experimental setting, not the
  original training dropout protocol. Base and 83 non-target adapters stay frozen.
- Defaults: 30 updates, 64 training-pool examples, 64 validation examples,
  4 disjoint probe examples, max length 2048, seed 42, LR 5e-5, decay .01.
  With 30 steps and batch size 1, only 30 of the 64 training-pool examples are
  actually visited. The precise visited source indices are saved.

## Reuse and scope

`rank_gain_probe_support.py` is a verbatim snapshot of eight loading/checking
functions from the validated local `minicpm_rank1_safe_probe_v3.py`, with its
source SHA256 recorded. That source is currently untracked; the snapshot makes
this branch runnable without modifying or committing the user's v3 script.
It directly loads the original official LoRA class, and uses ordinary HF loading
without the training loader's `path + state_dict` workaround. It rejects missing,
unexpected and mismatched weights. `ExpandedLoRA` is imported unchanged from the
existing implementation. No dependency installation is required.

The local probe uses `autograd.grad` rather than importing the official probe's
DeepSpeed registry. It computes only the same target base-weight gradient,
restoring all parameter flags/grad references, module modes and torch RNG.
No optimizer is created for probing. No simulated gradients are used by the
real experiment.

## Data and validation

The existing MetaMath pilot train/dev files are matched exactly against the raw
MetaMathQA source by query and answer. Sample and original-question hashes are
recorded. Entire train/dev source pools must have no shared sample or original
question group. Probe and continuation-training pools are selected without group
overlap using a local fixed RNG. No NLL is used in selection. Insufficient groups,
unmatched source records, empty supervision or a truncated answer cause failure.
The existing `ChatTrainingDataset` provides the real assistant-only labels with
no padding (batch size 1); effective token counts use shifted labels.

These are only within-experiment isolation guarantees: PRE has already seen the
historical training corpus; the existing dev set is **not a new blind test**.
Validation reports `sum(loss_mean * effective_tokens) / sum(effective_tokens)`
and every sample's loss, count and full-logit byte hash. B/C insertion must match
A's full logits for **all 64 validation examples**, not just the next token.
This checks those inputs, not every possible input.

## CPU unit tests (no GPU)

```bash
cd /home/hongchang/whf/MiniCPM
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONPATH=.:finetune \
  /home/hongchang/miniconda3/envs/minicpm5/bin/python -B -m unittest \
  test_minicpm_single_layer_rank_gain -v
```

Tests cover group isolation, actual label masking and refusal to truncate,
probe averaging/state restoration/exception handling, both initialization
geometries, full-logit preservation, CPU tiny three-arm optimization with frozen
non-target weights, token-weighted NLL, truthful failure reports, and loader reuse.
These tests do not constitute a real MiniCPM GPU result.

## Recommended real GPU command (not executed during implementation)

Run alone after checking `nvidia-smi`; default minimum free VRAM is 30 GiB,
which is only an admission check, not a guarantee against OOM. BF16 base, FP32
added factors, eager attention; all arms run sequentially on one GPU.

```bash
cd /home/hongchang/whf/MiniCPM
OUT="finetune/adaptive_lora/outputs/rank4_to5_layer16_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=.:finetune \
  /home/hongchang/miniconda3/envs/minicpm5/bin/python -B minicpm_single_layer_rank_gain.py \
  --module model.layers.16.self_attn.q_proj \
  --steps 30 --train-samples 64 --validation-samples 64 --probe-batches 4 \
  --max-length 2048 --seed 42 --lr 5e-5 --weight-decay 0.01 \
  --output-dir "$OUT"
```

To select the unrun backup, replace only `--module` with
`model.layers.18.self_attn.q_proj` and use another unique output directory.

Full CLI: `--model`, `--adapter`, `--module`, `--train-data`, `--dev-data`,
`--source-data`, `--output-dir`, `--steps`, `--train-samples`,
`--validation-samples`, `--probe-batches`, `--max-length`, `--seed`, `--lr`,
`--weight-decay`, `--min-free-gib`. Paths default to the fixed local inputs above.

## Artifacts and interpretation

The directory must not exist. `report.json` includes versions, Git/code/data/model
hashes, configuration, probe statistics, timing, peak allocated/reserved memory,
per-arm validation before/after, gradients, parameter counts and invariants.
`A/B/C_results.json`, `A/B/C_train.jsonl`, selected-data copies and
`data_manifest.json` preserve the sample mapping. Target-factor `.pt` files are
explicitly **not** full adapters or PEFT checkpoints. No historical file is written.
All 381 frozen base weights are rechecked against safetensors after each arm;
all other adapters and old factor identities/scaling are checked.

Exceptions after output creation write `FAIL` with the real traceback and partial
results. Existing output directories are refused. A process killed externally
(e.g. SIGKILL) cannot finalize a Python report; never interpret `RUNNING` or a
missing report as PASS. Success requires both added factors to change and receive
real nonzero LM gradients, not just weight decay. There is no gain threshold:
PASS means experimental checks succeeded, not that rank expansion improved NLL.
The historical step-1600 AdamW state remains unavailable and unverified.
