# MiniCPM residual-SVD rank gain: fixed final research protocol

This protocol is fixed before the new GPU runs. It follows an already used Dev
set and a prior 30-step negative/inconclusive result; it is **not a blind test**.
No GPU audit or training is launched by implementation or unit tests.

## Scope and reuse

New entry: `minicpm_rank_gain_decisive.py`; CPU statistics:
`minicpm_rank_gain_statistics.py`; tests: `test_minicpm_rank_gain_decisive.py`.
Original `minicpm_single_layer_rank_gain.py`, `rank_gain_probe_support.py`, tests,
training entry points, checkpoints and old results are unchanged. We import the
validated real safetensors loader, all-84 adapter restore, `ExpandedLoRA`,
rank-aware orthogonal basis, tangent residual, direction normalization and
full-logit/NLL evaluation. Missing/mismatched weights fail, never random-load.

Only layers 16 and 18 q_proj, saved rank 4, are permitted. Seeds are exactly
42,43,44,45,46. A keeps rank 4; B adds unit projected-random A / zero B; C adds
unit residual-leading-right-singular A / zero B. Both added factors use FP32,
unit branch scaling, identical old A/B and old scaling at insertion.
Effective capacity becomes 5; numerical update rank is still 4 at insertion.
All 84 adapters are loaded; only the target old/new factors are trainable.

Every arm uses fresh AdamW (LR 5e-5, decay .01, default betas .9/.999, eps 1e-8),
200 steps, batch size 1, dropout disabled, constant LR, no scheduler. Historical
step-1600 Adam moments are unavailable and are never restored or fabricated.
No intermediate validation step is selected as the endpoint.

## Fixed samples and independent RNG streams

Default split seed is `20261026`, constant across both modules and all seeds.
Exact query/response matching to the raw MetaMathQA JSONL supplies sample and
original-question hashes. Full pilot train/dev pools must be group-disjoint.
A fixed permutation of unique training groups selects 200 training samples,
32 construction samples, then 32 independent audit samples; no overlap within
these three sets. All 400 existing Dev samples (GSM200/MATH200) retain their
original order across **all** coordinates. Samples are not selected by loss.

`SHA256('rank_gain_decisive_v1:{seed}:{domain}')[:8]` defines separate 32-bit
streams for `training_order`, `random_direction`, and `probe_order`. The training
stream permutes the same 200 samples once. A/B/C in a coordinate share exactly
this order. The probe stream permutes the fixed construction set into two
16-sample halves. The random-direction stream only initializes B's direction.
Sample membership is controlled solely by the fixed split seed.

CLI `--split-seed` is recorded and must match across all ten runs. Changing it
would constitute a new protocol, not an unreported tuning opportunity. The
same applies to changing data/model paths. The aggregator refuses mixed
settings, hashes, code, tokenization or software versions.

The inherited chat training dataset applies assistant-only causal label masking
and dynamic lengths with cap 2048. Incomplete answers are never silently cut;
overlength/empty supervised samples cause failure, not filtering/replacement.
PRE has already seen historical training data, so the isolation is only for
these current estimation/continuation/validation roles.

## Explicit token-weighted normalization

Let `n_i` count nonignored **shifted** labels, and `ell_i` be the model's mean
masked causal NLL for sample i. The objective is

`L = sum_i n_i ell_i / sum_i n_i`.

Construction/audit full-weight gradients accumulate per-sample mean gradients
in FP32 with coefficients `n_i / sum(n)`. Native BF16 base gradient arithmetic
retains its finite-precision limits; no FP32-full-model equivalence is claimed.
For batch-size-one updates, backward uses

`ell_i * n_i / mean_training_tokens`.

Thus a uniform-example SGD objective targets the same token risk, rather than
the old experiment's unweighted per-example objective. AdamW and one-pass
optimization still have order dependence. Every step records raw `ell_i`,
`n_i`, `n_i ell_i`, scaled backward objective and factor gradient norms.
This deliberate normalization change is specific to the new protocol.

Validation reports both token-weighted NLL and the unweighted per-question mean,
plus every question's NLL, NLL sum, tokens, GSM/MATH family, source indices/hashes.

## Gate 1: independent gradient direction audit (no training)

- Estimate `G_construct` on 32 construction samples, using two independently
  computed token-weighted 16-sample half gradients and their token counts.
- Use existing `R=(I-P_B)G_construct(I-P_A)` and its first right singular vector.
- Random direction is projected into the old A row-space orthogonal complement.
- Both added A have unit norm; zero B and unit scaling are identical.
- Estimate full `G_audit` on the disjoint 32 audit samples.
- Preserve full-gradient scores `S_full(a)=||G_audit a.T||_F^2` for B/C and
  their C/B ratio. Legacy `random_score` / `residual_score` still mean the
  **full-gradient** scores of the random / residual-SVD directions; their
  semantics are not changed. Explicit new fields are `full_gradient_scores`.
- Additionally compute `R_audit=(I-P_B)G_audit(I-P_A)` with the existing
  identity-free residual formula, and `S_res(a)=||R_audit a.T||_F^2` for B/C,
  saved as `residual_gradient_scores` with a separate C/B ratio. Audit basis
  QR and score reductions use float64. A zero B score produces a null ratio
  and explicit reason, never infinity or an invented win.
- Generate 128 unit random directions orthogonal to old A with an independent
  CPU generator. Its fixed seed is `seed_stream(20261026,
  'audit_random_reference_v1')`, independent of the experimental B-direction
  seed and global RNG. These reference directions never depend on audit loss
  or gradients. Save all residual scores, mean, median, linear-interpolated
  95th percentile, C's descending rank among the 128 references plus C, tie
  range/counts and midrank percentile. Rank 1 is best; ties are explicit.
- Record absolute cosine between the two **construction half** residual-leading
  right vectors. A vanished construction residual/rank-deficient factor fails
  explicitly, since no C direction exists. A zero audit gradient gives zero
  scores, null ratios and a failed resource gate.

The preregistered **heuristic resource screen** is:

1. C residual audit score **strictly exceeds** the 128 reference scores' 95th
   percentile (equality fails).
2. Construction-half absolute cosine is **at least 0.20** (equality passes).

`audit_gate_pass` is their conjunction. Thresholds, individual checks and failure
reasons are logged. A failure means **方向证据不足、暂停本方案训练**, not proof that
this method is mathematically ineffective. A pass is not a mathematical or
statistical guarantee of generalization benefit. Thresholds must not be revised
in response to real audit scores.

The audit gradient never selects or adjusts C, B, or a reference direction. The
new gate is a logged resource-screen recommendation for human review after
`probe-audit`; it does not automatically launch training. This audit-only patch
also leaves the existing explicit `train` command unchanged, so the reviewer is
responsible for pausing training when the gate fails. If paused, the training
endpoint remains unobserved. The step-200 statistical success criteria below
are unchanged and separate from this heuristic screen.

## Gate 2: fixed complete paired training

A command runs one module/seed coordinate with all three arms sequentially.
Each starts from original PRE factors, same base, same order and fresh AdamW.
All 400 initial validation full-logit byte hashes must match PRE/A for B/C;
this is a check on those inputs, not a proof for every possible input.

Only step 200's Dev400 is the primary endpoint. Record per-arm before/after NLL,
all train steps, factor initial/final/change norms, actual added `B@A` Frobenius
norm (rank-one Gram identity), parameter counts, timings and peak GPU memory.
After every arm, all frozen base tensors are compared with safetensors; 83
other adapters, original factor identities and old rank/scaling must be intact.
In-memory target PRE factors are restored in `finally`, including exceptions.
No source checkpoint is ever saved over.

`--resume-from OLD` reads fully completed arms with exactly matching run/data/
code fingerprints and artifact hashes, copies them into a **new** output
folder, and skips their training. An incomplete arm is rerun from PRE for 200
steps; there is no mid-arm optimizer resume. A result without its final artifact
hash manifest is not considered completed. Corrupt artifacts fail loudly.
Completed all-three reuse avoids model loading. Old reports/directories are not
modified. SIGKILL cannot finalize a report: missing/RUNNING is never PASS.

## Predeclared statistics and decision

Require the complete grid of 2 modules × 5 seeds × A/B/C with matching Dev
question indices, sample/original-question SHA256, family, effective tokens,
software/code/data fingerprints and exactly 200 steps. Failed/incomplete runs
cannot silently disappear. Verify all artifact hashes and paired train orders.

For each coordinate, compute NLL as `sum(nll_sum)/sum(tokens)`; the primary point
estimate averages the ten paired C−B differences with equal coordinate weight.
C−A and B−A are also reported, along with per-seed/module absolute NLL and
contrasts, per-question improvements/harms/ties, distributions and GSM/MATH.
Question-level counts refer to the mean difference over coordinates; separate
coordinate counts are also retained.

Use **10,000 paired crossed bootstrap draws**, fixed RNG seed 42:

1. Resample original-question groups with replacement. Keep every sample in a
   selected group together, with multiplicity. Recompute token denominators.
2. Independently resample the five seed IDs with replacement.
3. Use exactly those same sampled question/seed IDs for all arms and both
   fixed module strata. Never treat repeated seed evaluations as new questions;
   do not bootstrap the two modules as if they were independent datasets.
4. Compute percentile 95% intervals for pooled and module contrasts.

Both problem and seed uncertainty are represented, but five seeds give limited
precision. Family/secondary intervals are descriptive, not simultaneous
family-wise guarantees. Existing Dev reuse limits generalization claims.

Report **观察到收益** only if all conditions hold:

- Pooled C−B ≤ −0.0005.
- Pooled C−B 95% upper bound < 0.
- At least 8/10 coordinates have C−B < 0.
- Both module mean C−B < 0.
- C−A 95% upper bound ≤ +0.0005.

The last condition explicitly preregisters **+0.0005 as the noninferiority
margin** for “C should not clearly worsen A”; it must not be selected after
seeing results. Report **观察到退化** if C−B 95% lower bound > 0 or C−A lower
bound > +0.0005. Otherwise report **证据不足**. Engineering FAIL is separate,
not a research conclusion. No outcome proves the mathematical method invalid.

## Commands (GPU commands provided for manual review only)

```bash
cd /home/hongchang/whf/MiniCPM
PY=/home/hongchang/miniconda3/envs/minicpm5/bin/python
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  "$PY" -B -m unittest test_minicpm_rank_gain_decisive test_minicpm_single_layer_rank_gain -v

CUDA_VISIBLE_DEVICES='' "$PY" -B minicpm_rank_gain_decisive.py dry-run \
  --module model.layers.16.self_attn.q_proj --seed 42 \
  --output-dir "finetune/adaptive_lora/outputs/decisive_dry_$(date +%Y%m%d_%H%M%S)_$$"
```

`dry-run` reads hashes, adapter structure, real source metadata and tokenizer;
it does not load the base model or execute a forward/probe/optimizer. All new
outputs must be absent. Each GPU command checks free memory (default 30 GiB,
not a guarantee against OOM). No other GPU workload should be assumed safe.

```bash
# Only direction audit; zero training updates.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PY" -B minicpm_rank_gain_decisive.py probe-audit \
  --module model.layers.16.self_attn.q_proj --seed 42 \
  --output-dir "finetune/adaptive_lora/outputs/decisive_probe_l16_s42_$(date +%Y%m%d_%H%M%S)_$$"

# Only after human review: one full paired A/B/C coordinate.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PY" -B minicpm_rank_gain_decisive.py train \
  --module model.layers.16.self_attn.q_proj --seed 42 \
  --output-dir finetune/adaptive_lora/outputs/decisive_train_l16_s42

# Recovery example, always a NEW directory, completed arms only.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PY" -B minicpm_rank_gain_decisive.py train \
  --module model.layers.16.self_attn.q_proj --seed 42 \
  --resume-from finetune/adaptive_lora/outputs/decisive_train_l16_s42 \
  --output-dir finetune/adaptive_lora/outputs/decisive_train_l16_s42_retry1
```

For the preregistered grid change only layer 16/18 and seed 42–46, with distinct
folders. No loop is automatically executed. After all ten coordinates complete,
pass exactly one final run directory per coordinate (replace a failed directory
with its successful retry, never include both):

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 "$PY" -B minicpm_rank_gain_statistics.py \
  --runs finetune/adaptive_lora/outputs/decisive_train_l{16,18}_s{42,43,44,45,46} \
  --output-dir "finetune/adaptive_lora/outputs/decisive_stats_$(date +%Y%m%d_%H%M%S)_$$"
```

Runner CLI: positional `dry-run|probe-audit|train`; `--module`, `--seed`,
`--split-seed`, `--model`, `--adapter`, `--train-data`, `--dev-data`,
`--source-data`, `--output-dir`, `--resume-from`, `--min-free-gib`.
Steps, sample counts, LR, weight decay, endpoint and decision thresholds are
fixed in `PROTOCOL`, with no hyperparameter-search CLI.

## Files and traceability

Each run saves `plan.json` (full protocol, separated streams, samples, token
counts, source/code/model/adapter/tokenizer SHA256 and Git/version information),
selected JSON data copies, `report.json` with real failure traceback if any,
`probe_audit.json`, and `directions.pt`. Each arm saves `result.json`,
`train.jsonl`, `target_factors.pt`, and final `artifacts.json` SHA256 manifest.
Factor files are explicitly not full adapters or PEFT checkpoints.
Statistics save `results.json`; no historical result is rewritten.
