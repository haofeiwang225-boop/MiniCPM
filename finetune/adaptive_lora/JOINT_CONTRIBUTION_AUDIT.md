# Step1600 historical contribution / joint-update audit

This standalone audit extends reviewed commit `3ece3e14811a097432c51af1a33c3af8a0580358`.
It reuses PRE/POST validation, historical factor extraction, factor rank/trust
checks, state preservation, temporary copies and the existing token-weighted NLL
evaluator. There is no training, optimizer, gradient probe, GID computation or
Grassmann candidate generation. Small-factor rank checks still use existing `orth()`.

## Frozen policies

- PRE: no change; ALL: all 83 historical changes.
- FA: A's 24 selected modules; FB: B's 30 selected modules.
- COMMON: 8; FA_ONLY: 16; FB_ONLY: 22; UNION: 46.
- RANDOM30: random seed 42, sampled from sorted Proposal-negative pool (54).
  A local `random.Random` is used. If the generic helper encounters a pool under
  30, it includes the entire pool and samples the balance from remaining historical
  candidates, explicitly logging fallback. This experiment's required pool is 54.
- Optional `--include_singletons`: one-at-a-time application of each of the 46
  UNION modules. This adds substantial forward cost; it is disabled by default.

Masks are reconstructed from the saved selected flags and verified against saved
Proposal/Selector slopes. A/B must share PRE/POST hashes, configuration and Proposal
slopes. The expected experiment cardinalities are checked, never synthesized.
A/B NLL results are not used to construct or accept any policy. No rollback based
on loss, step-size search, resampling seeds or direction regeneration occurs.

## CPU preflight (no model load or forward)

Run from the repo root in the `ralora` environment:

```bash
cd /home/hongchang/whf/MiniCPM
OUT=finetune/adaptive_lora/outputs
RUN="$OUT/metamath_pilot_geo1600_rebuilt_20261013_221250_seed42"
CUDA_VISIBLE_DEVICES='' \
TRITON_CACHE_DIR=/tmp/minicpm_adaptive_triton PYTHONPATH=finetune \
/home/hongchang/miniconda3/envs/ralora/bin/python -B \
  -m adaptive_lora.audit_joint_contributions preflight \
  --results_a "$OUT/rsc_v3_audit_A_20261020/results.json" \
  --results_b "$OUT/rsc_v3_audit_B_20261020/results.json" \
  --pre_adapter "$RUN/correction_checkpoints/step_1600_pre.pt" \
  --post_adapter "$RUN/correction_checkpoints/step_1600_post.pt" \
  --historical_audit "$OUT/geo1600_cpu_audit_v1/cpu_audit.json" \
  --train_source /home/hongchang/whf/datasets/MetaMathQA/MetaMathQA-395K.jsonl \
  --evaluation_data finetune/adaptive_lora/data/metamath_pilot_v1/dev.json \
  --evaluation_metadata finetune/adaptive_lora/data/metamath_pilot_v1/dev.source_metadata.json \
  --calibration_indices $(seq 88 119) \
  --evaluation_indices $(seq 0 399) \
  --random_seed 42 \
  --output_dir "$OUT/rsc_v3_joint_preflight_review"
```

`plan.json` freezes all module lists, random seed/pool, indices, data/source and
PRE/POST hashes, A/B input hashes, historical side counts, original-question
metadata and precision. Training metadata is reconstructed by exact
query/response matching with existing `build_source_metadata()`, included in the
new plan; no existing dataset/metadata is edited. Missing or ambiguous provenance
fails closed. Existing output directories are refused.

For a separate evaluation dataset, selected evaluation original-question groups
must not overlap **any** training group. If evaluation uses the training file
itself, selected groups must avoid both new calibration groups and all A/B
Proposal, Selector and historical audit groups (not merely different row IDs).
Such a set is explicitly labelled reserved training samples, not dev/generalization.
Samples of the same original question within a group are allowed and identified;
NLL is token-weighted, not original-question-weighted.

The command above uses existing Dev, which has been evaluated previously. It is
held out from training but is not a new blinded test. Do not use its scores to
choose a new mask, seed or threshold and then call the same set confirmatory.

A completed local CPU precheck is available at
`outputs/rsc_v3_joint_preflight_20261020/plan.json` (not committed). Do not overwrite
it; the example command deliberately uses a new directory.

## Explicit server forward-only run (not executed during development)

```bash
cd /home/hongchang/whf/MiniCPM
OUT=finetune/adaptive_lora/outputs
# "noclobber" also protects the log. The program independently refuses old output.
set -o noclobber
CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_DIR=/tmp/minicpm_adaptive_triton \
PYTHONPATH=finetune nohup /home/hongchang/miniconda3/envs/ralora/bin/python -B \
  -m adaptive_lora.audit_joint_contributions run \
  --plan "$OUT/rsc_v3_joint_preflight_20261020/plan.json" \
  --device cuda \
  --output_dir "$OUT/rsc_v3_joint_evaluation_20261020" \
  > "$OUT/rsc_v3_joint_evaluation_20261020.log" 2>&1 &
```

The explicit run verifies every frozen input hash again, reloads exact PRE,
revalidates historical candidates and masks, then evaluates all policies on the
same two sample sets. It preserves saved factor dtype and uses A/B's saved base
precision. Nine policies on 32+400 samples require 3888 single-sample forwards;
no such real-model evaluation was run during development. Optional singleton
policies add 46*(32+400) forwards. Model memory remains one model, not nine.

The output `results.json` contains policy member lists; actual changed-module
counts; per-layer and concatenated scaled LoRA update Frobenius norms; per-group
token-weighted NLL and deltas; per-sample index, token count, NLL sum and mean NLL;
interaction contrasts; input provenance and PRE restoration status. No parameter
objects, rank, scaling or optimizer/scheduler states are changed. Temporary copy_
restores values exactly but advances Tensor version counters.

## Interpretation fixed before evaluation

Let L(S) be token-weighted NLL under subset S, on exactly the same samples.

- Direction selection: compare FB vs RANDOM30 (equal cardinality), FA/FB vs ALL,
  and common vs exclusive blocks. Report update norms too: cardinality matching
  does not match rank budget, A/B mix or update energy. A single random seed is
  a fixed reference, not a statistically representative random baseline.
- Conditional contribution: L(FA)-L(COMMON), L(FB)-L(COMMON).
- Block interaction for FA:
  L(FA)-L(COMMON)-L(FA_ONLY)+L(PRE); similarly FB.
- UNION interaction:
  L(UNION)-L(COMMON)-L(FA_ONLY)-L(FB_ONLY)+2*L(PRE).
  Negative means joint loss is lower than the additive block prediction;
  positive means higher. These are finite-update interactions, not gradient claims.
- With singleton policies, additionally report
  [L(S)-L(PRE)] - sum_{l in S}[L({l})-L(PRE)].
  This distinguishes joint effects from summed one-layer effects, without
  claiming order-independent causal attribution or Shapley values.

The concatenated norm is sqrt(sum_l ||s_l delta(B_l A_l)||_F^2), computed with
existing float64 low-rank Gram formulas. It is not a global network-function
trust bound. Never rank or prune masks using the evaluation set.

## Tests

```bash
CUDA_VISIBLE_DEVICES='' TRITON_CACHE_DIR=/tmp/minicpm_adaptive_triton \
PYTHONPATH=finetune:finetune/adaptive_lora/tests \
/home/hongchang/miniconda3/envs/ralora/bin/python -B -m unittest \
  test_joint_contributions test_cross_batch_acceptance \
  test_geometric_direction_correction test_geometric_audit -v
```
