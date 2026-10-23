# GID-RSC v2 prototype

This opt-in path does not change GID/rank allocation, v1 geometry, or the default
training mode. It is not evidence of improved generalization. The audited v1
run had a 480256 parameter *budget* and 480000 actual adapter parameters.

## Decision and transaction

1. Proposal is the original fixed training calibration, represented as batches
   of exactly one sample. Selection is a disjoint set of indices of the **same
   active training dataset**, defaulting to the first eight unused indices.
   Both index and encoded-content overlaps are rejected. `rsc_training_samples.json`
   records source SHA256, sample tensor SHA256, indices and loss semantics.
   The source must be an explicit `train.json`/`train.jsonl`; there is no Selection
   path option. Hashes are verified at every event. Synthetic tests are explicitly
   marked synthetic. This is a provenance guard, not proof against a deliberately
   renamed or mislabeled dataset supplied by an operator.
2. At the PRE model, v1 full-weight gradient probe and the same v1 planner generate
   at most one factor candidate per layer. Gap gates, Grassmann targets, eta grid,
   numerical rank and relative LoRA-update trust region are unchanged.
3. Autograd measures the selected factor's gradient on **each** sample, with
   dropout disabled. For group S and candidate factor F*,
   `d_S = mean_i <d loss_i / d F, F* - F>`.
   `loss_i` is the existing model loss (token mean within an individual sample).
   Samples then have equal weights, not weights proportional to token count.
   Scaling `s` is already included in the model derivative. Proposal and
   Selection derivatives for every layer use exactly the same PRE model.
4. Require `d_P < -proposal_margin` AND `d_S < -selection_margin`. Defaults are
   both zero: strictly negative slopes. No alternative side is substituted after
   this filter; v1's chosen A/B candidate is either kept or rejected.
5. Try a fixed short list, default `[1, .5, .25]`, in that order. For all retained
   candidates simultaneously, `F(t) = F + t (F* - F)`. This is a **factor-space
   linear chord, not a Grassmann geodesic**. Cast to the actual factor dtype,
   check both factor ranks, finite values and the same relative `sBA` trust region,
   then run real Proposal and Selection forward losses. Accept the first t with
   each mean loss **strictly below** its PRE loss, and
   `PRE loss - trial loss >= loss_min_improvement` (default 0).
   The positive margin remains configurable. Each group logs `strict_improvement`,
   `numerically_equal`, or `loss_increase`; a strict decrease smaller than a
   positive margin is rejected as `below_min_improvement`. Equality is the exact
   equality of the observed sample-mean losses, with no tolerance that can admit
   an increase. BF16-rounded equal losses are rejected even with nonzero factor
   changes and strictly negative directional derivatives.
   No qualifying t means no-op. No optimizer state is touched during trials.
6. All trial copies are restored in `finally`. Only after joint acceptance,
   copy the accepted values in place and pop Adam state for factors that actually
   changed. Keep all other states, groups, LR, scheduler and Parameter objects.
   An exception aborts the event, restores factors and any removed state objects,
   writes `error.json`, and propagates; it does not silently continue training.
   Parameter values are restored after rejected trials; PyTorch internal tensor
   version counters may advance from temporary copies. Events therefore run only
   at the established post-step/zero-grad boundary, with no live backward graph.

## CLI and outputs

Add to the existing Dense training command:

```bash
--direction_correction geometric_v2 --adaptive_lora false \
--direction_correction_steps 1600 --batch_size 1 \
--rsc_selection_samples 8 \
--rsc_proposal_descent_margin 0 --rsc_selection_descent_margin 0 \
--rsc_loss_min_improvement 0 --rsc_trial_strengths 1 .5 .25
```

`--rsc_selection_indices 8 9 10 11 12 13 14 15` explicitly selects training
indices instead of `--rsc_selection_samples`. Proposal still comes from the
original calibration manifest. Correction and diagnostic steps cannot overlap.
Use a new output directory. Do not launch long training before review.

Each `corrections_v2/step_t` has `modules.jsonl` and `summary.json`, or
`error.json` on an aborted event. Original v1 plans are nested as `v1_plan`, so
v1's proposed `applied_*` fields are not confused with the actual v2 commit.
Outer rows record both sets' sample derivatives, rejection reasons, actual side
and strength. Summary records every trial's per-sample losses/rank/trust checks,
Adam reset counts, training sample provenance, code hashes, Git version, dtype,
model/config and invariants. Adapter snapshots go in `correction_checkpoints_v2`.

## Cost and limitations

For L target layers and Kp Proposal samples, geometric planning retains v1's
L*Kp full-weight probe forwards/backwards and SVDs. With candidates, the additional
factor-derivative passes cost Kp+Ks forwards/backwards (all selected factors in
one pass). Joint checking costs Kp+Ks PRE forwards plus at most
Ntrial*(Kp+Ks) trial forwards, no backward. With Kp=Ks=8 and three trials, that
is up to 16 extra factor-gradient passes and 64 loss-only forwards. Live full
weight gradients/SVDs remain per-layer; pending factors/gradients are small.
The implementation avoids residual SVDs used only for v1's extra diagnostics;
it does not change the common GID/tangent definitions or rank allocation.

Selection is training data used by the decision, **not held-out validation**.
Repeatedly using fixed subsets may overfit the acceptance rule. Near-zero slopes
and BF16 quantization need further GPU/FP32 comparison. The strict rule refuses
equal quantized losses; it cannot establish that a tiny observed decrease is
statistically reliable. Joint acceptance is local to these two subsets;
it does not prove future-gradient alignment, lower Dev loss or better accuracy.
All thresholds/strengths must be fixed before evaluating final Dev/test results.

## Read-only offline step1600 replay

`finetune/offline_replay_rsc_v2.py` loads the historical `run_config.json`, the
local base through `load_training_model`, and the PRE adapter through `load_adapter`.
It verifies the training-data and original calibration SHA256 against run_config;
the adapter digest must be supplied independently from a trusted historical record.
It refuses a different Proposal index set: exactly 0..7, eight singleton batches.
Selection is exactly training indices 8..15. No independent dev/test source can
be supplied. Encoded sample hashes, disjointness and training provenance are
checked through the existing RSC manifest implementation.

Example (replace the placeholders; this documentation does not launch a run):

```bash
PYTHONPATH=finetune /home/hongchang/miniconda3/envs/ralora/bin/python -B \
  finetune/offline_replay_rsc_v2.py \
  --run_config /absolute/historical/run/run_config.json \
  --pre_adapter /absolute/historical/run/correction_checkpoints/step_1600_pre.pt \
  --expected_adapter_sha256 TRUSTED_HISTORICAL_PRE_SHA256 \
  --output_dir finetune/adaptive_lora/outputs/rsc_v2_offline_review_new \
  --device cuda
```

Base dtype and geometry configuration default to the saved run configuration;
`--dtype float32` is an explicit recorded precision override. The v2 thresholds
default to zero and trial strengths to `1 .5 .25`, matching the training CLI.
The offline entry shares `compute_rsc_candidates` with the real training event:
the original v1 planner, factor autograd derivatives, and joint loss search are
not copied or replaced. It creates **no optimizer or scheduler**, never commits
the returned values, and performs no Adam reset or training steps.

Every result is labeled `offline_diagnostic_only`; `would_accept`/`would_apply`
describe hypothetical acceptance, while `committed=false` and `applied_side=none`
remain explicit. `modules.jsonl` contains original plans, sides, both derivative
vectors and gating masks. `summary.json` contains every joint trial's per-sample
losses, acceptance status, rank/trust checks, stage wall times synchronized with
CUDA, and peak allocated/reserved memory (loaded model plus computation).
`candidate_factors.pt` contains proposed factor targets, **not a trained adapter**.
`rsc_training_samples.json` and source/config hashes make the diagnostic traceable.

The output directory must not exist. Historical files are read-only, and their
hashes are checked again at exit. PRE factor values/identities are checked on
success and exceptions, with restoration followed by a loud failure if a
computation left a factor changed. Temporary `copy_` operations can increment
Tensor version counters even after exact value restoration; replay requires no
live backward graph. It does not claim to restore those counters or historical
optimizer/RNG/data-order state. Selection remains **training decision data**, not
an independent validation set. This is a research prototype, not a claim of
generalization improvement.

The PRE adapter has no complete Adam/scheduler/data-order/RNG state. Loading it
alone cannot exactly resume the remaining 400 steps. Strict continuation needs
an actual full training-state checkpoint or an independently verified replay
from the start, and must not be claimed by this prototype.
