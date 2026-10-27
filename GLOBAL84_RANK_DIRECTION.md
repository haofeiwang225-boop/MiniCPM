# Global84 rank and direction study: q/v-stratified first validation

Base commit: `541b139b8ed33efc3dac63d3eb5b1eea19ba6dce`.
The preceding single-layer residual-SVD study did **not** meet its success
criteria (reported C−B mean −7.47187e−5, paired interval across zero). It is not
assumed effective. This code is a new research prototype, not evidence of gain.

No original training entry point, model, adapter, benchmark, old script or
result is modified. Implementation runs CPU tests and a CPU dry-run only.

## Audited model, checkpoint and exact budget

The immutable PRE is
`finetune/adaptive_lora/outputs/metamath_pilot_geo1600_rebuilt_20261013_221250_seed42/correction_checkpoints/step_1600_pre.pt`,
SHA256 `d79b1b86a36a9809f465ef8c2d0a3251cf18f176832e0e3a48bb719e065d6769`.
Base is `/home/hongchang/whf/models/MiniCPM5-2B`.

CPU inventory reads **all 84 saved A/B tensors**, configs, ranks, scalings and
base safetensors shape metadata. The saved original adapter has **480000**
parameters; do not substitute a nominal budget of 480256.

| Type | Layers | Actual matrix shape (out × in) | Cost per new rank | New ranks | New parameters |
|---|---:|---:|---:|---:|---:|
| q_proj | 42 | 2048 × 2048 | 4096 | 4 | 16384 |
| v_proj | 42 | 256 × 2048 | 2304 | 4 | 9216 |

A trains 480000 parameters. B/C/D/E train **505600** parameters each, including
exactly **25600** new ones. Both types compete separately. This is **not** free
q/v global optimization. Every layer can receive 0,1,2,3,4 new ranks; four is
sufficient because the entire type budget is four. Same-layer directions are
mutually orthogonal and orthogonal to old A at initialization. They are ordinary
trainable factors afterwards; no manifold constraint is added to training.

## Data protocol

Fixed split seed `20261027`, exact source matching via existing MetaMath
provenance code, grouping by `original_question_sha256`:

- 400 training examples, one epoch / 400 updates.
- 32 construction examples.
- 32 selection examples.
- 32 independent audit examples.
- All 400 existing Dev examples in fixed original order, GSM200/MATH200.

Full pilot train/dev pools are checked for sample and question-group overlap.
The four training-side subsets have different original-question groups. Dev
metadata is used only to verify isolation/identity, never to score candidates.
Dev labels are not passed to probe or allocate. Dev has prior use and is **not
an independent blind test**. PRE also saw historical training data; within-stage
isolation is not a claim that these training examples are new to PRE.

Existing chat/assistant-only label masking is reused. Batch size one, dynamic
length, maximum 2048. Overlong/empty-label answers fail rather than truncating,
filtering, or silently replacing examples. Source, model, adapter, tokenizer,
code and software hashes/versions are frozen in `plan.json`.

All three seeds (42,43,44) use this same split. A/B/C/D/E at one seed share the
same permutation of the 400 training examples. Random direction, Krylov start,
random allocation and training-order streams are domain-separated hashes.

## One backward collects every layer

`global84_rank_direction.collect_weight_gradients()` freezes all parameters,
then temporarily enables only the 84 original target base weights for autograd.
A/B remain frozen. For **each batch**, it executes one model forward and one
backward, registering post-accumulate hooks for every target. Each hook copies
its native gradient to CPU FP32, accumulates it, and clears the GPU leaf `.grad`.
It checks every target received a gradient on every batch (including valid zero
gradients). Missing/nonfinite gradients are errors, never omissions.

If `n_i` is the number of effective shifted labels and `ell_i` the model mean
masked NLL, gradients estimate

`G = sum_i n_i grad(ell_i) / sum_i n_i`.

Each group is partitioned into four fixed blocks of eight examples. One pass
per example, **96 total forwards and 96 backwards**, not 84 times that number.
The 12 block gradient caches support weighting, uncertainty and audit without
another model pass. Block means combine by effective token counts. Accumulation
is FP32 on CPU; native BF16 backward precision is not represented as FP32
backward. About 0.74 GiB of FP32 gradients are held on CPU per block; all 12
blocks occupy about 8.86 GiB on disk. Disk space is checked before GPU loading.
One layer's blocks are read at a time for direction analysis.

Flags, existing `.grad` objects, module modes, Python/NumPy/CPU/CUDA RNG and
frozen target weights are restored/checked, even after exceptions. No optimizer
exists in probing. Losses, token counts, per-target hook counts, wall time,
CPU accumulator bytes and CUDA allocated/reserved peaks are recorded. OOM is a
real FAIL; no automatic sample reduction, loss change or omitted layer fallback.

## Four direction families, no dense Gram

Let `P=I-Q_A Q_A^T` and `R=(I-Q_B Q_B^T) G_c P` using existing rank-aware bases
and identity-free residual code. All frames have four unit orthogonal columns.

1. **Random:** local-seed Gaussian vectors projected into P, followed by QR.
2. **Residual SVD:** leading right singular directions of R, solved through
   the matrix-free symmetric operator `v -> P R^T(R Pv)`.
3. **Projected full SVD:** leading right singular directions of `G_c P`, using
   `v -> P G_c^T(G_c Pv)`; diagnostic candidate family, no separate training arm.
4. **Cross-alignment:** largest **algebraic** eigenvectors of
   `v -> 0.5 P [G_c^T(G_s Pv) + G_s^T(G_c Pv)]`.
   Their Rayleigh quotient is exactly `U(v)=dot(G_c v,G_s v)`.

The solver uses a full-reorthogonalized Krylov/Rayleigh-Ritz space capped at
128 vectors and a <=128×128 projected eigensystem. It never forms a 2048×2048
Gram or identity/projector. Rectangular original gradients/residuals themselves
are retained as required. Repeated/zero eigenvalues permit deterministic
complement restarts. Negative eigenvalues are not replaced by magnitude-ranked
ones. The relative Ritz residual is normalized by `||AQ||_F`, with fixed
`1e-4` tolerance; nonconvergence raises an error rather than silently changing
algorithms/parameters. This is an iterative approximation, not a proof that a
finite Krylov run found the exact global eigenoptimum. Null spectra may yield
arbitrary zero-proxy directions, which are logged as such, not positive gain.

Initialization uses A_add=V^T, B_add=0, FP32 added factors and unit extra scaling,
with unchanged old rank/scaling. Existing `ExpandedLoRA` and validated real
loader/84-adapter restore are imported unchanged. Before each arm's training,
all 400 Dev full-logit byte hashes must equal the unexpanded PRE logits.

## Frozen allocation scores and five arms

The following scoring choices are declared before new GPU results:

- For each residual frame prefix k, C uses
  `sum_{j<=k} dot(R_construct v_j, R_selection v_j)`.
- For each optimized alignment frame prefix k, D/E use
  `sum_{j<=k} dot(G_construct v_j, G_selection v_j)`.
- Four token-weighted selection-block gains give a descriptive block uncertainty.
  With normalized block weights w and block prefix gains u:
  `SE_proxy = sqrt(sum(w^2 (u-mean)^2)/(1-sum(w^2)))`.
- Prefix value is `mean - 1.0*SE_proxy`. Direction covariance is preserved by
  summing within each block **before** computing prefix uncertainty. Record
  mean, penalty, value/cost and marginal prefix value. This is a heuristic
  penalty, **not an unbiased confidence bound**, especially because alignment
  directions reuse selection data for fitting and scoring.
- Exact dynamic programming chooses integer prefixes 0..4 over all 42 layers
  within each type, summing these values subject to exactly four ranks. Costs
  are equal within a type, so maximizing value or value/type-cost is equivalent.
  Scores can be negative. Exact budget is never relaxed: forced negative
  marginal values are counted. Ties use a fixed lexicographic rule.

| Arm | Allocation | Added directions |
|---|---|---|
| A | none | none |
| B | four independent uniform module draws **with replacement**, per type | random |
| C | residual-prefix DP | residual SVD |
| D | cross-alignment-prefix DP | random |
| E | **identical manifest to D** | optimized cross-alignment |

D/E allocation scores are computed using the optimized candidate potential;
D deliberately substitutes its fixed random frame to isolate direction choice.
B/D share the same per-module random frames at a given seed. E−D therefore
holds allocation fixed; D−B changes allocation under random directions; E−C
compares two combined policies. This design does not claim independent
unbiased estimation of selection gains or global optimality of actual loss.
Cross-layer interactions are not modeled by the additive allocation proxy.

Audit gradients are loaded only after frames and selection prefix scores exist.
They are used solely to log full audit energy, construction/audit U, selection
and audit block signs/std, and fixed-direction construction-half **response**
cosine. The latter is not an independently estimated subspace cosine. These
scores never enter DP, re-rank directions, or adjust thresholds. Do not tune
using audit or final Dev results. Audit transfer is a predictive diagnostic,
not a theoretical guarantee of training improvement.

## Full-model training and endpoint

All original 84 A/B pairs are trainable, including unexpanded modules. Every
base weight stays frozen. Each arm independently restores PRE and creates new
AdamW (LR 5e-5, weight decay .01, default betas .9/.999 and eps 1e-8), dropout
off, constant LR, batch size one. Historical optimizer state is unavailable and
never fabricated. Backward loss is `ell_i*n_i/mean_training_tokens`, matching
the token-weighted objective under uniform example sampling.

Each arm runs exactly 400 updates, then fixed Dev400 token-weighted NLL. No
best-step selection. Before/after overall and GSM/MATH NLL, per-question NLL,
source hashes/tokens, all train losses and gradients, actual parameter counts,
factor changes, added BA norms, timings and memory peaks are saved. Old factor
identities/ranks/scalings persist. All frozen base tensors are checked against
real safetensors. Unlike single-layer tests, the other adapters are **not**
claimed unchanged: all 84 intentionally train. In-memory PRE factors are
restored on exit. Source checkpoints and all old results remain untouched.

A custom `adapter.pt` stores the full research state; it is **not a PEFT
adapter**. Files/directories are never overwritten. Failures retain traceback;
RUNNING/missing/FAIL never means a successful experiment. A whole completed arm
can be run separately with `--arms`; there is no fake mid-training resume.

## Paired CPU statistics

Require all three seeds × five arms. Match exact frozen protocol/code/data,
source order, per-question hashes/family/tokens, parameter counts and step 400.
Check artifact SHA256. Primary comparisons: E−D, E−C; secondary: D−B, E−B, E−A.

10,000 paired crossed bootstrap draws, fixed seed 42: resample original question
groups together and independently resample the three seed IDs; the same draws
apply to all arms. Recompute token sums/denominators on every draw. Repeated
questions across seeds are not new independent samples. Report overall,
GSM/MATH, per-seed values, per-question changes and percentile 95% intervals.

Three seeds provide limited precision. No minimum-effect threshold was supplied
for this new study, so the old single-layer 0.0005 threshold is **not imported**.
The script describes whether both primary intervals support improvement,
whether a primary interval supports degradation, or evidence is insufficient.
These descriptive intervals do not assert simultaneous multiple-comparison
guarantees or prove efficacy. Engineering PASS only means checks completed.
Existing Dev reuse and selection fitting limit generalization conclusions.

## Commands — review before manually starting any GPU workload

```bash
cd /home/hongchang/whf/MiniCPM
PY=/home/hongchang/miniconda3/envs/minicpm5/bin/python

# CPU tests, no GPU probe/training.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
 "$PY" -B -m unittest test_global84_rank_direction test_global84_experiment \
 test_minicpm_rank_gain_decisive test_minicpm_single_layer_rank_gain -v

# Real adapter metadata + CPU tokenizer/source/data checks; no base-model forward.
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 "$PY" -B global84_experiment.py dry-run \
 --output-dir "finetune/adaptive_lora/outputs/global84_dry_$(date +%Y%m%d_%H%M%S)_$$"
```

Choose a new root per complete study. All children below must initially be
absent. Probe is shared by the three seeds because PRE and sample sets are fixed.

```bash
RUN="finetune/adaptive_lora/outputs/global84_$(date +%Y%m%d_%H%M%S)_$$"

# 1. PROBE: one user-started GPU run, no optimizer.
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
 "$PY" -B global84_experiment.py probe --output-dir "$RUN/probe"

# 2. ALLOCATE: CPU direction analysis; can be costly, but never trains.
# Optional --analysis-device cuda uses GPU only when explicitly requested.
for seed in 42 43 44; do
 CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  "$PY" -B global84_experiment.py allocate --seed "$seed" \
  --probe-dir "$RUN/probe" --output-dir "$RUN/allocate_s$seed" || break
done

# 3. TRAIN: only after inspecting allocation; serial GPU runs, no concurrency.
# Each invocation runs A/B/C/D/E sequentially, each with a fresh PRE + AdamW.
for seed in 42 43 44; do
 CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  "$PY" -B global84_experiment.py train --seed "$seed" \
  --allocation-dir "$RUN/allocate_s$seed" --output-dir "$RUN/train_s$seed" || break
done

# 4. STATISTICS: CPU, only after all 15 arm/seed results are complete.
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 \
 "$PY" -B global84_statistics.py \
 --runs "$RUN/train_s42" "$RUN/train_s43" "$RUN/train_s44" \
 --output-dir "$RUN/statistics"
```

For manual partial batches, e.g. `--arms A B` and later `--arms C D E`, use new
output directories for each invocation and pass all successful directories to
statistics. Duplicate seed/arm results are rejected. Incomplete/failed runs
are not silently excluded. No command above has been automatically executed
on GPU during development.

CLI stages record full parameters and source hashes. `--min-free-gib` defaults
to 30 for user-started GPU stages; this is an admission check, not an OOM
prediction. No automatic GPU memory workarounds alter samples or objectives.
No new dependencies are installed. The loader, safe zero-B expansion, source
matching, chat masking, token NLL and paired bootstrap reuse existing code.
