# Fixed-rank temporal capacity-demand diagnostics

Base: cf04530a52ae5beaa9c2e7ce4a0dedc09cf8a153. Independent entry point;
no changes to Global84, official training, source checkpoints or datasets.
Only normal training changes A/B. Diagnostics never correct factors or expand rank.

Run each seed as one uninterrupted 400-step trajectory, fresh AdamW, no historical
momentum, no reset, constant LR 5e-5, decay .01, dropout disabled. All 84 old
A/B factors (480000 parameters) train. Batch size one; loss is sample mean
masked NLL times effective tokens / mean training tokens, exactly Global84.
Snapshots copy factors to CPU after steps 0/100/200/300/400. They are not full
optimizer-resume checkpoints. The same Global84 split and tokenizer are used at
all stages: train400, construction/alignment/selection/audit32 each, Dev400.
Dev is used for source isolation only, never for fitting/allocation.

Each stage runs 128 forwards/backwards, each collecting all 84 base gradients
simultaneously using collect_weight_gradients. A/B freeze only during probe;
flags, gradient references, modes and RNG restore. One stage's ~11.81 GiB FP32
cache is temporary and removed after analysis (also on Python exceptions).
Keep at least 14 GiB free disk; external process termination may leave a temporary
folder, which must not be mistaken for completed output. GPU admission checks
30 GiB free but does not guarantee avoiding OOM. No automatic sample omission.

C and DE use unchanged Global84 fitting, prefix scores, Krylov parameters,
tolerances and exact stratified allocation. Each hypothetical allocation has
4 q ranks and 4 v ranks, cost25600. No allocation is applied to the model.
Audit is read only after fitting/scoring, for independent proxy diagnostics.

Construction is the reference for CT, ER, gradient/residual norms and M(k).
For zero gradients CT=ER=0 and zero_gradient=true; CT+ER=1 is asserted only for
nonzero energy. Top singular values are converged Ritz approximations, not an
exact full SVD. Original solver residuals are retained in the analysis record.
Numerically zero singular directions are omitted from overlap and return null
when fewer than k directions are available. Repeated singular values can make
individual top-k boundaries non-identifiable; interpret these overlaps cautiously.
Block consistency is all six pairwise residual-gradient cosines for each role,
with zero-denominator cosines null. No large identity or projector is built.

Outputs: report.json, train.jsonl, five adapter_step_N.pt, per-block probe
profiles, temporal_metrics.csv (including nested raw scores/independent audit),
allocation_by_stage.json, rank_turnover.csv, subspace_stability.csv, results.json.
Rank turnover = sum absolute per-module rank differences / (2 * total added
rank). Spearman uses tied average ranks on first-prefix marginal scores and is
null for constant vectors. Entered/exited/persistent modules are recorded per
transition. summarize compares paired seeds at each time, never treats 15
checkpoints as independent experiments. Reports retain input/code hashes.

Scientific conclusions require joint inspection of normalized residual energy,
absolute norms, M(k), consistency, allocation changes and independent audit:
- reproducible demand changes across time;
- only gradient strength changes with unchanged allocation;
- instability dominated by sample noise;
- insufficient evidence.
The automatic default is insufficient evidence; no threshold is invented to
force a positive conclusion. These are capacity proxies, not actual validation
NLL gains or proof that SVD directions optimize performance.

## Commands (GPU commands are manual only)
```bash
cd /home/hongchang/whf/MiniCPM
PY=/home/hongchang/miniconda3/envs/minicpm5/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
CUDA_VISIBLE_DEVICES='' "$PY" -B -m unittest test_global84_temporal_rank_demand test_global84_revision test_global84_rank_direction test_global84_experiment test_minicpm_rank_gain_decisive test_minicpm_single_layer_rank_gain -v
RUN="finetune/adaptive_lora/outputs/temporal_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES='' "$PY" -B global84_temporal_rank_demand.py dry-run --output-dir "$RUN/dry"
# Each invocation trains and diagnoses one full trajectory. Serial, not parallel.
for seed in 42 43 44; do
 CUDA_VISIBLE_DEVICES=0 "$PY" -B global84_temporal_rank_demand.py train --seed "$seed" --output-dir "$RUN/seed$seed" || break
done
CUDA_VISIBLE_DEVICES='' "$PY" -B global84_temporal_rank_demand.py summarize --runs "$RUN/seed42" "$RUN/seed43" "$RUN/seed44" --output-dir "$RUN/summary"
```
All output directories must be new. Do not execute GPU commands before review.
