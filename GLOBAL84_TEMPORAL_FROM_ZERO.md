# From-base fixed-rank temporal audit (research prototype)

Independent entry: global84_temporal_from_zero.py. Existing Global84/temporal
and formal training files remain unchanged. No step1600 adapter is loaded.
Scope: pretrained base, GID initialization, then fresh AdamW continuously for
2000 updates. Not a bitwise reproduction of historical initialization.

## Historical audit
Read metamath_pilot_control_2000_seed42/run_config.json and the original training
loop. Historical settings: rank2 provisional, alpha4, Kaiming A, zero B, initial
probe8, Dense budget480256, minrank1/maxrank4, BF16 base/FP32 factors, dropout0.1,
gradient checkpointing, AdamW LR5e-5/decay0.01, linear 1-step/2000 scheduler.
Historical loss is per-example masked token mean, **not** Global84's extra
sample-token weighting. This entry preserves historical loss and scheduler.
Initialization uses eight first train samples, equal-example mean full gradients,
all84 simultaneously per sample. SVD/GID reuse svd_with_gid. The unchanged pure
allocate_dense_ranks function is loaded from its AST to avoid eager DeepSpeed
registry imports; its source hash is recorded. SVD runs on CPU FP32; numerical
agreement is not claimed bitwise identical to historical CUDA SVD. Actual
allocated budget is measured, never assumed 480000.

Generate a single initial_adapter.pt using initialize, then all three seeds
load it with the same data/config/code manifest. Training dropout/order seed
changes only after loading shared factors. Initial GID computation is not run
by the single-sample smoke. The smoke checks original weights and simultaneous
84-gradient collection on provisional zero-B factors, not full initialization.

## Data
Use all historical train2000 and Dev400 for provenance exclusion. Scan original
MetaMath in source order, choosing first64 distinct GSM and64 distinct MATH
original-question groups outside BOTH pools. Deterministic seed20261031 shuffles
128 groups before splitting into four groups of32; per-role family counts are
recoverable from metadata. No length-based replacement: invalid/overlong samples
fail. Tokenization/labels reuse ChatTrainingDataset. Data and model files/code
hashes and token counts are recorded. All probe samples are fixed across seeds
and stages. Initial GID's train8 is separate from temporal probe128.

## Observations
Stages 0/100/400/800/1200/1600/2000 reuse temporal stage analysis, unchanged
Krylov scores/tolerances and exact DP. Every stage uses one temporary ~11.81GiB
gradient cache and needs14GiB free disk. All overlap/energy/normalized metrics
are retained. Numeric A/B ranks are recorded at every stage; B=0 at stage0
means degenerate tangent coverage, not evidence of learned capacity deficiency.

Compare stage-specific Selection/Audit scores with frozen step0 allocation.
Runner-up gap: exclude each chosen prefix in turn, rerun the SAME DP, and take
the best distinct feasible allocation. This enumerates every possible distinct
allocation's first changed decision; no new allocation algorithm is substituted.
Audit full-gradient U is independent diagnostic only (including for C); it is
not the residual Selection objective and is labelled accordingly.

Training saves one-step decay-only counterfactual difference, computed in actual
factor storage precision, plus full A/B gradient activity/change statistics.
This is a numerical counterfactual, not a proof of unique causal attribution.
Probes check RNG and Adam tensor versions; no optimizer/scheduler replacement.
Snapshots are A/B-only and cannot resume full optimizer state. Base weights are
checked against safetensors. All output directories must be new.

## Commands
```bash
cd /home/hongchang/whf/MiniCPM
PY=/home/hongchang/miniconda3/envs/minicpm5/bin/python
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
RUN="finetune/adaptive_lora/outputs/temporal_zero_$(date +%Y%m%d_%H%M%S)_$$"
CUDA_VISIBLE_DEVICES='' "$PY" -B global84_temporal_from_zero.py dry-run --output-dir "$RUN/dry"
CUDA_VISIBLE_DEVICES=0 "$PY" -B global84_temporal_from_zero.py smoke --output-dir "$RUN/smoke"
# Only after review: initial GID is a full eight-sample GPU probe plus CPU SVD.
CUDA_VISIBLE_DEVICES=0 "$PY" -B global84_temporal_from_zero.py initialize --output-dir "$RUN/initial"
for seed in 42 43 44; do
 CUDA_VISIBLE_DEVICES=0 "$PY" -B global84_temporal_from_zero.py train --seed "$seed" --initial-adapter "$RUN/initial/initial_adapter.pt" --output-dir "$RUN/seed$seed" || break
done
CUDA_VISIBLE_DEVICES='' "$PY" -B global84_temporal_from_zero.py summarize --runs "$RUN/seed42" "$RUN/seed43" "$RUN/seed44" --output-dir "$RUN/summary"
```

Three seeds are paired trajectories, not21 independent observations. Spectral
capacity proxies and Audit U cannot establish validation-loss benefit or an
optimal performance direction. Results retain an insufficient-evidence default.
Full initialization/2000-step trajectories and seven-stage summary on actual
runs are NOT validated by CPU tests or the single-sample smoke.

## Validation record
102 CPU tests passed, including dropout-enabled follow-on trajectory equality
with/without an inserted probe (losses, parameters and Adam state). Real-data
dry-run PASS at /tmp/temporal_zero_dry_20261031/report.json. Full-model single
sample GPU smoke PASS at /tmp/temporal_zero_gpu_smoke_20261031/report.json:
381 base tensors verified, one forward/backward, all84 gradient matrices,
peak allocated6457571840 bytes, reserved6585057280 bytes. No full initial GID
run or 2000-step training was executed. Initial actual budget/hash are pending
the explicit initialize command, not inferred from the old step1600 adapter.

## Equal-example / checkpointing consistency correction

Supersedes the earlier token-weighted temporal probe and one-forward smoke.
Temporal construction/alignment/selection/audit gradients now all average
per-example masked mean loss gradients. collect_equal_gradients reuses the
original collector on singleton batches, with one forward/backward per example
collecting all84 weights, then accumulates with 1/K. All four block means and
allocation/diagnostic scores use equal block weights [8,8,8,8]. Actual effective
token counts remain in plan.json and per-sample probe profiles for audit only.
Original Global84 functions are NOT modified or monkeypatched: private function
global bindings inject equal counts/collector into the reused stage and analysis
functions. Initialization's original eight equal-example means are unchanged.

Historical fields are explicitly validated, including model family, Dense
structure, provisional rank2, alpha4, bounds1..4, Kaiming, dynamic length2048,
checkpointing, no adaptation or correction. Any mismatch fails. Initialization
checks all84 names, factor shapes, zero B, rank bounds and actual parameter sum
against allocator output/budget. Serialized factors and initial SHA are checked;
training checks the shared initial file's SHA and revalidates loaded factors.

Checkpointing now explicitly uses use_reentrant=False as in the original
training entry. No enable_input_require_grads hook is installed: non-reentrant
checkpointing supports parameter gradients without requiring frozen embedding
outputs to require grad. Real smoke verifies this rather than assuming it.

smoke now runs two fresh-AdamW trajectories of TWO consecutive optimizer steps
each, dropout0.1, provisional rank2 on all84 actual model modules. One trajectory
inserts a two-example equal-weight probe after step1; the other does not. It
requires bitwise equal subsequent losses, factors and Adam step/exp_avg/exp_avg_sq,
finite gradients/moments, nonzero A/B gradient activity across the two steps, and
unchanged disk-verified base weights. A's first-step gradient may be zero because
B starts zero. No full GID initialization or 2000-step training is run by smoke.
This does not validate a real seven-stage run or post-GID allocation by itself.

Consistency-fix validation: 106 CPU tests passed (including unequal supervised
lengths vs independent per-example gradient mean, private block-count binding,
historical config rejection and initialization invariants). Dry-run PASS:
/tmp/zero_equal_dry_20261031/report.json. Real training-mode smoke PASS:
/tmp/zero_equal_training_smoke_20261031/report.json. Both two-step loss traces
were [0.5087690353393555, 1.131565809249878], with bitwise matching factors and
Adam states after the inserted probe. All84 A/B factors had nonzero gradient
activity over the two steps; 381 base tensors verified unchanged. Peak allocated
6445760000 bytes. No full GID initialization or 2000-step training executed.
