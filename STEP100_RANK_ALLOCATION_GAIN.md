# Step100 allocation-only real rank gain

Independent entry `global84_step100_rank_gain.py`; reviewed base 413b67eca5627bfa3197ef49bb073eae7549eda8. Old training, GID and temporal programs are unchanged.

## Frozen design

Three paired seeds 42/43/44, source step100, fresh AdamW for every arm. Six arms: base; random allocation (four draws with replacement within each q/v type); static_C (explicitly the historical step0 **C**, not DE list); current_C (step100 C); static_DE (step0 DE); current_DE (step100 DE). No directions are fitted here. Allocation audit samples and Dev never select random directions.

All grown arms add q4/v4 = 25600 parameters, checked from actual shapes. Original 480000 parameters remain trainable; grown total 505600. Shared CPU-local random unit A rows are keyed by seed/module/added-rank index. New B is zero, new branch scaling is 1; old scaling remains alpha/r. No orthogonal or SVD direction substitution. Multiple added ranks in one module are supported.

Wrapper reproduces the official Dense forward arithmetic and shares its **single** dropout tensor with the new branch. Old A/B Parameter identity/value and scaling are preserved at insertion. Full-model train/eval logits must be bitwise equal before any update. First-step new B gradient/update, second-step new A update, all old/new gradient activity and final changes are required. Gradient evidence, rather than weight decay alone, gates engineering PASS.

300 updates, batch1, masked sample-mean loss, dynamic2048, dropout.1, BF16 base/FP32 factors, nonreentrant checkpointing; AdamW defaults with weight decay.01. At local step t, used LR is `5e-5*(1-(100+t-1)/2000)`. This reproduces the historical linear global-step curve without restarting at full LR. All84 original adapters train; base stays frozen.

## Pairing limits

Historical snapshots contain factors, not AdamW/RNG. This is **not seamless continuation**. Source logs do not contain explicit sample indices. The audited historical DataLoader permutation is reconstructed using seed and runtime, and positions101–400 are saved/hash-recorded. It cannot be independently verified against a historical sample-index manifest. Every new arm does share the exact saved order and dropout seed. Same seed alone is not a claim of bitwise historical dropout continuation. Train-mode insertion equivalence is tested explicitly. Data/source/checkpoint/code hashes are checked; incompatible historical code fails closed.

## Evaluation and statistics

Frozen Dev400 (GSM200/MATH200) has prior research use, not blind. Evaluate local0/100/200/300; primary endpoint300, no best-checkpoint selection. Store each masked-loss sum, shifted supervised token count, sample NLL and source metadata. Report token-weighted and question-mean loss plus GSM/MATH. No automatic retries. Failed runs retain traceback/report and cannot be consumed as successful full training.

CPU statistics require all three seeds and all six arms, matching metadata/token counts. Pair seed IDs and original-question clusters across arms for10000 bootstrap draws. Primary current_C−static_C and current_DE−static_DE use97.5% Bonferroni intervals; random comparisons and GSM/MATH subgroup analyses exploratory95%. Only three independent training trajectories: intervals have substantial limitations, not1200 independent training repeats. Actual validation results, not gradient proxies or engineering PASS, determine observed benefit.

## Commands

```bash
cd /home/hongchang/whf/MiniCPM
PY=/home/hongchang/miniconda3/envs/minicpm5/bin/python
SRC=finetune/adaptive_lora/outputs/temporal_zero_equal_20261031_161721_4051873
COMMON=(--source-root "$SRC" --initial-adapter "$SRC/initial/initial_adapter.pt"
 --model /home/hongchang/whf/models/MiniCPM5-2B
 --train-data finetune/adaptive_lora/data/metamath_pilot_v1/train.json
 --dev-data finetune/adaptive_lora/data/metamath_pilot_v1/dev.json
 --source-data /home/hongchang/whf/datasets/MetaMathQA/MetaMathQA-395K.jsonl)
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
CUDA_VISIBLE_DEVICES='' "$PY" -B global84_step100_rank_gain.py dry-run "${COMMON[@]}" --output-dir /tmp/rank_gain_new_dry
"$PY" -B global84_step100_rank_gain.py smoke "${COMMON[@]}" --output-dir /tmp/rank_gain_new_smoke
# Explicit user launch only; sequential, never parallel on the single GPU.
RUN=finetune/adaptive_lora/outputs/step100_rank_allocation_six_v2
mkdir "$RUN"
for seed in 42 43 44; do
 "$PY" -B global84_step100_rank_gain.py train "${COMMON[@]}" --seed "$seed" --output-dir "$RUN/seed$seed" > "$RUN/seed$seed.log" 2>&1 || break
done
CUDA_VISIBLE_DEVICES='' "$PY" -B global84_step100_rank_statistics.py --runs "$RUN/seed42" "$RUN/seed43" "$RUN/seed44" --output-dir "$RUN/statistics"
```

Paths are CLI inputs, never embedded as execution defaults. Existing output directories are refused. `--arms` permits separate arm smoke; statistics still require a complete six-arm run per seed. Model admission requires30GiB free and output admission2GiB disk. Actual longest-sample peak may exceed the short smoke peak; monitor `nvidia-smi -l 2`. Outputs include order/hash, allocations, root provenance/report, per-arm train.jsonl, result.json and research adapter.pt (old+new branches, **not PEFT**).

## Executed validation

Previous revision: real full-model seed42, five arms, two updates each: PASS. Both train/eval insertion logits equal; all factors active/updated;381 base tensors compared with safetensors;480000/505600 exact parameter totals. Report: `/tmp/step100_gain_gpu_smoke_20261101/report.json`. Peak allocated approximately5.32GiB for these short samples, not a bound for2048 tokens. No full300-step run or Dev400 outcome has been computed. Historical AdamW remains unavailable.

CPU tests include shared-dropout RNG, single/multiple rank insertion, multiple modules, exact budgets/order, effective masked token evaluation and synthetic paired statistics; inherited loading/training/data/statistics regression tests remain enabled.

## Protocol revision: six arms and decay counterfactual

Adds static_DE from historical allocation_by_stage.json stage0 DE; **five** grown arms each add25600, base adds0. Training data/order, loss, LR, AdamW recreation and random direction rules are unchanged. Primary endpoints are overall Dev400 current_C−static_C and current_DE−static_DE, with Bonferroni97.5% paired bootstrap intervals across these two comparisons. Comparisons against random remain exploratory; group names, not fixed array offsets, identify contrasts.

For each added factor, save gradient norm at steps1 and2. For step2 A, copy its immediate pre-update value and compute the actual-dtype AdamW decay-only counterfactual `A_decay=A_before*(1-lr*weight_decay)`. Require positive step2 A gradient norm AND positive `||A_after-A_decay||`. A mere change from initialization is insufficient. This is a fresh optimizer and first-step A gradient is zero under zero B; the audit tests real beyond-decay optimizer movement, not just weight shrinkage.

The existing random allocation can have a different rank-concentration histogram. A histogram-matched random control is **deferred**, not silently added to the six arms: a future preregistered extension should separately permute each reference allocation's complete added-rank vector among compatible q modules and among compatible v modules (including zeros), using fixed independent seeds. This preserves per-type rank histogram and budget. It must not select permutations using Dev outcomes. Current random comparisons therefore still mix module identity and concentration effects; the primary temporal comparisons do not by themselves isolate these effects either.

Revision validation (seed42): CPU117/117 regression tests PASS; final statistics-only adjustment followed by9/9 targeted tests PASS. Real complete MiniCPM six-arm two-step smoke PASS at `/tmp/step100_six_arm_revision_20261101/report.json`. All six actual sample orders match order.json and its SHA256; train/eval insertion logits match;381 base tensors verified per arm. Added-A second-step beyond-decay norms are positive in all five grown arms. No300-step training executed. This smoke does not verify all three seeds or full-length training outcomes.
