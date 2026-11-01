# Step100 allocation-only real rank gain

Independent entry `global84_step100_rank_gain.py`; reviewed base 413b67eca5627bfa3197ef49bb073eae7549eda8. Old training, GID and temporal programs are unchanged.

## Frozen design

Three paired seeds 42/43/44, source step100, fresh AdamW for every arm. Five arms: base; random allocation (four draws with replacement within each q/v type); static_C (explicitly the historical step0 **C**, not DE list); current_C (step100 C); current_DE (step100 DE). No directions are fitted here. Allocation audit samples and Dev never select random directions.

All grown arms add q4/v4 = 25600 parameters, checked from actual shapes. Original 480000 parameters remain trainable; grown total 505600. Shared CPU-local random unit A rows are keyed by seed/module/added-rank index. New B is zero, new branch scaling is 1; old scaling remains alpha/r. No orthogonal or SVD direction substitution. Multiple added ranks in one module are supported.

Wrapper reproduces the official Dense forward arithmetic and shares its **single** dropout tensor with the new branch. Old A/B Parameter identity/value and scaling are preserved at insertion. Full-model train/eval logits must be bitwise equal before any update. First-step new B gradient/update, second-step new A update, all old/new gradient activity and final changes are required. Gradient evidence, rather than weight decay alone, gates engineering PASS.

300 updates, batch1, masked sample-mean loss, dynamic2048, dropout.1, BF16 base/FP32 factors, nonreentrant checkpointing; AdamW defaults with weight decay.01. At local step t, used LR is `5e-5*(1-(100+t-1)/2000)`. This reproduces the historical linear global-step curve without restarting at full LR. All84 original adapters train; base stays frozen.

## Pairing limits

Historical snapshots contain factors, not AdamW/RNG. This is **not seamless continuation**. Source logs do not contain explicit sample indices. The audited historical DataLoader permutation is reconstructed using seed and runtime, and positions101–400 are saved/hash-recorded. It cannot be independently verified against a historical sample-index manifest. Every new arm does share the exact saved order and dropout seed. Same seed alone is not a claim of bitwise historical dropout continuation. Train-mode insertion equivalence is tested explicitly. Data/source/checkpoint/code hashes are checked; incompatible historical code fails closed.

## Evaluation and statistics

Frozen Dev400 (GSM200/MATH200) has prior research use, not blind. Evaluate local0/100/200/300; primary endpoint300, no best-checkpoint selection. Store each masked-loss sum, shifted supervised token count, sample NLL and source metadata. Report token-weighted and question-mean loss plus GSM/MATH. No automatic retries. Failed runs retain traceback/report and cannot be consumed as successful full training.

CPU statistics require all three seeds and all five arms, matching metadata/token counts. Pair seed IDs and original-question clusters across arms for10000 bootstrap draws. Primary C−random and DE−random use97.5% Bonferroni intervals; others exploratory95%. Only three independent training trajectories: intervals have substantial limitations, not1200 independent training repeats. Actual validation results, not gradient proxies or engineering PASS, determine observed benefit.

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
RUN=finetune/adaptive_lora/outputs/step100_rank_allocation_v1
mkdir "$RUN"
for seed in 42 43 44; do
 "$PY" -B global84_step100_rank_gain.py train "${COMMON[@]}" --seed "$seed" --output-dir "$RUN/seed$seed" > "$RUN/seed$seed.log" 2>&1 || break
done
CUDA_VISIBLE_DEVICES='' "$PY" -B global84_step100_rank_statistics.py --runs "$RUN/seed42" "$RUN/seed43" "$RUN/seed44" --output-dir "$RUN/statistics"
```

Paths are CLI inputs, never embedded as execution defaults. Existing output directories are refused. `--arms` permits separate arm smoke; statistics still require a complete five-arm run per seed. Model admission requires30GiB free and output admission2GiB disk. Actual longest-sample peak may exceed the short smoke peak; monitor `nvidia-smi -l 2`. Outputs include order/hash, allocations, root provenance/report, per-arm train.jsonl, result.json and research adapter.pt (old+new branches, **not PEFT**).

## Executed validation

Real full-model seed42, all five arms, two updates each: PASS. Both train/eval insertion logits equal; all factors active/updated;381 base tensors compared with safetensors;480000/505600 exact parameter totals. Report: `/tmp/step100_gain_gpu_smoke_20261101/report.json`. Peak allocated approximately5.32GiB for these short samples, not a bound for2048 tokens. No full300-step run or Dev400 outcome has been computed. Historical AdamW remains unavailable.

CPU tests include shared-dropout RNG, single/multiple rank insertion, multiple modules, exact budgets/order, effective masked token evaluation and synthetic paired statistics; inherited loading/training/data/statistics regression tests remain enabled.
