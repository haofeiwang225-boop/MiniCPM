#!/usr/bin/env bash
# Read-only diagnostics; this script never invokes the training entry point.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${RALORA_PYTHON:-/home/hongchang/miniconda3/envs/ralora/bin/python}
RUN_DIR=${RUN_DIR:-finetune/adaptive_lora/outputs/dense_trajectory_500_calib8}
DIAG_ROOT=${DIAG_ROOT:-finetune/adaptive_lora/outputs/dense_diagnostic_suite}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/minicpm_adaptive_triton}
cli=("$PY" finetune/diagnose_dense_lora.py)
common=(--run_dir "$RUN_DIR" --seed 42)
read -r -a steps <<< "${STEPS:-2 20 100 500}"
checkpoint() {
  local step=$1
  if [ -f "$RUN_DIR/checkpoints/step_${step}.pt" ]; then
    echo "$RUN_DIR/checkpoints/step_${step}.pt"
  elif [ "$step" = "$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["max_steps"])' "$RUN_DIR/run_config.json")" ] && [ -f "$RUN_DIR/adapter.pt" ]; then
    echo "$RUN_DIR/adapter.pt"
  else
    echo "Missing checkpoint step $step in $RUN_DIR. Diagnostic JSON cannot reconstruct weights; no training started." >&2
    return 1
  fi
}
case "${1:-}" in
 gaps)
  inputs=()
  for step in 2 20 100 500; do inputs+=("$RUN_DIR/diagnostics/step_$step/modules.jsonl"); done
  "${cli[@]}" gaps --inputs "${inputs[@]}" --output_dir "$DIAG_ROOT/gaps"
  ;;
 initial)
  inputs=()
  for k in 1 2 4 8; do
    "${cli[@]}" initial "${common[@]}" --k "$k" --budget 480256 --output_dir "$DIAG_ROOT/initial/k$k"
    inputs+=("$DIAG_ROOT/initial/k$k/modules.jsonl")
  done
  "${cli[@]}" compare --inputs "${inputs[@]}" --output_dir "$DIAG_ROOT/initial/comparison"
  ;;
 repeat)
  adapter=$(checkpoint 500)
  "${cli[@]}" probe "${common[@]}" --adapter "$adapter" --k 8 --repeats 3 --output_dir "$DIAG_ROOT/repeat500"
  "${cli[@]}" compare --inputs "$DIAG_ROOT/repeat500/repeat_1/modules.jsonl" "$DIAG_ROOT/repeat500/repeat_2/modules.jsonl" "$DIAG_ROOT/repeat500/repeat_3/modules.jsonl" --output_dir "$DIAG_ROOT/repeat500/comparison"
  ;;
 calibration)
  # Source run has 9 training samples and batch_size=1. Sets overlap heavily.
  "${cli[@]}" initial "${common[@]}" --k 8 --budget 480256 --indices 0 1 2 3 4 5 6 7 --output_dir "$DIAG_ROOT/calibration/set_a"
  "${cli[@]}" initial "${common[@]}" --k 8 --budget 480256 --indices 1 2 3 4 5 6 7 8 --output_dir "$DIAG_ROOT/calibration/set_b"
  "${cli[@]}" initial "${common[@]}" --k 8 --budget 480256 --indices 0 1 2 3 4 5 7 8 --output_dir "$DIAG_ROOT/calibration/set_c"
  "${cli[@]}" compare --inputs "$DIAG_ROOT/calibration/set_a/modules.jsonl" "$DIAG_ROOT/calibration/set_b/modules.jsonl" "$DIAG_ROOT/calibration/set_c/modules.jsonl" --output_dir "$DIAG_ROOT/calibration/comparison"
  ;;
 checkpoint_k)
  for step in "${steps[@]}"; do checkpoint "$step" >/dev/null; done
  for step in "${steps[@]}"; do
    adapter=$(checkpoint "$step"); inputs=()
    for k in 1 2 4 8; do
      "${cli[@]}" probe "${common[@]}" --adapter "$adapter" --k "$k" --output_dir "$DIAG_ROOT/checkpoint_k/step_${step}_k$k"
      inputs+=("$DIAG_ROOT/checkpoint_k/step_${step}_k$k/modules.jsonl")
    done
    "${cli[@]}" compare --inputs "${inputs[@]}" --output_dir "$DIAG_ROOT/checkpoint_k/comparison_$step"
  done
  ;;
 oracle)
  adapter=$(checkpoint 500)
  selected=()
  for layer in 0 20 41; do
    selected+=("model.layers.$layer.self_attn.q_proj" "model.layers.$layer.self_attn.v_proj")
  done
  "${cli[@]}" probe "${common[@]}" --adapter "$adapter" --k 8 --modules "${selected[@]}" --oracle_modules "${selected[@]}" --output_dir "$DIAG_ROOT/oracle500"
  ;;
 random)
  for step in "${steps[@]}"; do checkpoint "$step" >/dev/null; done
  for step in "${steps[@]}"; do
    adapter=$(checkpoint "$step")
    "${cli[@]}" probe "${common[@]}" --adapter "$adapter" --k 8 --random_trials 100 --save_gradients --drift_rank 8 --output_dir "$DIAG_ROOT/random/step_$step"
  done
  ;;
 drift)
  inputs=()
  for step in 2 20 100 500; do
    checkpoint "$step" >/dev/null
    test -d "$DIAG_ROOT/random/step_$step/gradient_cache" || { echo "Missing cached gradients for step $step; run random first." >&2; exit 1; }
    inputs+=("$DIAG_ROOT/random/step_$step")
  done
  "${cli[@]}" drift --inputs "${inputs[@]}" --output_dir "$DIAG_ROOT/drift"
  ;;
 *) echo "Usage: $0 {gaps|initial|repeat|calibration|checkpoint_k|oracle|random|drift}" >&2; exit 2 ;;
esac
