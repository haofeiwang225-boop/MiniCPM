#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
has_steps=false
has_output=false
for arg in "$@"; do
  case "$arg" in
    --direction_correction_steps) has_steps=true ;;
    --output_dir|--output_dir=*) has_output=true ;;
  esac
done
if ! "$has_steps" || ! "$has_output"; then
  echo "Pass --direction_correction_steps ... and --output_dir NEW_DIRECTORY explicitly." >&2
  exit 2
fi
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/minicpm_adaptive_triton}
mkdir -p "$TRITON_CACHE_DIR"
exec "${RALORA_PYTHON:-/home/hongchang/miniconda3/envs/ralora/bin/python}" \
  finetune/train_adaptive_lora.py \
  --model_name_or_path /home/hongchang/whf/models/MiniCPM5-2B --model_family minicpm5 \
  --device cuda --dtype bfloat16 --analysis_device cuda \
  --batch_size 1 --model_max_length 256 --lora_rank 2 --lora_alpha 4 \
  --lora_structure dense --dense_rank_max 4 --dense_lora_param_budget 480256 \
  --gradient_est_n_steps 8 --adaptive_probe_batches 8 \
  --adaptive_lora false --direction_correction geometric \
  "$@"
