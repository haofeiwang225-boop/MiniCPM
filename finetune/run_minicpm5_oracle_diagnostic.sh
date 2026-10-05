#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-smoke}
if [ "$#" -gt 0 ]; then shift; fi
case "$mode" in
  smoke) options=(--adaptive_lora true --max_steps 4 --adaptive_probe_step 2
                  --output_dir finetune/adaptive_lora/outputs/dense_smoke_initial8_diag8) ;;
  trajectory) options=(--adaptive_lora false --max_steps 500 --diagnostic_steps 2 20 100 500
                       --output_dir finetune/adaptive_lora/outputs/dense_trajectory_500_initial8_diag8) ;;
  *) echo "Usage: $0 {smoke|trajectory} [additional args]" >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/minicpm_adaptive_triton}
mkdir -p "$TRITON_CACHE_DIR"
exec "${RALORA_PYTHON:-/home/hongchang/miniconda3/envs/ralora/bin/python}" \
  finetune/train_adaptive_lora.py \
  --model_name_or_path /home/hongchang/whf/models/MiniCPM5-2B --model_family minicpm5 \
  --device cuda --dtype bfloat16 --analysis_device cuda --model_max_length 256 \
  --batch_size 1 --lora_rank 2 --lora_alpha 4 --lora_structure dense --dense_rank_max 4 --dense_lora_param_budget 480256 \
  --adaptive_rank_max 32 --adaptive_rank_step_cap 2 \
  --gradient_est_n_steps 8 --adaptive_probe_batches 8 \
  --direction_correction none --adaptive_rank_init zero_b_residual_v \
  "${options[@]}" "$@"
