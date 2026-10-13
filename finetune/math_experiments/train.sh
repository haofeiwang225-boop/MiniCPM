#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${MODE:?vanilla, budget_matched, shuffled_gid or replay_gid}"
: "${OUTPUT_DIR:?Set new OUTPUT_DIR}"
[[ ! -e "$OUTPUT_DIR" ]] || { echo 'Output already exists' >&2; exit 2; }
exec 9>/tmp/minicpm_math_experiments_gpu.lock
flock -n 9 || { echo 'Another math experiment holds GPU lock' >&2; exit 2; }
extra=()
case "$MODE" in
 vanilla|budget_matched) ;;
 shuffled_gid|replay_gid) extra=(--rank_manifest "${RANK_MANIFEST:-finetune/adaptive_lora/outputs/metamath_pilot_control_2000_seed42/initial_probe/rank.json}") ;;
 *) echo 'Unknown MODE' >&2; exit 2 ;;
esac
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TRITON_CACHE_DIR=/tmp/minicpm_adaptive_triton
exec /home/hongchang/miniconda3/envs/ralora/bin/python -B finetune/train_adaptive_lora.py \
 --model_name_or_path /home/hongchang/whf/models/MiniCPM5-2B --model_family minicpm5 \
 --train_data_path finetune/adaptive_lora/data/metamath_pilot_v1/train.json --output_dir "$OUTPUT_DIR" \
 --seed 42 --batch_size 1 --dynamic_length true --model_max_length 2048 --max_steps "${STEPS:-2000}" \
 --learning_rate 5e-5 --weight_decay 0.01 --device cuda --dtype bfloat16 --analysis_device cuda \
 --lora_structure dense --dense_init_mode "$MODE" --dense_lora_param_budget 480000 --dense_rank_max 4 \
 --lora_rank 2 --lora_alpha 4 --lora_dropout 0.1 --weight_a_init_method kaiming \
 --adaptive_lora false --direction_correction none --gradient_est_n_steps 8 --adaptive_probe_batches 8 "${extra[@]}"
