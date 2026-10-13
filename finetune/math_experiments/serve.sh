#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${MODEL:?Set MODEL to base or verified merged model directory}"
: "${SERVER_DIR:?Set a new SERVER_DIR}"
[[ ! -e "$SERVER_DIR" ]] || { echo 'Refusing to overwrite SERVER_DIR' >&2; exit 2; }
mkdir -p "$SERVER_DIR"
exec 9>/tmp/minicpm_math_experiments_gpu.lock
flock -n 9 || { echo 'Another math experiment holds the GPU lock' >&2; exit 2; }
free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
(( free >= 40000 )) || { echo 'Need at least 40000 MiB free for this sequential server plan' >&2; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
VENV="${VLLM_ENV:-/home/hongchang/whf/Supervised Fine-Tuning/vllm}"
"$VENV/bin/python" -B finetune/math_experiments/record_server.py --model "$MODEL" --output "$SERVER_DIR/server.json" --port "${PORT:-8000}" --seed 42
args=(serve "$MODEL" --served-model-name math-model --host 127.0.0.1 --port "${PORT:-8000}" --dtype bfloat16 --max-model-len 131072 --generation-config vllm --gpu-memory-utilization 0.65 --max-num-seqs 1 --seed 42)
printf '%q ' "$VENV/bin/vllm" "${args[@]}" > "$SERVER_DIR/command.txt"
printf '\n' >> "$SERVER_DIR/command.txt"
exec "$VENV/bin/vllm" "${args[@]}"
