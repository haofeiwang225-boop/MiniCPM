#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${OUTPUT_DIR:?Set new output directory}"
: "${SERVER_DIR:?Set server record directory}"
exec /home/hongchang/miniconda3/envs/minicpm5/bin/python -B finetune/math_experiments/dev_accuracy.py \
 --server_record "$SERVER_DIR/server.json" --base_url "http://127.0.0.1:${PORT:-8000}/v1" \
 --max_samples "${SAMPLES:-10}" --output_dir "$OUTPUT_DIR"
