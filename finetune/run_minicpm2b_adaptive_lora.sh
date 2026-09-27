#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${RALORA_PYTHON:-/home/hongchang/miniconda3/envs/ralora/bin/python}" \
  finetune/train_adaptive_lora.py --adaptive_lora true "$@"
