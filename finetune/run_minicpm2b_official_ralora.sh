#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${RALORA_PYTHON:-/home/hongchang/miniconda3/envs/ralora/bin/python}" \
  finetune/train_adaptive_lora.py --lora_structure block --adaptive_lora false "$@"
