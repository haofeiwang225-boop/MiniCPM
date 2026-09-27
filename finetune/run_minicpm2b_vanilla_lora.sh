#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Original PEFT entrypoint, target modules, rank, initialization and forward unchanged.
exec "${PEFT_PYTHON:-python}" finetune/finetune.py --use_lora true "$@"
