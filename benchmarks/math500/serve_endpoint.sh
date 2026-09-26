#!/usr/bin/env bash
set -euo pipefail
# Dedicated local endpoint. Do not enable a reasoning parser: preserve raw </think>.
VLLM_BIN="${VLLM_BIN:-/home/hongchang/whf/Supervised Fine-Tuning/vllm/bin/vllm}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec "$VLLM_BIN" serve /home/hongchang/whf/models/MiniCPM5-2B \
  --served-model-name MiniCPM5-2B \
  --host 127.0.0.1 --port 8000 \
  --dtype bfloat16 \
  --max-model-len 131072 \
  --generation-config vllm \
  --gpu-memory-utilization 0.65 \
  --max-num-seqs 1
