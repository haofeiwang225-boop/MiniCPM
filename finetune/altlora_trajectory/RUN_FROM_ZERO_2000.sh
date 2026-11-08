#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO/finetune:$REPO${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-/home/hongchang/miniconda3/envs/minicpm5/bin/python}"
OUT="${OUT:-$REPO/finetune/altlora_trajectory/outputs/modes_from_zero_seed42_2000steps_20261108}"
case "${1:-}" in
 formal)
   : "${CONFIG:?Set CONFIG to reviewed input manifest JSON}"
   "$PYTHON" -B -m altlora_from_zero.run prepare --config "$CONFIG" --output "$OUT"
   for arm in vanilla altlora_false altlora_true; do
     "$PYTHON" -B -m altlora_from_zero.run train --output "$OUT" --arm "$arm"
   done
   # Evaluation intentionally separate: endpoints must serve verified exports.
   ;;
 resume)
   : "${ARM:?Set ARM}" "${CHECKPOINT:?Set same-arm CHECKPOINT}"
   "$PYTHON" -B -m altlora_from_zero.run train --output "$OUT" --arm "$ARM" --resume "$CHECKPOINT"
   ;;
 train-arm)
   : "${ARM:?Set ARM}"
   "$PYTHON" -B -m altlora_from_zero.run train --output "$OUT" --arm "$ARM"
   ;;
 summarize) "$PYTHON" -B -m altlora_from_zero.summarize --root "$OUT" ;;
 *) echo 'Usage: formal | resume | train-arm | summarize' >&2; exit 2 ;;
esac
