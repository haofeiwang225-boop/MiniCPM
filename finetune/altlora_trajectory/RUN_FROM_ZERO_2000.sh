#!/usr/bin/env bash
set -euo pipefail
case "${1:-}" in
  formal|summarize)
    printf '%s\n' 'NOT IMPLEMENTED: stage-1 protocol review only. Training, initialization, measurement and benchmark runners are not yet connected.' >&2
    exit 2
    ;;
  *) printf '%s\n' 'Usage: RUN_FROM_ZERO_2000.sh formal|summarize (disabled pending implementation and review)' >&2; exit 2 ;;
esac
