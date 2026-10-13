#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
[[ "${PROTOCOL_FROZEN:-no}" == yes ]] || { echo 'Freeze models/protocol before using MATH-500. Set PROTOCOL_FROZEN=yes only after review.' >&2; exit 2; }
: "${OUTPUT_DIR:?Set new MATH-500 output directory}"
: "${SERVER_DIR:?Set server record directory}"
[[ ! -e "$OUTPUT_DIR" ]] || exit 2
mkdir -p "$OUTPUT_DIR"
# Config outside the output directory: existing endpoint evaluator rejects nonempty outputs.
config=$(mktemp /tmp/minicpm_math500_config_XXXXXX.yaml)
export config
/home/hongchang/miniconda3/envs/minicpm5/bin/python -B - <<'PY'
import json,os,yaml
from pathlib import Path
r=json.loads((Path(os.environ['SERVER_DIR'])/'server.json').read_text())
c=yaml.safe_load(Path('benchmarks/math500/config.yaml').read_text())
c.update(model=r['model'],served_model_name=r['served_model_name'],base_url=f"http://127.0.0.1:{r['port']}/v1",output_dir=os.environ['OUTPUT_DIR'],server_record=r,seed=r['seed'])
Path(os.environ['config']).write_text(yaml.safe_dump(c))
PY
exec /home/hongchang/miniconda3/envs/minicpm5/bin/python -B benchmarks/math500/evaluate.py --config "$config"
