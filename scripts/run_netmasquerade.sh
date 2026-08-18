#!/usr/bin/env bash
# Step 2: train NetMasquerade and reproduce feedback/no-feedback evaluation.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "${PYTHON_BIN:-python}" -u -m advGenerate.train \
  --rl-config "$ROOT/src/advGenerate/config/sac.yaml" \
  --bert-config "$ROOT/src/trafficMimic/config/bert.yaml" "$@"
