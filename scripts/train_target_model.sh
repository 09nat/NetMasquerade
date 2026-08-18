#!/usr/bin/env bash
# Step 1: train and evaluate the released NetBeacon target classifier.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${1:-$ROOT/src/advGenerate/environments/NetBeacon/config/netbeacon.yaml}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
cd "$ROOT"
exec "${PYTHON_BIN:-python}" -u -m advGenerate.environments.NetBeacon.train \
  --config "$CONFIG"
