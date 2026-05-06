#!/usr/bin/env bash
# Launch script for Keural Vision Encoder training.
#
# Usage:
#   bash scripts/launch_train.sh                 # POC config, full settings
#   bash scripts/launch_train.sh smoke           # Smoke-test config, 5 steps
#   bash scripts/launch_train.sh poc-synthetic   # POC config but on synthetic data
#
# The synthetic mode is for verifying the loop runs correctly. To use real
# CC3M/LAION data, plug your loader into data/datasets.py and remove
# --synthetic flags below.

set -euo pipefail

MODE="${1:-poc-synthetic}"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

case "$MODE" in
  smoke)
    echo "Running smoke-test (5 steps, tiny config)..."
    python -m training.train \
      --config configs/keural_smoke.yaml \
      --synthetic --synthetic-size 64 \
      --max-steps 5 \
      --output-dir "./outputs/smoke-${TIMESTAMP}"
    ;;
  poc-synthetic)
    echo "Running POC config on synthetic data (architecture validation)..."
    python -m training.train \
      --config configs/keural_tiny_poc.yaml \
      --synthetic --synthetic-size 5000 \
      --output-dir "./outputs/poc-synth-${TIMESTAMP}"
    ;;
  poc)
    echo "Running POC config on real data..."
    echo "NOTE: This requires you to have plugged a real loader into data/datasets.py."
    python -m training.train \
      --config configs/keural_tiny_poc.yaml \
      --output-dir "./outputs/poc-${TIMESTAMP}"
    ;;
  *)
    echo "Unknown mode: $MODE"
    echo "Usage: $0 {smoke|poc-synthetic|poc}"
    exit 1
    ;;
esac
