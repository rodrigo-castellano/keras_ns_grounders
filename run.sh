#!/usr/bin/env bash
# run.sh — Train on Countries S3 with DCR + RotatE
# Usage: bash run.sh  (or ./run.sh after chmod +x run.sh)

set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$HOME/miniconda3/envs/gpu/bin/python"

echo "Training: countries_s3 | DCR | RotatE"
cd "$ROOT_DIR"
"$PYTHON" experiments/runner.py \
    --d countries_s3 \
    --m dcr \
    --g backward_1_1 \
    --kge rotate \
    --epochs 100
