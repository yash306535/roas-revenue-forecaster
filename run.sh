#!/usr/bin/env bash
# Single entry point for the testing pipeline.
#   ./run.sh <DATA_DIR> <MODEL_PATH> <OUTPUT_PATH>
# All three are optional; sensible defaults are used for local runs.
set -euo pipefail

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

FEATURES_FILE="features.parquet"

mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "==> 1/2 Generating features from: $DATA_DIR"
python src/generate_features.py --data-dir "$DATA_DIR" --out "$FEATURES_FILE"

echo "==> 2/2 Producing predictions"
python src/predict.py \
    --features "$FEATURES_FILE" \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH"

echo "Done. Predictions written to $OUTPUT_PATH"
