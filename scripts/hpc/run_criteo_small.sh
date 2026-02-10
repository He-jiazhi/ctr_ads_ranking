#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:-data/criteo_train.tsv}"
OUT_DIR="${2:-outputs/criteo_small}"
MAX_ROWS="${3:-2000000}"

echo "Running small Criteo experiment"
echo "DATA_PATH=$DATA_PATH"
echo "OUT_DIR=$OUT_DIR"
echo "MAX_ROWS=$MAX_ROWS"

python -m src.cli train \
  --data_path "$DATA_PATH" \
  --out_dir "$OUT_DIR" \
  --max_rows "$MAX_ROWS" \
  --chunksize 20000 \
  --models lr ftrl \
  --use_frequency_encoding

python -m src.cli evaluate --run_dir "$OUT_DIR"
python -m src.cli ranking --run_dir "$OUT_DIR" --k 10
python -m src.cli slice --run_dir "$OUT_DIR" --topk_values 10
python -m src.cli simulate --run_dir "$OUT_DIR" --budget 500000
