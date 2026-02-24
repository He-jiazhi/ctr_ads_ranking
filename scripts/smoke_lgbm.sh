#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
DATA_PATH="${DATA_PATH:-data/criteo_train.tsv}"
OUT="${1:-outputs/smoke_lgbm_$(date +%Y%m%d_%H%M%S)}"

echo "Using python: ${PYTHON_BIN}"
echo "Using data: ${DATA_PATH}"
echo "Output dir: ${OUT}"

"${PYTHON_BIN}" -m src.cli train \
  --data_path "${DATA_PATH}" \
  --out_dir "${OUT}" \
  --models lgbm \
  --lgbm_fit \
  --max_rows 200000 \
  --chunksize 50000

"${PYTHON_BIN}" -m src.cli evaluate --run_dir "${OUT}"
"${PYTHON_BIN}" -m src.cli ranking --run_dir "${OUT}" --k 10
"${PYTHON_BIN}" -m src.cli simulate --run_dir "${OUT}" --budget 5000000

"${PYTHON_BIN}" - <<PY
from pathlib import Path
import pandas as pd

out = Path("${OUT}")
pred_path = out / "metrics" / "test_pred.parquet"
df = pd.read_parquet(pred_path)
if "p_lgbm" not in df.columns:
    raise SystemExit(f"p_lgbm not found in {pred_path}")
print(f"p_lgbm exists in {pred_path} (rows={len(df)})")
PY
