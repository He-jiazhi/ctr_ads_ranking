#!/usr/bin/env bash
#SBATCH -J ctr_ftrl_full
#SBATCH -p preempt
#SBATCH -A ycheng
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 10:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
set -euo pipefail
module purge
module load python/3.11.9
cd "$HOME/ctr_work/projects/ctr_ads_ranking"
source .venv/bin/activate
DATA="$HOME/ctr_work/data/criteo_train.tsv"
OUT="$HOME/ctr_work/runs/ftrl_full_$(date +%m%d_%H%M%S)"
python -m src.cli train \
  --data_path "$DATA" \
  --out_dir "$OUT" \
  --max_rows 45840617 \
  --chunksize 200000 \
  --models ftrl \
  --ftrl_alpha 0.05 \
  --ftrl_beta 1.0 \
  --ftrl_l1 1.0 \
  --ftrl_l2 1.0
python -m src.cli evaluate --run_dir "$OUT"
python -m src.cli ranking  --run_dir "$OUT" --k 10
python -m src.cli slice    --run_dir "$OUT" --topk_values 10
python -m src.cli simulate --run_dir "$OUT" --budget 5000000
echo "DONE. Run dir: $OUT"
