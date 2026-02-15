#!/usr/bin/env bash
#SBATCH -J ctr_lr_full
#SBATCH -p preempt
#SBATCH -A ycheng
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 08:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
set -euo pipefail
module purge
module load python/3.11.9
cd "$HOME/ctr_work/projects/ctr_ads_ranking"
source .venv/bin/activate

DATA="$HOME/ctr_work/data/criteo_train.tsv"
OUT="$HOME/ctr_work/runs/lr_full_$(date +%m%d_%H%M%S)"
python -m src.cli train \
  --data_path "$DATA" \
  --out_dir "$OUT" \
  --max_rows 45840617 \
  --chunksize 200000 \
  --models lr \
  --lr_alpha 1e-2 --lr_l1_ratio 0.0
echo "DONE. Run dir: $OUT"
