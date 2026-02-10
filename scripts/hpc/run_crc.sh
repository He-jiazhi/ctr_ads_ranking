#!/usr/bin/env bash
#SBATCH -J ctr_lr
#SBATCH -p smp
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail

module load python/3.11.9
cd $HOME/ctr_work/projects/ctr_ads_ranking
source .venv/bin/activate

DATA=$HOME/ctr_work/data/criteo_train.tsv
OUT=$HOME/ctr_work/runs/lr_2m

python -m src.cli train \
  --data_path "$DATA" \
  --out_dir "$OUT" \
  --max_rows 2000000 \
  --chunksize 200000 \
  --models lr \
  --lr_alpha 1e-2 --lr_l1_ratio 0.0

python -m src.cli evaluate --run_dir "$OUT"
python -m src.cli ranking  --run_dir "$OUT" --k 10
python -m src.cli slice    --run_dir "$OUT" --topk_values 10
python -m src.cli simulate --run_dir "$OUT" --budget 5000000
