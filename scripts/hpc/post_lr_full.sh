#!/usr/bin/env bash
#SBATCH -J ctr_lr_full_post
#SBATCH -p preempt
#SBATCH -A ycheng
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

set -euo pipefail
module purge
module load python/3.11.9
cd "$HOME/ctr_work/projects/ctr_ads_ranking"
source .venv/bin/activate
RUN_DIR="/ihome/ycheng/jih253/ctr_work/runs/lr_full_0210_151111"

python -m src.cli evaluate --run_dir "$RUN_DIR"
python -m src.cli ranking  --run_dir "$RUN_DIR" --k 10
python -m src.cli slice    --run_dir "$RUN_DIR" --topk_values 10
python -m src.cli simulate --run_dir "$RUN_DIR" --budget 5000000
echo "DONE postprocess: $RUN_DIR"
