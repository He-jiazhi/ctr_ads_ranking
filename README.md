# CTR Prediction & Ads Ranking Optimization (Offline)

End-to-end **CTR prediction → offline ranking evaluation → budgeted delivery simulation** on large-scale ad click logs (Criteo DAC format).  
Designed to be **laptop-friendly** via **streaming / chunked training**, with reproducible CLI runs.  
(Full-scale experiments were run on Pitt CRC/HPC; see `scripts/hpc/`.)

---

## What’s in this repo

- **Leakage-safe split**: time-based split by log order (train/val/test = 80%/10%/10%) within `--max_rows`
- **Feature engineering**
  - numeric transforms (safe `log1p`, missing handling)
  - **feature hashing** for high-cardinality categoricals (`C1..C26`)
  - optional **frequency encoding**
- **Models**
  - Logistic Regression via SGD (streaming baseline)
  - **FTRL-Proximal** (online learning baseline for CTR)
  - LightGBM (optional; best for smaller/sampled runs in this repo)
- **Evaluation**
  - AUC / LogLoss / Brier
  - offline ranking metric: **NDCG@K** on pseudo-groups
  - slice / segment analysis by categorical keys (`C1/C2/C3`)
- **Offline “delivery” simulation** under budget constraints (expected clicks / revenue proxy)

> This is an **offline** project: no online serving, no real-time auction.  
> The delivery simulation is a simplified proxy to demonstrate decision-making under budget.

---

## Dataset

This repo expects the **Criteo Display Advertising Challenge** format:

- `label`, `I1..I13` (numerical), `C1..C26` (categorical)
- TSV file **without header**
- Total columns = **40**

### Option A (recommended): official Criteo tarball (no Kaggle auth needed)

```bash
mkdir -p data
curl -L -o data/criteo_kaggle_dac.tar.gz \
  "http://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz"
tar -xzf data/criteo_kaggle_dac.tar.gz -C data
mv data/train.txt data/criteo_train.tsv
```

### Option B: Kaggle CLI 

```bash
kaggle competitions download -c criteo-display-ad-challenge -f train.txt -p data
unzip data/train.txt.zip -d data
mv data/train.txt data/criteo_train.tsv
```

### Quick sanity check


```bash
python - << 'PY'
import csv
p="data/criteo_train.tsv"
with open(p, "r") as f:
    row = next(csv.reader(f, delimiter="\t"))
print("n_cols =", len(row))  # expect 40
print("first_5 =", row[:5])
print("last_5  =", row[-5:])
PY
```

> If your downloaded file is named differently, just rename it to `data/criteo_train.tsv`.


---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

Optional for parquet I/O (recommended):
```bash
pip install pyarrow
```

Optional model:
```bash
pip install lightgbm
```


---

## Quickstart (smoke run)

Train models on a small sample:

### Train (LR, 200k rows)

```bash
python -m src.cli train \
  --data_path data/criteo_train.tsv \
  --out_dir outputs/smoke_lr \
  --max_rows 200000 \
  --chunksize 50000 \
  --models lr \
  --lr_alpha 1e-2 \
  --lr_l1_ratio 0.0
```

### Evaluate + ranking + slice + simulation

```bash
python -m src.cli evaluate --run_dir outputs/smoke_lr
python -m src.cli ranking  --run_dir outputs/smoke_lr --k 10
python -m src.cli slice    --run_dir outputs/smoke_lr --topk_values 10
python -m src.cli simulate --run_dir outputs/smoke_lr --budget 5000000
```


## Results (Criteo DAC)

**Split:** time-based by log order, within `--max_rows` (train/val/test = 80%/10%/10%)  
**NDCG grouping:** pseudo-groups via row buckets (default group_size=50)
**Features:** hashing for categoricals (`n_hash_buckets=2^20`), no frequency encoding unless explicitly via --use_frequency_encoding


### Offline CTR metrics

| Run | Model | max_rows | Val AUC | Val LogLoss | Test AUC | Test LogLoss | NDCG@10 |
|---|---|---:|---:|---:|---:|---:|---:|
| LR (10M) | SGD LR | 10,000,000 | 0.7323 | 0.4955 | 0.7259 | 0.5094 | 0.5565 |
| LR (full) | SGD LR | 45,840,617 | 0.7337 | 0.4997 | 0.7333 | 0.5046 | 0.5636 |
| FTRL (full) | FTRL-Prox | 45,840,617 | 0.7754 | 0.4929 | 0.7737 | 0.4986 | 0.6265 |
| LGBM (smoke) | LightGBM | 200,000 | 0.7782 | 0.4753 | 0.7672 | 0.4859 | 0.6314 |

> Metrics are computed on the internal val/test splits induced by `--max_rows`.

### Budgeted delivery simulation (budget = 5,000,000)

| Run | expected_clicks | realized_clicks | expected_revenue | avg_pctr_selected |
|---|---:|---:|---:|---:|
| LR (10M)   | 248,132.99 | 259,595 | 281,223.90 | 0.24813 |
| LR (full)  | 1,190,458.33 | 1,188,541 | 1,349,350.15 | 0.25970 |
| FTRL (full) | 1,101,680.75 | 1,188,541 | 1,248,783.00 | 0.24033 |
| LGBM (smoke) | 5,305.77 | 5,385 | 6,035.28 | 0.26529 |


---

## Notes on NDCG grouping

CTR logs don’t include a natural “query/session” group like search ranking datasets.  
To still evaluate ranking quality, this repo computes **NDCG@K** over pseudo-groups, e.g. row buckets:

- group id ≈ index // group_size (default group_size=50)
- Smaller group size → noisier NDCG  
- Larger group size → more stable NDCG but less granular  

---

## Repo structure

- `src/data.py` : chunked loading + time split  
- `src/features.py` : hashing + frequency encoding + sparse matrices  
- `src/models.py` : SGD logistic, FTRL-Proximal, LightGBM wrapper  
- `src/eval.py` : AUC/LogLoss, ranking metrics, slicing  
- `src/simulate.py` : simple delivery simulation under budget constraints  
- `src/cli.py` : command-line entrypoints  
- `scripts/hpc/` : Slurm scripts used on Pitt CRC/HPC (recommended for large runs)

---

## Reproducibility tips

- Start small: `--max_rows 200000`  
- Scale up gradually: 2M → 10M → full (I/O + time)
- If you hit memory issues, prefer smaller --chunksize.  

## Example commands

### LR (10M rows)
```bash
python -m src.cli train \
  --data_path data/criteo_train.tsv \
  --out_dir outputs/lr_10m \
  --max_rows 10000000 \
  --chunksize 200000 \
  --models lr \
  --lr_alpha 1e-2 --lr_l1_ratio 0.0

python -m src.cli evaluate --run_dir outputs/lr_10m
python -m src.cli ranking  --run_dir outputs/lr_10m --k 10
python -m src.cli slice    --run_dir outputs/lr_10m --topk_values 10
python -m src.cli simulate --run_dir outputs/lr_10m --budget 5000000
```

### LR (Full, 45.8M rows)

```bash
python -m src.cli train \
  --data_path data/criteo_train.tsv \
  --out_dir outputs/lr_full \
  --max_rows 45840617 \
  --chunksize 200000 \
  --models lr \
  --lr_alpha 1e-2 --lr_l1_ratio 0.0

python -m src.cli evaluate --run_dir outputs/lr_full
python -m src.cli ranking  --run_dir outputs/lr_full --k 10
python -m src.cli slice    --run_dir outputs/lr_full --topk_values 10
python -m src.cli simulate --run_dir outputs/lr_full --budget 5000000
```

### FTRL (Full)

```bash
python -m src.cli train \
  --data_path data/criteo_train.tsv \
  --out_dir outputs/ftrl_full \
  --max_rows 45840617 \
  --chunksize 200000 \
  --models ftrl \
  --ftrl_alpha 0.05 \
  --ftrl_beta 1.0 \
  --ftrl_l1 1.0 \
  --ftrl_l2 1.0

python -m src.cli evaluate --run_dir outputs/ftrl_full
python -m src.cli ranking  --run_dir outputs/ftrl_full --k 10
python -m src.cli slice    --run_dir outputs/ftrl_full --topk_values 10
python -m src.cli simulate --run_dir outputs/ftrl_full --budget 5000000
```

## LightGBM (smoke)
```bash
bash scripts/smoke_lgbm.sh outputs/smoke_lgbm
```
> LightGBM in this repo is intended for smoke/sampled runs. Full-data training may require further engineering (sampling / external memory / distributed training).
