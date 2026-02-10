import numpy as np
import pandas as pd
from pathlib import Path

from src.data import CRITEO_COLS, CRITEO_NUM_COLS, CRITEO_CAT_COLS

def main(out_path="data/synth_criteo.tsv", n=200000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame()
    # binary labels with some signal
    df["label"] = rng.binomial(1, 0.2, size=n)
    for c in CRITEO_NUM_COLS:
        x = rng.normal(size=n).astype(np.float32)
        df[c] = x
    for c in CRITEO_CAT_COLS:
        # 1k categories
        df[c] = rng.integers(0, 1000, size=n).astype(str)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", header=False, index=False)
    print("Wrote", out_path, "shape", df.shape)

if __name__ == "__main__":
    main()