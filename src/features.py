from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp


from sklearn.utils import murmurhash3_32

def _hash_token(token: str, n_buckets: int) -> int:
    return murmurhash3_32(token, positive=True) % n_buckets


@dataclass
class FeatureConfig:
    n_hash_buckets: int = 2 ** 20  # ~1M dims (sparse)
    use_frequency_encoding: bool = True
    min_freq: int = 2  # for frequency encoding table
    numeric_fill: float = 0.0


class FrequencyEncoder:
    """
    Frequency encoding for categorical columns.
    Encodes each category to its frequency (count) in training data.
    """
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.tables: Dict[str, Dict[str, float]] = {}
        self.default: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, cat_cols: Sequence[str]) -> "FrequencyEncoder":
        for c in cat_cols:
            vc = df[c].astype("string").fillna("NA").value_counts()
            vc = vc[vc >= self.min_freq]
            total = float(len(df))
            table = (vc / total).to_dict()
            self.tables[c] = table
            # default is 0 frequency for unseen/rare
            self.default[c] = 0.0
        return self

    def transform(self, df: pd.DataFrame, cat_cols: Sequence[str]) -> np.ndarray:
        feats = []
        for c in cat_cols:
            s = df[c].astype("string").fillna("NA")
            table = self.tables.get(c, {})
            default = self.default.get(c, 0.0)
            feats.append(s.map(lambda x: table.get(str(x), default)).astype("float32").to_numpy())
        return np.vstack(feats).T  # shape (n, len(cat_cols))


def build_hashed_csr(
    df: pd.DataFrame,
    cat_cols: Sequence[str],
    n_buckets: int,
    add_column_prefix: bool = True,
) -> sp.csr_matrix:
    """
    Build a CSR matrix for hashed one-hot categorical features.

    Each (row, col) has value 1.0 where col = hash(f"{col}={val}") % n_buckets.
    """
    n = len(df)
    indptr = np.zeros(n + 1, dtype=np.int64)
    indices_list: List[int] = []
    data_list: List[float] = []

    for i, row in enumerate(df[cat_cols].itertuples(index=False, name=None)):
        cols = []
        for c, v in zip(cat_cols, row):
            v = "NA" if pd.isna(v) else str(v)
            token = f"{c}={v}" if add_column_prefix else v
            cols.append(_hash_token(token, n_buckets))
        # de-dup within row to avoid repeated indices
        cols = sorted(set(cols))
        indices_list.extend(cols)
        data_list.extend([1.0] * len(cols))
        indptr[i + 1] = indptr[i] + len(cols)

    indices = np.array(indices_list, dtype=np.int32)
    data = np.array(data_list, dtype=np.float32)
    return sp.csr_matrix((data, indices, indptr), shape=(n, n_buckets), dtype=np.float32)


def build_numeric_matrix(df: pd.DataFrame, num_cols, fill: float = 0.0) -> np.ndarray:
    Xn = df[num_cols].copy()

    # 1) to numeric + fill missing
    Xn = Xn.apply(pd.to_numeric, errors="coerce")
    Xn = Xn.fillna(fill)

    # 2) robust guards: clip negatives, cap huge values to avoid overflow
    # (Criteo numeric should be >=0, but some pipelines / oddities may create -1)
    X = Xn.to_numpy(dtype=np.float64, copy=False)
    X = np.maximum(X, 0.0)          # remove negatives -> avoid log1p(-1) = -inf
    X = np.minimum(X, 1e12)         # cap extreme outliers

    # 3) stabilize scale
    X = np.log1p(X)

    # 4) ensure finite then cast
    X[~np.isfinite(X)] = 0.0
    return X.astype(np.float32, copy=False)


def combine_features(
    X_num: np.ndarray,
    X_hash: sp.csr_matrix,
    X_freq: Optional[np.ndarray] = None,
) -> sp.csr_matrix:
    mats = [sp.csr_matrix(X_num)]
    if X_freq is not None:
        mats.append(sp.csr_matrix(X_freq))
    mats.append(X_hash)
    return sp.hstack(mats, format="csr")


def iter_hashed_rows(
    df: pd.DataFrame,
    num_cols: Sequence[str],
    cat_cols: Sequence[str],
    n_buckets: int,
    numeric_fill: float = 0.0,
    freq_encoder: Optional[FrequencyEncoder] = None,
) -> Iterable[Tuple[np.ndarray, List[int], np.ndarray]]:
    """
    Yield per-row representation for streaming models (FTRL).

    Returns:
      - numeric array (len = len(num_cols))
      - list of hashed indices (categoricals)
      - optional frequency features array
    """
    # frequency features (vectorized once per chunk)
    freq_feats = None
    if freq_encoder is not None:
        freq_feats = freq_encoder.transform(df, cat_cols)

    col_idx = {c: i for i, c in enumerate(df.columns)}
    num_idx = [col_idx[c] for c in num_cols]
    cat_idx = [col_idx[c] for c in cat_cols]

    for i, row in enumerate(df.itertuples(index=False, name=None)):
        # numeric
        x_num = np.empty(len(num_idx), dtype=np.float32)
        for j, k in enumerate(num_idx):
            v = row[k]
            if pd.isna(v):
                x_num[j] = numeric_fill
            else:
                try:
                    x_num[j] = float(v)
                except Exception:
                    x_num[j] = numeric_fill

        # categorical -> hashed indices
        cols = []
        for c, k in zip(cat_cols, cat_idx):
            v = row[k]
            v = "NA" if pd.isna(v) or v == "" else str(v)
            token = f"{c}={v}"
            cols.append(_hash_token(token, n_buckets))
        cols = sorted(set(cols))

        # frequency features
        if freq_feats is None:
            x_freq = np.array([], dtype=np.float32)
        else:
            x_freq = freq_feats[i].astype(np.float32)

        yield x_num, cols, x_freq