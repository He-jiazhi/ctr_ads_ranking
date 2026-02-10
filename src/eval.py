from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


@dataclass
class EvalResults:
    auc: float
    logloss: float
    brier: float


def eval_binary(y_true: np.ndarray, p_pred: np.ndarray) -> EvalResults:
    auc = float(roc_auc_score(y_true, p_pred))
    import numpy as np
    from sklearn.metrics import log_loss

    p = np.clip(p_pred, 1e-15, 1 - 1e-15)
    ll = float(log_loss(y_true, p))
    brier = float(brier_score_loss(y_true, p_pred))
    return EvalResults(auc=auc, logloss=ll, brier=brier)


def platt_calibration(base_model, X_train, y_train, X_val, y_val):
    """
    Platt scaling (sigmoid) using CalibratedClassifierCV.
    Works for sklearn-like estimators.
    """
    cal = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    cal.fit(X_val, y_val)
    return cal


def isotonic_calibration(base_model, X_train, y_train, X_val, y_val):
    cal = CalibratedClassifierCV(base_model, method="isotonic", cv="prefit")
    cal.fit(X_val, y_val)
    return cal


def ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    """NDCG@k for a single group."""
    order = np.argsort(-y_score)[:k]
    gains = y_true[order]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float(np.sum(gains * discounts))

    ideal = np.sort(y_true)[::-1][:k]
    idcg = float(np.sum(ideal * discounts))

    return (dcg / idcg) if idcg > 0 else np.nan


def group_ndcg(
    df: pd.DataFrame,
    y_col: str,
    score_col: str,
    group_col: str,
    k: int = 10,
    max_groups: Optional[int] = 20000,
) -> float:
    """Compute mean NDCG@K across groups (skip groups with idcg=0)."""
    ndcgs = []
    for i, (_, g) in enumerate(df.groupby(group_col)):
        if max_groups is not None and i >= max_groups:
            break
        y = g[y_col].to_numpy(dtype=np.float32)
        s = g[score_col].to_numpy(dtype=np.float32)
        ndcgs.append(ndcg_at_k(y, s, k))

   
    ndcgs = [x for x in ndcgs if not np.isnan(x)]
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def slice_metrics(
    df: pd.DataFrame,
    y_col: str,
    score_col: str,
    slice_col: str,
    topk_values: int = 10,
    min_rows: int = 1000,
) -> pd.DataFrame:
    """Compute AUC / LogLoss per slice value for top frequent values."""
    tmp = df.copy()
    tmp[slice_col] = tmp[slice_col].astype("string").fillna("NA")
    vc = tmp[slice_col].value_counts()
    keep = list(vc.head(topk_values).index)

    rows = []
    for val in keep:
        g = tmp[tmp[slice_col] == val]
        if len(g) < min_rows:
            continue
        y = g[y_col].to_numpy()
        p = g[score_col].to_numpy()
        r = eval_binary(y, p)
        rows.append({
            "slice": slice_col,
            "value": val,
            "n": len(g),
            "auc": r.auc,
            "logloss": r.logloss,
            "brier": r.brier,
        })

    
    dfm = pd.DataFrame(rows)
    if dfm.empty:
        return pd.DataFrame(columns=["slice", "value", "n", "auc", "logloss", "brier"])
    return dfm.sort_values(["auc"], ascending=False).reset_index(drop=True)