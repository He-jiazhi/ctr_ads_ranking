from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class SimConfig:
    budget: int = 500_000
    seed: int = 42


def simulate_delivery(
    df: pd.DataFrame,
    p_col: str = "p_ctr",
    label_col: str = "label",
    bid_col: Optional[str] = None,
    budget: int = 500_000,
    seed: int = 42,
) -> dict:
    """
    Very simple offline delivery simulation.

    Assumptions:
    - Each row is an eligible impression
    - We select top 'budget' impressions by expected value: pCTR * bid
    - If bid_col is None, we create a synthetic bid based on a numeric feature-like distribution

    Outputs:
    - expected clicks, realized clicks (from labels), expected revenue, avg pCTR, etc.
    """
    rng = np.random.default_rng(seed)
    tmp = df.copy()

    if bid_col is None:
        # synthetic bid (log-normal) in arbitrary units
        tmp["bid"] = rng.lognormal(mean=0.0, sigma=0.5, size=len(tmp)).astype(np.float32)
        bid_col = "bid"

    tmp["score"] = tmp[p_col].astype(np.float32) * tmp[bid_col].astype(np.float32)

    picked = tmp.nlargest(budget, "score")
    expected_clicks = float(picked[p_col].sum())
    realized_clicks = float(picked[label_col].sum())
    expected_revenue = float((picked[p_col] * picked[bid_col]).sum())
    avg_pctr = float(picked[p_col].mean())
    return {
        "budget": int(budget),
        "expected_clicks": expected_clicks,
        "realized_clicks": realized_clicks,
        "expected_revenue": expected_revenue,
        "avg_pctr_selected": avg_pctr,
    }