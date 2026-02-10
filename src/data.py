from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm


CRITEO_NUM_COLS = [f"I{i}" for i in range(1, 14)]
CRITEO_CAT_COLS = [f"C{i}" for i in range(1, 27)]
CRITEO_COLS = ["label"] + CRITEO_NUM_COLS + CRITEO_CAT_COLS


@dataclass
class SplitConfig:
    """Time-based split by log order (row order)."""
    train_frac: float = 0.8
    val_frac: float = 0.1
    test_frac: float = 0.1

    def __post_init__(self):
        assert abs(self.train_frac + self.val_frac + self.test_frac - 1.0) < 1e-9


def iter_criteo_chunks(
    path: str,
    chunksize: int = 1_000_000,
    max_rows: Optional[int] = None,
    has_header: bool = False,
) -> Iterator[pd.DataFrame]:
    """
    Stream Criteo TSV as chunks.

    The Kaggle Criteo train file typically has no header.
    """
    read_rows = 0
    for chunk in pd.read_csv(
        path,
        sep="\t",
        header=0 if has_header else None,
        names=CRITEO_COLS if not has_header else None,
        chunksize=chunksize,
        na_values="",
        low_memory=False,
    ):
        if max_rows is not None:
            remaining = max_rows - read_rows
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining].copy()
        read_rows += len(chunk)
        yield chunk


def split_by_order(
    n_rows: int,
    split: SplitConfig,
) -> Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]:
    """
    Return index ranges [start, end) for train/val/test.
    """
    n_train = int(n_rows * split.train_frac)
    n_val = int(n_rows * split.val_frac)
    n_test = n_rows - n_train - n_val
    train = (0, n_train)
    val = (n_train, n_train + n_val)
    test = (n_train + n_val, n_train + n_val + n_test)
    return train, val, test


def load_small_dataframe(
    path: str,
    max_rows: int,
    has_header: bool = False,
) -> pd.DataFrame:
    """Convenience loader for smaller runs."""
    df = next(iter_criteo_chunks(path, chunksize=max_rows, max_rows=max_rows, has_header=has_header))
    return df


def add_row_index(df: pd.DataFrame, start_index: int) -> pd.DataFrame:
    df = df.copy()
    df["_row_id"] = range(start_index, start_index + len(df))
    return df


def compute_total_rows(path: str, has_header: bool = False) -> int:
    """
    Count rows quickly by iterating through file. This is O(file) but robust.
    For very large files, you can skip and use max_rows to limit.
    """
    n = 0
    for chunk in iter_criteo_chunks(path, chunksize=2_000_000, max_rows=None, has_header=has_header):
        n += len(chunk)
    return n