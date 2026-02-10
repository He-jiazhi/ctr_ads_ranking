from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp
from tqdm import tqdm

from .data import CRITEO_CAT_COLS, CRITEO_NUM_COLS, SplitConfig, iter_criteo_chunks, split_by_order
from .eval import eval_binary, group_ndcg, slice_metrics
from .features import FeatureConfig, FrequencyEncoder, build_hashed_csr, build_numeric_matrix, combine_features, iter_hashed_rows
from .models import FTRLConfig, FTRLProximal, LRConfig, StreamingLogisticRegression, LightGBMWrapper
from .simulate import simulate_delivery


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def cmd_train(args):
    out = Path(args.out_dir)
    _ensure_dir(out)
    (out / "meta").mkdir(exist_ok=True)

    # We stream chunks and do a time-based split by row order.
    # If max_rows is set, split happens within those max_rows.
    max_rows = args.max_rows
    split = SplitConfig()

    # Determine total rows (within max_rows) approximately:
    # We'll count as we stream
    total = 0
    for chunk in iter_criteo_chunks(args.data_path, chunksize=args.chunksize, max_rows=max_rows, has_header=args.has_header):
        total += len(chunk)

    train_range, val_range, test_range = split_by_order(total, split)

    # Fit frequency encoder on TRAIN only (requires holding some train data).
    # For large runs, we estimate frequencies from a sample or from a first pass over train.
    freq_enc = FrequencyEncoder(min_freq=args.min_freq) if args.use_frequency_encoding else None

    # First pass: collect a sample of train for freq encoding (bounded memory)
    if freq_enc is not None:
        sample_rows = min(args.freq_fit_rows, train_range[1])
        df_sample = []
        read = 0
        for chunk in iter_criteo_chunks(args.data_path, chunksize=args.chunksize, max_rows=max_rows, has_header=args.has_header):
            if read >= sample_rows:
                break
            take = min(len(chunk), sample_rows - read)
            df_sample.append(chunk.iloc[:take])
            read += take
        df_sample = pd.concat(df_sample, axis=0)
        freq_enc.fit(df_sample, CRITEO_CAT_COLS)
        # save freq tables
        (out / "meta" / "freq_tables.json").write_text(json.dumps(freq_enc.tables)[:5_000_000], encoding="utf-8")

    fcfg = FeatureConfig(n_hash_buckets=args.n_hash_buckets, use_frequency_encoding=args.use_frequency_encoding, min_freq=args.min_freq)

    # Models
    models = {}
    if "lr" in args.models:
        models["lr"] = StreamingLogisticRegression(LRConfig(alpha=args.lr_alpha, l1_ratio=args.lr_l1_ratio))
    if "ftrl" in args.models:
        d_num = len(CRITEO_NUM_COLS)
        d_freq = len(CRITEO_CAT_COLS) if args.use_frequency_encoding else 0
        models["ftrl"] = FTRLProximal(
            FTRLConfig(alpha=args.ftrl_alpha, beta=args.ftrl_beta, l1=args.ftrl_l1, l2=args.ftrl_l2),
            d_num=d_num,
            d_freq=d_freq,
            n_hash=args.n_hash_buckets,
        )
    if "lgbm" in args.models:
        models["lgbm"] = LightGBMWrapper()

    # Second pass: stream and train
    cur = 0
    for chunk in tqdm(iter_criteo_chunks(args.data_path, chunksize=args.chunksize, max_rows=max_rows, has_header=args.has_header), desc="train_stream"):
        n = len(chunk)
        start = cur
        end = cur + n
        cur = end

        # Determine which split this chunk overlaps; we'll train only on train range
        if end <= train_range[0] or start >= train_range[1]:
            continue

        # slice to train portion
        s0 = max(0, train_range[0] - start)
        s1 = min(n, train_range[1] - start)
        df = chunk.iloc[s0:s1].copy()

        y = df["label"].astype("int8").to_numpy()

        X_num = build_numeric_matrix(df, CRITEO_NUM_COLS, fill=args.numeric_fill)
        X_hash = build_hashed_csr(df, CRITEO_CAT_COLS, args.n_hash_buckets)
        X_freq = freq_enc.transform(df, CRITEO_CAT_COLS) if freq_enc is not None else None
        X = combine_features(X_num, X_hash, X_freq)

        if "lr" in models:
            models["lr"].partial_fit(X, y)

        if "ftrl" in models:
            rows = iter_hashed_rows(df, CRITEO_NUM_COLS, CRITEO_CAT_COLS, args.n_hash_buckets, numeric_fill=args.numeric_fill, freq_encoder=freq_enc)
            models["ftrl"].fit(rows, y)

        # LightGBM typically needs full data; train on a bounded sample for demonstration
        if "lgbm" in models and args.lgbm_fit:
            # only collect some rows, then fit once later (handled in separate small-data path)
            pass

    # Save models
    if "lr" in models:
        models["lr"].save(str(out / "lr.joblib"))
    if "ftrl" in models:
        models["ftrl"].save(str(out / "ftrl.joblib"))

    meta = {
        "data_path": args.data_path,
        "max_rows": max_rows,
        "total_rows_used": total,
        "split": {"train": train_range, "val": val_range, "test": test_range},
        "feature": {"n_hash_buckets": args.n_hash_buckets, "use_frequency_encoding": args.use_frequency_encoding, "min_freq": args.min_freq},
    }
    (out / "meta" / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved run to: {out}")


def _load_run_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "meta" / "run_meta.json").read_text(encoding="utf-8"))


def _load_freq_tables(run_dir: Path) -> Optional[dict]:
    p = run_dir / "meta" / "freq_tables.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _make_freq_encoder_from_tables(tables: dict) -> FrequencyEncoder:
    fe = FrequencyEncoder(min_freq=1)
    fe.tables = tables
    fe.default = {k: 0.0 for k in tables.keys()}
    return fe


def _predict_on_split(args, run_dir: Path, split_name: str) -> pd.DataFrame:
    meta = _load_run_meta(run_dir)
    total = meta["total_rows_used"]
    train_range, val_range, test_range = meta["split"]["train"], meta["split"]["val"], meta["split"]["test"]
    target = {"train": train_range, "val": val_range, "test": test_range}[split_name]

    freq_tables = _load_freq_tables(run_dir)
    freq_enc = _make_freq_encoder_from_tables(freq_tables) if freq_tables is not None else None

    # Load models
    lr = None
    ftrl = None
    if (run_dir / "lr.joblib").exists():
        lr = StreamingLogisticRegression(LRConfig()).load(str(run_dir / "lr.joblib"))
    if (run_dir / "ftrl.joblib").exists():
        ftrl = FTRLProximal.load(str(run_dir / "ftrl.joblib"))

    rows_out = []
    cur = 0
    for chunk in tqdm(iter_criteo_chunks(meta["data_path"], chunksize=args.chunksize, max_rows=meta["max_rows"], has_header=args.has_header), desc=f"predict_{split_name}"):
        n = len(chunk)
        start = cur
        end = cur + n
        cur = end
        if end <= target[0] or start >= target[1]:
            continue
        s0 = max(0, target[0] - start)
        s1 = min(n, target[1] - start)
        df = chunk.iloc[s0:s1].copy()

        y = df["label"].astype("int8").to_numpy()
        out_df = pd.DataFrame({"label": y})

        # create pseudo group id for ranking: (C1 + time bucket)
        out_df["group_id"] = df["C1"].astype("string").fillna("NA") + "_" + ((np.arange(len(df)) + start) // 50).astype(str)

        X_num = build_numeric_matrix(df, CRITEO_NUM_COLS, fill=args.numeric_fill)
        X_hash = build_hashed_csr(df, CRITEO_CAT_COLS, meta["feature"]["n_hash_buckets"])
        X_freq = freq_enc.transform(df, CRITEO_CAT_COLS) if freq_enc is not None else None
        X = combine_features(X_num, X_hash, X_freq)

        if lr is not None:
            p = lr.predict_proba(X)[:, 1]
            out_df["p_lr"] = p

        if ftrl is not None:
            rows = iter_hashed_rows(df, CRITEO_NUM_COLS, CRITEO_CAT_COLS, meta["feature"]["n_hash_buckets"], numeric_fill=args.numeric_fill, freq_encoder=freq_enc)
            p = ftrl.predict_proba(rows)
            out_df["p_ftrl"] = p

        # carry some categorical columns for slicing
        for c in ["C1", "C2", "C3"]:
            out_df[c] = df[c].astype("string").fillna("NA").to_numpy()

        rows_out.append(out_df)

    pred = pd.concat(rows_out, axis=0).reset_index(drop=True)
    return pred


def cmd_evaluate(args):
    run_dir = Path(args.run_dir)
    out = run_dir / "metrics"
    _ensure_dir(out)

    for split in ["val", "test"]:
        pred = _predict_on_split(args, run_dir, split)
        y = pred["label"].to_numpy()

        for col in [c for c in pred.columns if c.startswith("p_")]:
            r = eval_binary(y, pred[col].to_numpy())
            print(f"[{split}] {col}: AUC={r.auc:.4f} LogLoss={r.logloss:.4f} Brier={r.brier:.4f}")
            (out / f"{split}_{col}_metrics.json").write_text(json.dumps(r.__dict__, indent=2), encoding="utf-8")

        pred.to_parquet(out / f"{split}_pred.parquet", index=False)
    print(f"Saved metrics to: {out}")


def cmd_ranking(args):
    run_dir = Path(args.run_dir)
    pred = pd.read_parquet(run_dir / "metrics" / "test_pred.parquet").reset_index(drop=True)

    # sanity
    if "label" not in pred.columns:
        raise ValueError(f"'label' not found in {pred.columns.tolist()}")
    pred["label"] = pred["label"].astype(np.float32)

    group_size = getattr(args, "group_size", 50)  # 默认 50，跟你 direct 一致
    pred["_grp"] = (np.arange(len(pred)) // group_size).astype("int64")

    score_cols = [c for c in pred.columns if c.startswith("p_")]
    if not score_cols:
        raise ValueError("No prediction columns starting with 'p_' found.")

    for col in score_cols:
        v = group_ndcg(
            pred,
            y_col="label",
            score_col=col,
            group_col="_grp",
            k=args.k,
            max_groups=None,
        )
        print(f"NDCG@{args.k} ({col}) = {v:.4f}")


def cmd_slice(args):
    run_dir = Path(args.run_dir)
    pred = pd.read_parquet(run_dir / "metrics" / "test_pred.parquet")

    score_cols = [c for c in pred.columns if c.startswith("p_")]
    if not score_cols:
        print("No prediction columns found (expected columns starting with 'p_').")
        return

    for col in score_cols:
        for slice_col in ["C1", "C2", "C3"]:
            dfm = slice_metrics(
                pred,
                y_col="label",
                score_col=col,
                slice_col=slice_col,
                topk_values=args.topk_values,
            )
            print(f"\nSlice: {slice_col} using {col}")
            if dfm is None or getattr(dfm, "empty", True):
                print("(no slice results; try smaller --topk_values or lower min_rows)")
            else:
                print(dfm.head(10).to_string(index=False))


def cmd_simulate(args):
    run_dir = Path(args.run_dir)
    pred = pd.read_parquet(run_dir / "metrics" / "test_pred.parquet")
    # choose a model probability column
    pcol = args.pcol
    if pcol is None:
        pcols = [c for c in pred.columns if c.startswith("p_")]
        pcol = pcols[0]
    pred = pred.rename(columns={pcol: "p_ctr"})
    res = simulate_delivery(pred, p_col="p_ctr", label_col="label", budget=args.budget)
    print(json.dumps(res, indent=2))


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # common
    def add_common(sp):
        sp.add_argument("--chunksize", type=int, default=500_000)
        sp.add_argument("--has_header", action="store_true")
        sp.add_argument("--numeric_fill", type=float, default=0.0)

    # train
    t = sub.add_parser("train")
    add_common(t)
    t.add_argument("--data_path", type=str, required=True)
    t.add_argument("--out_dir", type=str, required=True)
    t.add_argument("--max_rows", type=int, default=None)
    t.add_argument("--models", nargs="+", default=["lr", "ftrl"], choices=["lr", "ftrl", "lgbm"])
    t.add_argument("--n_hash_buckets", type=int, default=2**20)
    t.add_argument("--use_frequency_encoding", action="store_true")
    t.add_argument("--min_freq", type=int, default=2)
    t.add_argument("--freq_fit_rows", type=int, default=2_000_000)
    # lr params
    t.add_argument("--lr_alpha", type=float, default=1e-5)
    t.add_argument("--lr_l1_ratio", type=float, default=0.0)
    # ftrl params
    t.add_argument("--ftrl_alpha", type=float, default=0.05)
    t.add_argument("--ftrl_beta", type=float, default=1.0)
    t.add_argument("--ftrl_l1", type=float, default=1.0)
    t.add_argument("--ftrl_l2", type=float, default=1.0)
    # lgbm
    t.add_argument("--lgbm_fit", action="store_true", help="(optional) train LightGBM on sampled data")
    t.set_defaults(func=cmd_train)

    # evaluate
    e = sub.add_parser("evaluate")
    add_common(e)
    e.add_argument("--run_dir", type=str, required=True)
    e.set_defaults(func=cmd_evaluate)

    # ranking
    r = sub.add_parser("ranking")
    r.add_argument("--run_dir", type=str, required=True)
    r.add_argument("--k", type=int, default=10)
    r.set_defaults(func=cmd_ranking)

    # slice
    s = sub.add_parser("slice")
    s.add_argument("--run_dir", type=str, required=True)
    s.add_argument("--topk_values", type=int, default=10)
    s.set_defaults(func=cmd_slice)

    # simulate
    sim = sub.add_parser("simulate")
    sim.add_argument("--run_dir", type=str, required=True)
    sim.add_argument("--budget", type=int, default=500_000)
    sim.add_argument("--pcol", type=str, default=None)
    sim.set_defaults(func=cmd_simulate)

    return p


def main():
    p = build_parser()
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()