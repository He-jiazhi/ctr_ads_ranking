from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import numpy as np
import scipy.sparse as sp
from joblib import dump, load
from sklearn.linear_model import SGDClassifier


@dataclass
class LRConfig:
    alpha: float = 1e-5
    l1_ratio: float = 0.0
    max_iter: int = 1  # for partial_fit loop
    random_state: int = 42


class StreamingLogisticRegression:
    """
    Logistic regression via SGDClassifier with partial_fit (streaming-friendly).
    """
    def __init__(self, config: LRConfig):
        self.config = config
        self.model = SGDClassifier(
            loss="log_loss",
            penalty="elasticnet" if config.l1_ratio > 0 else "l2",
            alpha=config.alpha,
            l1_ratio=config.l1_ratio,
            max_iter=config.max_iter,
            learning_rate="optimal",
            random_state=config.random_state,
        )
        self._is_fit = False

    def partial_fit(self, X: sp.csr_matrix, y: np.ndarray):
        if not self._is_fit:
            self.model.partial_fit(X, y, classes=np.array([0, 1]))
            self._is_fit = True
        else:
            self.model.partial_fit(X, y)
        return self

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        # SGDClassifier supports predict_proba for log_loss
        return self.model.predict_proba(X)

    def save(self, path: str):
        dump(self.model, path)

    def load(self, path: str):
        self.model = load(path)
        self._is_fit = True
        return self


@dataclass
class FTRLConfig:
    alpha: float = 0.05
    beta: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    # feature dims: numeric + freq + hashed
    n_features: int = 2 ** 20 + 13 + 26  # will be overwritten


class FTRLProximal:
    """
    FTRL-Proximal for logistic regression (Google ad click paper style).

    Supports streaming updates on sparse features.

    Feature vector is represented as:
      - dense numeric features (np.ndarray)
      - dense freq features (np.ndarray, optional)
      - sparse hashed categorical indices (List[int])

    We map these into a single index space:
      [0..d_num-1] numeric
      [d_num..d_num+d_freq-1] freq
      [offset_hash..offset_hash+n_hash-1] hashed
    """
    def __init__(self, config: FTRLConfig, d_num: int, d_freq: int, n_hash: int):
        self.config = config
        self.d_num = d_num
        self.d_freq = d_freq
        self.n_hash = n_hash
        self.offset_hash = d_num + d_freq
        self.n_features = self.offset_hash + n_hash

        # z and n vectors
        self.z = np.zeros(self.n_features, dtype=np.float32)
        self.n = np.zeros(self.n_features, dtype=np.float32)

    def _indices_and_values(self, x_num: np.ndarray, x_freq: np.ndarray, hash_idx: List[int]):
        idx = []
        val = []
        # numeric
        for j, v in enumerate(x_num):
            if v != 0:
                idx.append(j)
                val.append(float(v))
        # freq
        for j, v in enumerate(x_freq):
            if v != 0:
                idx.append(self.d_num + j)
                val.append(float(v))
        # hashed one-hot
        for h in hash_idx:
            idx.append(self.offset_hash + h)
            val.append(1.0)
        return idx, val

    def _weights(self, indices: List[int]) -> np.ndarray:
        w = np.zeros(len(indices), dtype=np.float32)
        for i, j in enumerate(indices):
            z = self.z[j]
            if abs(z) <= self.config.l1:
                w[i] = 0.0
            else:
                sign = -1.0 if z < 0 else 1.0
                w[i] = (sign * self.config.l1 - z) / (
                    (self.config.beta + np.sqrt(self.n[j])) / self.config.alpha + self.config.l2
                )
        return w

    @staticmethod
    def _sigmoid(x: float) -> float:
        # numerically stable sigmoid
        if x >= 0:
            z = np.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = np.exp(x)
            return z / (1.0 + z)

    def predict_proba_row(self, x_num: np.ndarray, x_freq: np.ndarray, hash_idx: List[int]) -> float:
        indices, values = self._indices_and_values(x_num, x_freq, hash_idx)
        w = self._weights(indices)
        wx = float(np.dot(w, np.array(values, dtype=np.float32)))
        return self._sigmoid(wx)

    def update_row(self, x_num: np.ndarray, x_freq: np.ndarray, hash_idx: List[int], y: int):
        indices, values = self._indices_and_values(x_num, x_freq, hash_idx)
        w = self._weights(indices)
        wx = float(np.dot(w, np.array(values, dtype=np.float32)))
        p = self._sigmoid(wx)
        g = p - float(y)  # gradient for log loss
        for j, xj, wj in zip(indices, values, w):
            gj = g * xj
            sigma = (np.sqrt(self.n[j] + gj * gj) - np.sqrt(self.n[j])) / self.config.alpha
            self.z[j] += gj - sigma * wj
            self.n[j] += gj * gj

    def predict_proba(self, rows: Iterable[Tuple[np.ndarray, List[int], np.ndarray]]) -> np.ndarray:
        preds = []
        for x_num, hash_idx, x_freq in rows:
            preds.append(self.predict_proba_row(x_num, x_freq, hash_idx))
        return np.array(preds, dtype=np.float32)

    def fit(self, rows: Iterable[Tuple[np.ndarray, List[int], np.ndarray]], y: np.ndarray):
        for (x_num, hash_idx, x_freq), yi in zip(rows, y):
            self.update_row(x_num, x_freq, hash_idx, int(yi))
        return self

    def save(self, path: str):
        dump(
            {
                "config": self.config.__dict__,
                "d_num": self.d_num,
                "d_freq": self.d_freq,
                "n_hash": self.n_hash,
                "z": self.z,
                "n": self.n,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "FTRLProximal":
        obj = load(path)
        cfg = FTRLConfig(**obj["config"])
        model = cls(cfg, obj["d_num"], obj["d_freq"], obj["n_hash"])
        model.z = obj["z"]
        model.n = obj["n"]
        return model


class LightGBMWrapper:
    """
    Optional LightGBM model.
    Only used if lightgbm is installed.
    """
    def __init__(self, **params):
        try:
            import lightgbm as lgb  # noqa
        except Exception as e:
            raise ImportError("lightgbm is not installed. Run: pip install lightgbm") from e
        import lightgbm as lgb
        self.lgb = lgb
        self.params = params or {}
        self.model = None

    def fit(self, X: sp.csr_matrix, y: np.ndarray, X_val: Optional[sp.csr_matrix] = None, y_val: Optional[np.ndarray] = None):
        params = {
            "objective": "binary",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "metric": ["auc", "binary_logloss"],
        }
        params.update(self.params)
        dtrain = self.lgb.Dataset(X, label=y)
        valid_sets = [dtrain]
        valid_names = ["train"]
        if X_val is not None and y_val is not None:
            dval = self.lgb.Dataset(X_val, label=y_val, reference=dtrain)
            valid_sets.append(dval)
            valid_names.append("val")
        self.model = self.lgb.train(params, dtrain, num_boost_round=200, valid_sets=valid_sets, valid_names=valid_names, verbose_eval=50)
        return self

    def predict_proba(self, X: sp.csr_matrix) -> np.ndarray:
        p = self.model.predict(X)
        return np.vstack([1 - p, p]).T

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model = self.lgb.Booster(model_file=path)
        return self