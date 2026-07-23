"""
latent.model — the stacked ensemble.

XGBoost and LightGBM each vote on whether the next five bars close higher.
A logistic regression sitting on top learns when to trust which. The
meta-learner is deliberately trained on *held-out* base predictions: if you
train it on predictions the base models made about their own training data,
it learns their overfitting rather than their signal.

An AUC of 0.60 on financial data is a strong result. 0.50 is a coin flip.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


class StackedEnsemble:
    """
    Fit with .fit(X, y), predict with .predict_proba(X).

    The split inside fit() is chronological, never random. Shuffling time
    series before splitting leaks the future into the past and is the most
    common way a backtest ends up meaningless.
    """

    def __init__(self, seed: int = 42, val_frac: float = 0.25):
        self.seed = seed
        self.val_frac = val_frac
        self.features: list[str] = []
        self.auc_xgb = self.auc_lgb = self.auc_meta = float("nan")

    # ── internals ────────────────────────────────────────────────────
    def _make_base(self):
        from lightgbm import LGBMClassifier
        from xgboost import XGBClassifier

        xgb = XGBClassifier(
            n_estimators=600, max_depth=5, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.75,
            min_child_weight=5, gamma=0.1,
            reg_alpha=0.1, reg_lambda=1.0,
            eval_metric="logloss", random_state=self.seed,
            n_jobs=-1, verbosity=0,
        )
        lgb = LGBMClassifier(
            n_estimators=600, num_leaves=40, learning_rate=0.03,
            subsample=0.75, colsample_bytree=0.75,
            min_child_samples=25, reg_alpha=0.1, reg_lambda=1.0,
            random_state=self.seed, n_jobs=-1, verbose=-1,
        )
        return xgb, lgb

    # ── public ───────────────────────────────────────────────────────
    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> "StackedEnsemble":
        self.features = list(X.columns)
        Xv = X.to_numpy(dtype=float)
        yv = np.asarray(y, dtype=int)

        cut = int(len(Xv) * (1 - self.val_frac))
        if cut < 50 or len(Xv) - cut < 30:
            raise ValueError(f"Not enough rows to train and validate ({len(Xv)}).")

        Xtr, Xva = Xv[:cut], Xv[cut:]
        ytr, yva = yv[:cut], yv[cut:]

        if len(np.unique(ytr)) < 2:
            raise ValueError("Training slice contains only one class.")

        self.xgb, self.lgb = self._make_base()
        self.xgb.fit(Xtr, ytr)
        self.lgb.fit(Xtr, ytr)

        # base predictions on data neither model has seen
        p_xgb = self.xgb.predict_proba(Xva)[:, 1]
        p_lgb = self.lgb.predict_proba(Xva)[:, 1]
        meta_X = np.column_stack([p_xgb, p_lgb])

        self.scaler = StandardScaler().fit(meta_X)
        self.meta = LogisticRegression(max_iter=1000, C=1.0)
        self.meta.fit(self.scaler.transform(meta_X), yva)

        if len(np.unique(yva)) > 1:
            p_meta = self.meta.predict_proba(self.scaler.transform(meta_X))[:, 1]
            self.auc_xgb = roc_auc_score(yva, p_xgb)
            self.auc_lgb = roc_auc_score(yva, p_lgb)
            self.auc_meta = roc_auc_score(yva, p_meta)
            if verbose:
                print(f"  AUC — xgb {self.auc_xgb:.3f} | lgb {self.auc_lgb:.3f} "
                      f"| stacked {self.auc_meta:.3f}")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated probability that the next five bars close higher."""
        Xv = X[self.features].to_numpy(dtype=float)
        meta_X = np.column_stack([
            self.xgb.predict_proba(Xv)[:, 1],
            self.lgb.predict_proba(Xv)[:, 1],
        ])
        return self.meta.predict_proba(self.scaler.transform(meta_X))[:, 1]

    def importances(self) -> pd.DataFrame:
        """Average gain-based importance across both base models."""
        xi = self.xgb.feature_importances_
        li = self.lgb.feature_importances_
        xi = xi / (xi.sum() or 1)
        li = li / (li.sum() or 1)
        return (pd.DataFrame({"feature": self.features, "importance": (xi + li) / 2})
                .sort_values("importance", ascending=False)
                .reset_index(drop=True))


def safe_features(df: pd.DataFrame, wanted: list[str]) -> list[str]:
    """
    Keep only the columns that actually exist and carry some variation.

    This is what lets the pipeline degrade instead of crashing when the HMM
    or the Hurst calculation didn't produce anything usable.
    """
    out = []
    for c in wanted:
        if c not in df.columns:
            continue
        col = df[c]
        if col.notna().sum() < len(df) * 0.5:
            continue
        if col.nunique(dropna=True) <= 1:
            continue
        out.append(c)
    return out
