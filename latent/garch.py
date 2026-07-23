"""
latent.garch — GARCH(1,1), fitted by maximum likelihood with L-BFGS-B.

Why bother when a rolling standard deviation exists: volatility clusters.
Quiet hours follow quiet hours and violent ones follow violent ones. A
rolling window tells you what volatility *was*. GARCH gives you a forecast
of what it is about to be, which is the number you actually want when
deciding how large a position to take.

    sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class GARCH11:
    """
    Plain GARCH(1,1) on demeaned returns.

    Attributes after fit(): omega, alpha, beta, converged, persistence.
    Persistence (alpha + beta) close to 1 means shocks decay slowly, which
    is normal and expected for crypto.
    """

    def __init__(self):
        self.omega = self.alpha = self.beta = None
        self.converged = False

    # ── likelihood ───────────────────────────────────────────────────
    @staticmethod
    def _neg_loglik(params: np.ndarray, e: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
            return 1e10

        n = len(e)
        var = np.empty(n)
        var[0] = e.var() if e.var() > 0 else 1e-8

        for t in range(1, n):
            var[t] = omega + alpha * e[t - 1] ** 2 + beta * var[t - 1]
            if var[t] <= 0 or not np.isfinite(var[t]):
                return 1e10

        ll = -0.5 * np.sum(np.log(2 * np.pi) + np.log(var) + e ** 2 / var)
        return -ll if np.isfinite(ll) else 1e10

    # ── public ───────────────────────────────────────────────────────
    def fit(self, returns: np.ndarray | pd.Series) -> "GARCH11":
        e = np.asarray(returns, dtype=float)
        e = e[np.isfinite(e)]
        if len(e) < 100:
            raise ValueError("Need at least 100 finite returns to fit GARCH.")
        e = e - e.mean()

        var0 = e.var()
        start = np.array([var0 * 0.05, 0.10, 0.85])

        res = minimize(
            self._neg_loglik, start, args=(e,),
            method="L-BFGS-B",
            bounds=[(1e-12, var0 * 10), (1e-6, 0.5), (1e-6, 0.995)],
            options={"maxiter": 500},
        )

        if res.success and np.isfinite(res.fun):
            self.omega, self.alpha, self.beta = res.x
            self.converged = True
        else:
            # a sane fallback rather than an exception — the pipeline
            # should degrade, not die
            self.omega, self.alpha, self.beta = var0 * 0.05, 0.10, 0.85
            self.converged = False
        return self

    @property
    def persistence(self) -> float:
        return float(self.alpha + self.beta)

    @property
    def long_run_var(self) -> float:
        return float(self.omega / max(1 - self.persistence, 1e-9))

    def conditional_vol(self, returns: np.ndarray | pd.Series) -> np.ndarray:
        """One-step-ahead conditional volatility (sigma, not variance)."""
        e = np.asarray(returns, dtype=float)
        e = np.nan_to_num(e, nan=0.0)
        e = e - np.mean(e[np.isfinite(e)]) if len(e) else e

        n = len(e)
        var = np.empty(n)
        var[0] = max(np.var(e), 1e-12)
        for t in range(1, n):
            var[t] = self.omega + self.alpha * e[t - 1] ** 2 + self.beta * var[t - 1]
            if not np.isfinite(var[t]) or var[t] <= 0:
                var[t] = var[t - 1]
        return np.sqrt(var)


def add_garch_vol(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Attach `garch_vol` and `vol_regime`.

    vol_regime is the ratio of forecast volatility to its own long-run
    average: above 1 means conditions are hotter than usual.
    """
    d = df.copy()
    try:
        model = GARCH11().fit(d["log_ret"].fillna(0).to_numpy())
        d["garch_vol"] = model.conditional_vol(d["log_ret"].fillna(0).to_numpy())
        if verbose:
            state = "converged" if model.converged else "fell back to defaults"
            print(f"  GARCH {state}: alpha={model.alpha:.3f} "
                  f"beta={model.beta:.3f} persistence={model.persistence:.3f}")
    except Exception as e:                                   # noqa: BLE001
        if verbose:
            print(f"  ! GARCH failed ({e}); using rolling std instead")
        d["garch_vol"] = d["log_ret"].rolling(24, min_periods=2).std().bfill().fillna(0)

    avg = d["garch_vol"].expanding(min_periods=50).mean()
    d["vol_regime"] = (d["garch_vol"] / avg.replace(0, np.nan)).fillna(1.0).clip(0, 5)
    return d


GARCH_FEATURES = ["garch_vol", "vol_regime"]
