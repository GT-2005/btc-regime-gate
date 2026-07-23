"""
latent.signals — the state-estimation layer.

Nothing here is imported from a black-box library. The Kalman filter, the
hidden-Markov model and the Hurst exponent are all written out, partly so
you can read what they actually do and partly so the dependency list stays
short enough to install anywhere.

    kalman_filter(prices)      level and velocity of the underlying trend
    GaussianHMM                three-state regime classifier
    hurst_exponent(series)     trending vs mean-reverting
    order_flow_imbalance(df)   crude buy/sell pressure from OHLCV
    vwap_zscore(df)            distance from rolling VWAP, in sigmas
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


# ─────────────────────────────────────────────────────────────────────────
# Kalman filter — local linear trend
# ─────────────────────────────────────────────────────────────────────────
def kalman_filter(
    prices: np.ndarray | pd.Series,
    process_var: float = 1e-5,
    measure_var: float = 1e-2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constant-velocity Kalman filter over a price series.

    State is [level, velocity]. The filter answers a question a moving
    average cannot: not "where has price been", but "where is it now and
    how fast is it moving".

    Returns (level, velocity), both same length as the input.

    process_var  — how much you believe the trend genuinely changes.
                   Larger = the filter reacts faster and is noisier.
    measure_var  — how noisy you believe the observed price is.
                   Larger = the filter smooths harder.
    """
    z = np.asarray(prices, dtype=float)
    n = len(z)
    if n == 0:
        return np.array([]), np.array([])

    # state transition: level += velocity, velocity persists
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])                      # we observe level only
    Q = np.array([[process_var, 0.0], [0.0, process_var]])
    R = np.array([[measure_var]])

    x = np.array([[z[0]], [0.0]])                   # initial state
    P = np.eye(2) * 1.0                             # initial uncertainty

    level = np.empty(n)
    velocity = np.empty(n)

    for t in range(n):
        # predict
        x = F @ x
        P = F @ P @ F.T + Q

        # update
        y = z[t] - (H @ x)[0, 0]                    # innovation
        S = (H @ P @ H.T + R)[0, 0]
        K = (P @ H.T) / S                           # Kalman gain
        x = x + K * y
        P = (np.eye(2) - K @ H) @ P

        level[t] = x[0, 0]
        velocity[t] = x[1, 0]

    return level, velocity


# ─────────────────────────────────────────────────────────────────────────
# Hidden Markov model — three regimes, Gaussian emissions
# ─────────────────────────────────────────────────────────────────────────
def _lse(a: np.ndarray, axis: int) -> np.ndarray:
    """
    log-sum-exp along one axis.

    scipy has this, but it is generic and we call it hundreds of thousands
    of times. A direct implementation is several times faster and that is
    the difference between the HMM taking seconds and taking minutes.
    """
    m = np.max(a, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    return np.squeeze(m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)), axis=axis)


class GaussianHMM:
    """
    A three-state HMM fitted by Baum-Welch, working in log space throughout
    so the forward-backward pass doesn't collapse to zero on long sequences.

    Nobody tells it what "bear" means. It finds three return distributions
    and we label them afterwards by mean return, which is why the labels
    come out stable run to run.

    max_fit_obs caps how much history is used for fitting. Baum-Welch is
    inherently sequential, so cost grows linearly with length; the most
    recent several thousand bars carry the information anyway.
    """

    def __init__(self, n_states: int = 3, n_iter: int = 50, tol: float = 1e-6,
                 seed: int = 42, max_fit_obs: int = 8000):
        self.k = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed
        self.max_fit_obs = max_fit_obs
        self.converged = False
        self.n_iter_run = 0

    # ── internals ────────────────────────────────────────────────────
    def _log_emission(self, x: np.ndarray) -> np.ndarray:
        """(T, k) matrix of log P(x_t | state)."""
        return norm.logpdf(x[:, None], self.mu[None, :], self.sigma[None, :])

    def _forward(self, log_b: np.ndarray) -> tuple[np.ndarray, float]:
        T = len(log_b)
        la = np.empty((T, self.k))
        la[0] = np.log(self.pi + 1e-300) + log_b[0]
        A = self.log_A
        for t in range(1, T):
            la[t] = _lse(la[t - 1][:, None] + A, axis=0) + log_b[t]
        return la, float(_lse(la[-1][None, :], axis=1)[0])

    def _backward(self, log_b: np.ndarray) -> np.ndarray:
        T = len(log_b)
        lb = np.zeros((T, self.k))
        A = self.log_A
        for t in range(T - 2, -1, -1):
            lb[t] = _lse(A + (log_b[t + 1] + lb[t + 1])[None, :], axis=1)
        return lb

    # ── public ───────────────────────────────────────────────────────
    def fit(self, x: np.ndarray) -> "GaussianHMM":
        x_all = np.asarray(x, dtype=float)
        x_all = x_all[np.isfinite(x_all)]
        if len(x_all) < self.k * 20:
            raise ValueError(f"Need at least {self.k * 20} finite observations to fit.")

        # fit on the most recent slice; predict later on everything
        x = x_all[-self.max_fit_obs:] if len(x_all) > self.max_fit_obs else x_all
        T = len(x)

        rng = np.random.default_rng(self.seed)
        qs = np.quantile(x, np.linspace(0.15, 0.85, self.k))
        self.mu = qs + rng.normal(0, x.std() * 0.05, self.k)
        self.sigma = np.full(self.k, max(x.std(), 1e-8))
        self.pi = np.full(self.k, 1 / self.k)
        A = np.full((self.k, self.k), 0.1 / (self.k - 1))
        np.fill_diagonal(A, 0.9)                     # regimes persist
        self.log_A = np.log(A)

        prev_ll = -np.inf
        for it in range(self.n_iter):
            self.n_iter_run = it + 1
            log_b = self._log_emission(x)
            la, ll = self._forward(log_b)
            lb = self._backward(log_b)

            # responsibilities
            log_g = la + lb - ll
            g = np.exp(log_g - log_g.max(axis=1, keepdims=True))
            g /= g.sum(axis=1, keepdims=True)

            # transition counts, all timesteps at once
            m = (la[:-1, :, None] + self.log_A[None, :, :]
                 + (log_b[1:] + lb[1:])[:, None, :] - ll)
            m -= m.max()
            xi = np.exp(m).sum(axis=0)
            xi /= xi.sum(axis=1, keepdims=True) + 1e-300

            # M-step
            self.pi = g[0] / g[0].sum()
            self.log_A = np.log(xi + 1e-300)
            w = g.sum(axis=0) + 1e-300
            self.mu = (g * x[:, None]).sum(axis=0) / w
            var = (g * (x[:, None] - self.mu) ** 2).sum(axis=0) / w
            self.sigma = np.sqrt(np.maximum(var, 1e-12))

            # relative test — log-likelihood magnitude scales with sample size,
            # so an absolute threshold means different things on different data
            if abs(ll - prev_ll) < self.tol * max(1.0, abs(prev_ll)):
                self.converged = True
                break
            prev_ll = ll

        # order states by mean return: 0 = bear, 1 = sideways, 2 = bull
        order = np.argsort(self.mu)
        self.mu, self.sigma = self.mu[order], self.sigma[order]
        self.pi = self.pi[order]
        self.log_A = self.log_A[np.ix_(order, order)]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Most likely state per observation (0 bear, 1 sideways, 2 bull)."""
        x = np.asarray(x, dtype=float)
        ok = np.isfinite(x)
        out = np.full(len(x), 1)                     # default to sideways
        if ok.sum() == 0:
            return out
        log_b = self._log_emission(x[ok])
        la, ll = self._forward(log_b)
        lb = self._backward(log_b)
        out[ok] = np.argmax(la + lb - ll, axis=1)
        return out


# ─────────────────────────────────────────────────────────────────────────
# Hurst exponent — rescaled range
# ─────────────────────────────────────────────────────────────────────────
def hurst_exponent(series: np.ndarray | pd.Series, max_lag: int = 40) -> float:
    """
    H > 0.5  trending, momentum should work
    H ~ 0.5  random walk
    H < 0.5  mean-reverting, fade the move

    Returns 0.5 if there isn't enough clean data to say anything.
    """
    s = np.asarray(series, dtype=float)
    s = s[np.isfinite(s)]
    if len(s) < max_lag * 2:
        return 0.5

    lags = np.arange(2, max_lag)
    tau = []
    for lag in lags:
        diff = s[lag:] - s[:-lag]
        sd = np.sqrt(np.std(diff))
        tau.append(sd if sd > 0 else 1e-12)

    try:
        slope = np.polyfit(np.log(lags), np.log(tau), 1)[0]
    except (np.linalg.LinAlgError, ValueError):
        return 0.5
    return float(np.clip(slope * 2.0, 0.0, 1.0))


def rolling_hurst(series: pd.Series, window: int = 100, step: int = 5) -> pd.Series:
    """
    Hurst over a rolling window.

    Computed every `step` bars and held constant in between. Hurst over a
    100-bar window barely moves from one bar to the next, so recomputing it
    every single bar burns time for no information.
    """
    arr = series.to_numpy(dtype=float)
    n = len(arr)
    vals = np.full(n, 0.5)
    max_lag = max(4, min(20, window // 3))
    lags = np.arange(2, max_lag)
    log_lags = np.log(lags)
    # slope of a least-squares line is fixed-denominator when x is fixed
    lx = log_lags - log_lags.mean()
    denom = float((lx ** 2).sum()) or 1e-12

    last = 0.5
    for i in range(window, n, step):
        w = arr[i - window:i]
        if not np.all(np.isfinite(w)):
            vals[i:i + step] = last
            continue
        tau = np.empty(len(lags))
        for j, lag in enumerate(lags):
            d = w[lag:] - w[:-lag]
            sd = np.sqrt(np.std(d))
            tau[j] = sd if sd > 0 else 1e-12
        ly = np.log(tau)
        slope = float((lx * (ly - ly.mean())).sum() / denom)
        last = float(np.clip(slope * 2.0, 0.0, 1.0))
        vals[i:i + step] = last

    return pd.Series(vals, index=series.index)


# ─────────────────────────────────────────────────────────────────────────
# Microstructure proxies
# ─────────────────────────────────────────────────────────────────────────
def order_flow_imbalance(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """
    Approximate buy/sell pressure from OHLCV alone.

    Real OFI needs the order book. Without it, where the bar closed inside
    its own range is the best cheap proxy: closing near the high means
    buyers were in control of that bar.
    """
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    pos = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng
    signed = pos.fillna(0) * df["volume"]
    return (signed.rolling(window, min_periods=1).sum()
            / df["volume"].rolling(window, min_periods=1).sum().replace(0, np.nan)
            ).fillna(0)


def vwap_zscore(df: pd.DataFrame, window: int = 48) -> pd.Series:
    """How far price sits from rolling VWAP, in standard deviations."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    pv = (tp * df["volume"]).rolling(window, min_periods=1).sum()
    vv = df["volume"].rolling(window, min_periods=1).sum().replace(0, np.nan)
    vwap = pv / vv
    dev = df["close"] - vwap
    sd = dev.rolling(window, min_periods=2).std().replace(0, np.nan)
    return (dev / sd).fillna(0).clip(-5, 5)


# ─────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────
SIGNAL_FEATURES = ["kf_velocity", "kf_gap", "hmm_regime", "hurst", "ofi", "vwap_z"]


def add_signal_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Attach the signal layer. Each block is wrapped: if one model fails, the
    column is filled with a neutral value and the pipeline carries on. It
    would rather trade on 25 features than crash on 29.
    """
    d = df.copy()

    # Kalman
    try:
        level, vel = kalman_filter(d["close"].to_numpy())
        d["kf_level"] = level
        # normalise so velocity is comparable at any price level
        d["kf_velocity"] = vel / d["close"].replace(0, np.nan)
        d["kf_gap"] = (d["close"] - d["kf_level"]) / d["atr"].replace(0, np.nan)
    except Exception as e:                                   # noqa: BLE001
        if verbose:
            print(f"  ! Kalman failed ({e}); filling neutral")
        d["kf_level"], d["kf_velocity"], d["kf_gap"] = d["close"], 0.0, 0.0

    # HMM
    try:
        r = d["log_ret"].fillna(0).to_numpy()
        hmm = GaussianHMM().fit(r)
        d["hmm_regime"] = hmm.predict(r)
        if verbose and not hmm.converged:
            print("  ! HMM did not fully converge — usual on short data, continuing")
    except Exception as e:                                   # noqa: BLE001
        if verbose:
            print(f"  ! HMM failed ({e}); defaulting every bar to sideways")
        d["hmm_regime"] = 1

    # Hurst
    try:
        d["hurst"] = rolling_hurst(d["close"], window=100)
    except Exception as e:                                   # noqa: BLE001
        if verbose:
            print(f"  ! Hurst failed ({e}); filling 0.5")
        d["hurst"] = 0.5

    # microstructure
    try:
        d["ofi"] = order_flow_imbalance(d)
        d["vwap_z"] = vwap_zscore(d)
    except Exception as e:                                   # noqa: BLE001
        if verbose:
            print(f"  ! Microstructure proxies failed ({e}); filling zero")
        d["ofi"], d["vwap_z"] = 0.0, 0.0

    return d
