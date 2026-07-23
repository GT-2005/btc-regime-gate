"""
latent.data — loading market data and turning it into model input.

Two public functions:

    load_data(path)            read an OHLCV file, clean it, sort it
    add_base_features(df)      attach the 30+ engineered columns

Everything here is deliberately dimensionless where it can be. If a feature
scales with price, the model learns "Bitcoin was cheap in 2020" instead of
learning anything useful.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED = ["date", "open", "high", "low", "close", "volume"]

# Common column spellings mapped onto ours.
ALIASES = {
    "time": "date", "timestamp": "date", "datetime": "date", "open time": "date",
    "o": "open", "h": "high", "l": "low", "c": "close",
    "vol": "volume", "v": "volume", "base volume": "volume",
    "adj close": "close", "price": "close",
}


# ─────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────
def load_data(path: str | Path) -> pd.DataFrame:
    """
    Read an OHLCV file and return it clean, lowercase and sorted oldest-first.

    Accepts .xlsx, .xls and .csv. Raises with a readable message rather than
    letting pandas throw something cryptic three functions later.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"No file at {path}\n"
            "Check the path. If you're on Windows, watch for backslashes."
        )

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() in {".csv", ".txt"}:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}. Use .xlsx or .csv.")

    # tidy the headers
    df.columns = [str(c).strip().lower() for c in df.columns]
    df = df.rename(columns={k: v for k, v in ALIASES.items() if k in df.columns})

    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing column(s): {missing}\n"
            f"Found: {list(df.columns)}\n"
            f"The loader needs: {REQUIRED}"
        )

    df = df[REQUIRED].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")

    bad_dates = int(df["date"].isna().sum())
    if bad_dates:
        warnings.warn(f"Dropping {bad_dates} row(s) with an unreadable date.")
        df = df.dropna(subset=["date"])

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])

    # ─── the single most important line in this file ───
    # If rows are out of order, every rolling window and the target column
    # silently pull future information into past rows. The backtest looks
    # wonderful and it is fiction.
    df = df.sort_values("date").drop_duplicates(subset="date", keep="last")

    return df.reset_index(drop=True)


def validate(df: pd.DataFrame) -> list[str]:
    """Return a list of human-readable problems. Empty list means it looks fine."""
    problems: list[str] = []

    if len(df) < 5000:
        problems.append(
            f"Only {len(df)} rows. Walk-forward folds need roughly 5,000+ "
            "to have anything to learn from."
        )

    gaps = df["date"].diff().dropna()
    if len(gaps):
        typical = gaps.median()
        big = int((gaps > typical * 1.5).sum())
        if big:
            problems.append(
                f"{big} gap(s) in the series, relative to a typical spacing of {typical}."
            )

    if (df["high"] < df["low"]).any():
        problems.append("Some rows have high below low. The data is corrupt.")

    body_ok = (df["high"] >= df[["open", "close"]].max(axis=1)) & \
              (df["low"] <= df[["open", "close"]].min(axis=1))
    if not body_ok.all():
        problems.append(f"{int((~body_ok).sum())} row(s) where open/close sit outside high/low.")

    if (df["volume"] <= 0).sum() > len(df) * 0.01:
        problems.append("More than 1% of bars have zero or negative volume.")

    return problems


# ─────────────────────────────────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────────────────────────────────
def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean()


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach every engineered column the rest of the pipeline expects.

    The target is binary: did the close rise over the next 5 bars?
    """
    d = df.copy()
    c, h, l, o, v = d["close"], d["high"], d["low"], d["open"], d["volume"]

    # ── returns ──────────────────────────────────────────────────────
    d["ret"] = c.pct_change()
    d["log_ret"] = np.log(c / c.shift(1))

    # ── momentum at three horizons ───────────────────────────────────
    for k in (3, 5, 10):
        d[f"mom_{k}"] = c.pct_change(k)

    # ── trend ────────────────────────────────────────────────────────
    for span in (9, 21, 50, 200):
        d[f"ema_{span}"] = c.ewm(span=span, adjust=False).mean()
    d["fast_slow"] = (d["ema_9"] - d["ema_21"]) / d["ema_21"].replace(0, np.nan)
    d["trend_up"] = (d["ema_50"] > d["ema_200"]).astype(int)

    # ── volatility ───────────────────────────────────────────────────
    d["atr"] = _atr(d, 14)
    d["atr_pct"] = d["atr"] / c.replace(0, np.nan)
    d["realised_vol"] = d["log_ret"].rolling(24, min_periods=2).std()

    # ── oscillators ──────────────────────────────────────────────────
    d["rsi"] = _rsi(c, 14)
    d["rsi_smooth"] = d["rsi"].ewm(span=5, adjust=False).mean()

    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    d["macd"] = (ema12 - ema26) / c.replace(0, np.nan)      # normalised
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]

    # ── participation ────────────────────────────────────────────────
    vol_ma = v.rolling(20, min_periods=1).mean()
    d["vol_ratio"] = v / vol_ma.replace(0, np.nan)
    d["vol_trend"] = d["vol_ratio"].rolling(5, min_periods=1).mean()

    # ── candle anatomy, scaled by ATR so it travels across price levels ──
    atr_safe = d["atr"].replace(0, np.nan)
    d["body_pct"] = (c - o) / atr_safe
    d["upper_wick"] = (h - np.maximum(o, c)) / atr_safe
    d["lower_wick"] = (np.minimum(o, c) - l) / atr_safe
    d["range_pct"] = (h - l) / atr_safe

    # ── target: does close rise over the next 5 bars? ────────────────
    d["target"] = (c.shift(-5) > c).astype(int)
    # the final 5 rows have no future to look at
    d.loc[d.index[-5:], "target"] = np.nan

    return d.replace([np.inf, -np.inf], np.nan)


#: Columns handed to the ML layer. Signal-layer outputs are appended later.
BASE_FEATURES = [
    "ret", "log_ret",
    "mom_3", "mom_5", "mom_10",
    "fast_slow", "trend_up",
    "atr_pct", "realised_vol",
    "rsi_smooth", "macd", "macd_hist",
    "vol_ratio", "vol_trend",
    "body_pct", "upper_wick", "lower_wick", "range_pct",
]
