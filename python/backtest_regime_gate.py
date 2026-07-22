"""
BTC Regime Gate v2.6 — Backtest
Proven edge: short-priority / short_only, SL 1 ATR, TP 3 ATR
Risk 2% per trade. Breakout playbook C disabled (it destroyed returns in tests).

v2.6: Optional VPIN toxicity filter (--vpin) from volume-clock papers.
  Yearly runner: --years 2019,2020,2021,2022,2024,2025
  Default stays plain short_only (best $ across those years); --vpin is research risk gate.

Honest note: ~1% year on 2025 short_only is real edge vs -10% B&H, but NOT 15%.
Pushing harder with noisy breakouts lost -15%. Next alpha needs new research, not spam.

Test on NEW data:
  python backtest_regime_gate.py --data /path/to/BTCUSDT_XXXX_full.xlsx
  python backtest_regime_gate.py --data /path/to/BTCUSDT_15m_All.csv --years 2019,2020,2021,2022,2024,2025
  python backtest_regime_gate.py --data /path/to/file.xlsx --mode long_only
  python backtest_regime_gate.py --data /path/to/file.xlsx --compare

Colab: always opens upload dialog (any year xlsx/csv).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEFAULT_ML_MODEL = os.path.join(_HERE, "models_btc15m", "btc15m_model.joblib")
USE_ML_FILTER = False
ML_THR_LONG = 0.60
ML_THR_SHORT = 0.40
ML_MODEL_PATH = DEFAULT_ML_MODEL
USE_VPIN_FILTER = False  # opt-in via --vpin; plain short_only still best $ across years

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
def resolve_data_path(cli_path: str | None = None) -> str:
    """Resolve Excel path: CLI > Colab upload > env > local Downloads."""
    if cli_path:
        if not os.path.isfile(cli_path):
            raise FileNotFoundError(f"File not found: {cli_path}")
        return cli_path

    # Google Colab → always prompt (any new dataset)
    try:
        from google.colab import files  # type: ignore

        print("Upload your OHLCV Excel (e.g. BTCUSDT_2024_full.xlsx or 2025) …")
        uploaded = files.upload()
        if not uploaded:
            raise FileNotFoundError("No file uploaded.")
        name = next(iter(uploaded.keys()))
        path = name if os.path.isfile(name) else os.path.join("/content", name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Upload saved but file not found: {path}")
        print("Using uploaded file:", path)
        return path
    except ImportError:
        pass  # not Colab

    candidates = [
        os.environ.get("BTC_DATA_PATH", ""),
        "/Users/ganeshtilekar/BTC_DATA/BTCUSDT_15m_All.csv",
        "/Users/ganeshtilekar/Downloads/BTCUSDT_2024_full.xlsx",
        "/Users/ganeshtilekar/Downloads/BTCUSDT_2025_full.xlsx",
        os.path.join(os.getcwd(), "BTCUSDT_2024_full.xlsx"),
        os.path.join(os.getcwd(), "BTCUSDT_2025_full.xlsx"),
    ]
    for p in candidates:
        if p and os.path.isfile(p):
            return p

    raise FileNotFoundError(
        "No data file found.\n"
        "  python backtest_regime_gate.py --data /path/to/file.xlsx|csv\n"
        "  or set BTC_DATA_PATH, or upload in Colab."
    )


DATA_PATH = None  # resolved in main()
STARTING_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02          # 2% — scales the proven edge (was 1% → ~0.4–1% return)
RISK_BREAKOUT = 0.02
MAX_POSITION_FRAC = 0.25
SLIPPAGE_BPS = 9.0
FEE_BPS_ONE_WAY = 4.0
BAR_TF = "15min"
HTF_TF = "1h"

# v2.4 final — restore proven Auto A/B edge, 2% risk, C off (C destroyed returns)
ADX_TREND = 30.0
ADX_RANGE = 18.0
ADX_BREAKOUT = 35.0
VOL_SHOCK_MULT = 2.5
PULLBACK_TOL = 0.35
EMA_SEP_ATR = 0.25
ATR_SL_TREND = 1.0
ATR_TP_TREND = 3.0
ATR_TRAIL_TREND = 2.5
BE_AT_R = 1.0
MAX_HOLD_TREND = 96
USE_FIXED_TP_A = True
ENABLE_BREAKOUT_C = False
BREAKOUT_LOOKBACK = 48
BREAKOUT_BUFFER_ATR = 0.35
ATR_SL_BREAK = 1.5
ATR_TP_BREAK = 6.0
ATR_TRAIL_BREAK = 3.0
MAX_HOLD_BREAK = 160
VWAP_Z_ENTRY = 2.2
VWAP_Z_EXIT = 0.3
RSI_OS, RSI_OB = 28.0, 72.0
ATR_SL_RANGE = 1.0
ATR_TP_RANGE = 2.0
MAX_HOLD_RANGE = 16
RVOL_MIN = 1.2
RVOL_BREAK = 1.5
SCORE_LONG = 70.0
COOLDOWN_BARS = 8
COOLDOWN_BREAK = 24
MIN_HOLD_BARS = 3

TRADE_MODE = "short_only"  # best $ return on 2025; use --mode auto for mixed


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────
def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    au = up.ewm(alpha=1 / n, adjust=False).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = au / ad.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    dn = -df["low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = atr(df, n)  # smoothed later consistently via Wilder on TR
    # Wilder ATR-style for DMI
    prev_c = df["close"].shift(1)
    tr_raw = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev_c).abs(),
            (df["low"] - prev_c).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_w = tr_raw.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def _parse_binance_ms(open_time: pd.Series) -> pd.DatetimeIndex:
    ot = open_time.astype("float64").to_numpy()
    ms = ot.copy()
    ms = np.where(ms > 1e14, ms / 1000.0, ms)
    ms = np.where(ms > 1e14, ms / 1000.0, ms)
    return pd.to_datetime(ms, unit="ms", utc=True)


def load_ohlcv(path: str) -> pd.DataFrame:
    """Load Excel (headered) or Binance kline CSV (headerless). Keeps trades/taker if present."""
    lower = path.lower()
    if lower.endswith(".csv"):
        peek = pd.read_csv(path, header=None, nrows=3)
        # headerless Binance: col0 numeric ms, col1–5 OHLCV
        if peek.shape[1] >= 6 and pd.api.types.is_numeric_dtype(peek.iloc[:, 0]):
            df = pd.read_csv(path, header=None)
            cols = [
                "open_time",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "close_time",
                "quote_vol",
                "trades",
                "taker_buy_base",
                "taker_buy_quote",
                "ignore",
            ]
            df.columns = cols[: df.shape[1]]
            df["timestamp"] = _parse_binance_ms(df["open_time"])
            keep = ["open", "high", "low", "close", "volume"]
            for extra in ("trades", "taker_buy_base"):
                if extra in df.columns:
                    keep.append(extra)
            for c in keep:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open", "high", "low", "close", "volume"])
            df = df[(df["timestamp"] >= "2017-08-01") & (df["timestamp"] <= "2030-12-31")]
            df = df[(df["high"] >= df["low"]) & (df["close"] > 0) & (df["volume"] >= 0)]
            df = df.sort_values("timestamp").drop_duplicates("timestamp")
            return df.set_index("timestamp")[keep].astype(float)

        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)

    df = df.rename(
        columns={
            "Open Time": "timestamp",
            "open_time": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    if "timestamp" not in df.columns:
        raise ValueError(f"No timestamp column in {path}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df.set_index("timestamp")
    keep = ["open", "high", "low", "close", "volume"]
    for extra in ("trades", "taker_buy_base"):
        if extra in df.columns:
            keep.append(extra)
    return df[keep].astype(float)


def prepare_timeframes(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """If bars are already ~15m, use as-is; if ~5m, resample up."""
    ohlcv = raw[["open", "high", "low", "close", "volume"]].copy()
    median_dt = ohlcv.index.to_series().diff().median()
    if median_dt is pd.NaT:
        raise ValueError("Cannot infer bar size from timestamps")
    if median_dt <= pd.Timedelta("7min"):
        df15 = resample_ohlcv(ohlcv, BAR_TF)
        src = ohlcv
        print(f"Detected ~{median_dt}; resampling 5m → 15m")
    elif median_dt <= pd.Timedelta("20min"):
        df15 = ohlcv.copy()
        src = ohlcv
        print(f"Detected ~{median_dt}; using as 15m")
    else:
        df15 = ohlcv.copy()
        src = ohlcv
        print(f"Detected ~{median_dt}; treating as base TF")

    # carry flow cols onto 15m if present (for ML)
    for extra in ("trades", "taker_buy_base"):
        if extra in raw.columns and median_dt <= pd.Timedelta("20min") and median_dt > pd.Timedelta("7min"):
            df15[extra] = raw[extra]
        elif extra in raw.columns and median_dt <= pd.Timedelta("7min"):
            df15[extra] = raw[extra].resample(BAR_TF, label="left", closed="left").sum()

    df1h = resample_ohlcv(src, HTF_TF)
    df1d = resample_ohlcv(src, "1D")
    return df15, df1h, df1d


_ML_PROBA_CACHE: dict[str, pd.Series] = {}


def score_ml_proba(ohlcv15: pd.DataFrame, model_path: str) -> pd.Series:
    """P(up) from saved LightGBM bundle; cached per (path, n_bars)."""
    import joblib
    from train_btc_patterns import build_features as build_ml_features

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"ML model not found: {model_path}")

    cache_key = f"{os.path.abspath(model_path)}|{len(ohlcv15)}|{ohlcv15.index[0]}|{ohlcv15.index[-1]}"
    if cache_key in _ML_PROBA_CACHE:
        return _ML_PROBA_CACHE[cache_key]

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols: list[str] = list(bundle["features"])

    ml_src = ohlcv15.copy()
    if "trades" not in ml_src.columns:
        ml_src["trades"] = 0.0
    if "taker_buy_base" not in ml_src.columns:
        ml_src["taker_buy_base"] = ml_src["volume"] * 0.5
    if ml_src.index.tz is None:
        ml_src.index = ml_src.index.tz_localize("UTC")
    else:
        ml_src.index = ml_src.index.tz_convert("UTC")

    print(f"ML gate: scoring {len(ml_src):,} bars with {os.path.basename(model_path)} …")
    ml = build_ml_features(ml_src)
    missing = [c for c in feature_cols if c not in ml.columns]
    if missing:
        raise ValueError(f"ML features missing from matrix: {missing[:10]}")
    X = ml[feature_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median(numeric_only=True))
    proba = pd.Series(model.predict_proba(X)[:, 1], index=ml.index, name="ml_p_up")
    _ML_PROBA_CACHE[cache_key] = proba
    return proba


def apply_ml_filter(
    feats: pd.DataFrame,
    ohlcv15: pd.DataFrame,
    model_path: str,
    thr_long: float = ML_THR_LONG,
    thr_short: float = ML_THR_SHORT,
    proba: pd.Series | None = None,
) -> pd.DataFrame:
    """Gate regime signals with trained P(up) model. Longs need high p; shorts need low p."""
    if proba is None:
        proba = score_ml_proba(ohlcv15, model_path)

    d = feats.copy()
    p_idx = proba.copy()
    if d.index.tz is None and p_idx.index.tz is not None:
        d.index = d.index.tz_localize("UTC")
    elif d.index.tz is not None and p_idx.index.tz is None:
        p_idx.index = p_idx.index.tz_localize("UTC")
    p = p_idx.reindex(d.index)
    d["ml_p_up"] = p
    long_ok = (p >= thr_long).fillna(False)
    short_ok = (p <= thr_short).fillna(False)

    before_long = int(d["long_sig_a"].sum() + d["long_sig_b"].sum() + d["long_sig_c"].sum())
    before_short = int(d["short_sig_a"].sum() + d["short_sig_b"].sum() + d["short_sig_c"].sum())
    for col in ("long_sig_a", "long_sig_b", "long_sig_c"):
        d[col] = d[col] & long_ok
    for col in ("short_sig_a", "short_sig_b", "short_sig_c", "short_hard_s"):
        if col in d.columns:
            d[col] = d[col] & short_ok
    after_long = int(d["long_sig_a"].sum() + d["long_sig_b"].sum() + d["long_sig_c"].sum())
    after_short = int(d["short_sig_a"].sum() + d["short_sig_b"].sum() + d["short_sig_c"].sum())
    print(
        f"ML gate thr_long={thr_long:.2f} thr_short={thr_short:.2f} | "
        f"long signals {before_long}→{after_long}, short signals {before_short}→{after_short}"
    )
    return d


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    ohlc = df.resample(rule, label="left", closed="left").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return ohlc.dropna()


def build_features(df15: pd.DataFrame, df1h: pd.DataFrame, df1d: pd.DataFrame | None = None) -> pd.DataFrame:
    d = df15.copy()
    d["atr"] = atr(d, 14)
    d["atr_ma"] = sma(d["atr"], 50)
    d["adx"] = adx(d, 14)
    d["ema21"] = ema(d["close"], 21)
    d["ema50"] = ema(d["close"], 50)
    d["ema_slope"] = ema(d["close"], 10)
    d["kf_vel"] = d["ema_slope"] - d["ema_slope"].shift(1)
    d["vol_sma"] = sma(d["volume"], 20)
    d["rvol"] = np.where(d["vol_sma"] > 0, d["volume"] / d["vol_sma"], 0.0)
    d["rsi"] = rsi(d["close"], 14)

    typ = (d["high"] + d["low"] + d["close"]) / 3.0
    vol_sum = d["volume"].rolling(48).sum()
    typ_vol = (typ * d["volume"]).rolling(48).sum()
    d["roll_vwap"] = np.where(vol_sum > 0, typ_vol / vol_sum, d["close"])
    d["vwap_z"] = (d["close"] - d["roll_vwap"]) / (d["close"] - d["roll_vwap"]).rolling(48).std().replace(0, np.nan)

    htf = df1h.copy()
    htf["ema50"] = ema(htf["close"], 50)
    htf["ema200"] = ema(htf["close"], 200)
    htf_flag = (htf["ema50"] > htf["ema200"]).astype(float).shift(1)
    d["htf_up"] = htf_flag.reindex(d.index, method="ffill")
    d["htf_dn"] = (1.0 - d["htf_up"]).where(d["htf_up"].notna(), np.nan)

    # Daily macro bias (lagged 1 day)
    # Faster than EMA50/200 alone: if price < daily EMA50 → short bias (no more long bleed)
    # Time-series momentum (Moskowitz): sign of ~60d daily return — KEEP paper filter
    if df1d is None:
        d["daily_up"] = True
        d["daily_dn"] = True
        d["tsmom_up"] = True
        d["tsmom_dn"] = True
    else:
        day = df1d.copy()
        day["ema50"] = ema(day["close"], 50)
        day["ema200"] = ema(day["close"], 200)
        day_ema50 = day["ema50"].shift(1)
        day_ema200 = day["ema200"].shift(1)
        day_close = day["close"].shift(1)
        bull = (day_close > day_ema50) & (day_ema50 > day_ema200)
        bear = day_close < day_ema50  # price below daily mid → prefer short sell
        d["daily_up"] = bull.reindex(d.index, method="ffill").fillna(False)
        d["daily_dn"] = bear.reindex(d.index, method="ffill").fillna(False)
        # lagged daily TS momentum (~60 trading days)
        tsm = (day["close"] / day["close"].shift(60) - 1.0).shift(1)
        d["tsmom_up"] = (tsm > 0).reindex(d.index, method="ffill").fillna(False)
        d["tsmom_dn"] = (tsm < 0).reindex(d.index, method="ffill").fillna(False)

    d["vol_shock"] = (d["atr_ma"] > 0) & ((d["atr"] / d["atr_ma"]) >= VOL_SHOCK_MULT)
    d["is_trend"] = (~d["vol_shock"]) & (d["adx"] >= ADX_TREND)
    d["is_range"] = (~d["vol_shock"]) & (d["adx"] <= ADX_RANGE)

    # Trend separation + deep pullback then strong reclaim (fewer, better entries)
    ema_sep = (d["ema21"] - d["ema50"]).abs() / d["atr"].replace(0, np.nan)
    strong_sep = ema_sep >= EMA_SEP_ATR
    pulled_long = d["low"].rolling(6).min() <= (d["ema21"] + PULLBACK_TOL * d["atr"])
    near_long = (d["low"] <= d["ema21"] + PULLBACK_TOL * d["atr"]) | (
        (d["close"] - d["ema21"]).abs() <= PULLBACK_TOL * d["atr"]
    )
    reclaim_long = (
        (d["close"] > d["ema21"])
        & (d["close"] >= d["open"])
        & (d["close"] > d["close"].shift(1))
        & (d["close"] > d["ema50"])
    )
    struct_up = (d["ema21"] > d["ema50"]) & strong_sep
    d["long_hard_a"] = (
        d["is_trend"]
        & (d["htf_up"] == 1)
        & struct_up
        & pulled_long
        & near_long
        & reclaim_long
        & (d["rvol"] >= RVOL_MIN)
        & (d["kf_vel"] >= 0)
    )

    # ── SHORT SELL (Playbook A) ─────────────────────────────────────────────
    # Classic short: SELL first at a HIGHER price (rally into EMA), then BUY BACK lower.
    # Profit = entry_sell_price - exit_cover_price (when price falls).
    pulled_short = d["high"].rolling(6).max() >= (d["ema21"] - PULLBACK_TOL * d["atr"])
    near_short = (d["high"] >= d["ema21"] - PULLBACK_TOL * d["atr"]) | (
        (d["close"] - d["ema21"]).abs() <= PULLBACK_TOL * d["atr"]
    )
    # Rejection candle after rally = open short (sell high)
    reclaim_short = (
        (d["close"] < d["ema21"])
        & (d["close"] <= d["open"])
        & (d["close"] < d["close"].shift(1))
        & (d["close"] < d["ema50"])
    )
    struct_dn = (d["ema21"] < d["ema50"]) & strong_sep
    # Bearish engulf / strong rejection boost for short sell
    bear_reject = (d["open"] > d["close"]) & (d["high"] >= d["ema21"]) & (d["close"] < d["open"].shift(1))
    d["short_hard_a"] = (
        d["is_trend"]
        & (d["htf_dn"] == 1)
        & struct_dn
        & pulled_short
        & near_short
        & reclaim_short
        & (d["rvol"] >= RVOL_MIN)
        & (d["kf_vel"] <= 0)
    )
    # Extra short-sell path: clear rejection at EMA in HTF downtrend
    d["short_hard_s"] = (
        d["is_trend"]
        & (d["htf_dn"] == 1)
        & struct_dn
        & bear_reject
        & (d["rvol"] >= RVOL_MIN)
        & (d["kf_vel"] <= 0)
        & (d["rsi"] >= 45)  # sell into relative strength, not already crashed
    )

    score_a_long = (
        d["long_hard_a"].astype(float) * 40
        + (d["htf_up"] == 1).astype(float) * 20
        + (d["rvol"] >= RVOL_MIN).astype(float) * 15
        + (d["kf_vel"] >= 0).astype(float) * 10
        + struct_up.astype(float) * 10
    )
    score_a_short = (
        d["short_hard_a"].astype(float) * 40
        + (d["htf_dn"] == 1).astype(float) * 20
        + (d["rvol"] >= RVOL_MIN).astype(float) * 15
        + (d["kf_vel"] <= 0).astype(float) * 10
        + struct_dn.astype(float) * 10
    )
    d["long_sig_a"] = d["long_hard_a"] & (score_a_long >= SCORE_LONG)
    d["short_sig_a"] = (d["short_hard_a"] | d["short_hard_s"]) & (score_a_short >= SCORE_LONG)
    # If rejection path fired, still allow short even if base score slightly lower
    d["short_sig_a"] = d["short_sig_a"] | (
        d["short_hard_s"] & (score_a_short + 40 >= SCORE_LONG)
    )

    # Playbook B — fade extremes only; target > stop (big win / small loss geometry)
    rev_up = (d["close"] > d["open"]) & (d["close"] > d["close"].shift(1))
    rev_dn = (d["close"] < d["open"]) & (d["close"] < d["close"].shift(1))
    d["long_hard_b"] = (
        d["is_range"]
        & (d["vwap_z"] <= -VWAP_Z_ENTRY)
        & (d["rsi"] <= RSI_OS)
        & rev_up
        & (d["rvol"] >= RVOL_MIN)
    )
    d["short_hard_b"] = (
        d["is_range"]
        & (d["vwap_z"] >= VWAP_Z_ENTRY)
        & (d["rsi"] >= RSI_OB)
        & rev_dn
        & (d["rvol"] >= RVOL_MIN)
    )
    score_b_long = (
        d["long_hard_b"].astype(float) * 40
        + (d["vwap_z"] <= -VWAP_Z_ENTRY).astype(float) * 20
        + (d["rvol"] >= RVOL_MIN).astype(float) * 15
        + (d["rsi"] <= RSI_OS).astype(float) * 15
        + rev_up.astype(float) * 10
    )
    score_b_short = (
        d["short_hard_b"].astype(float) * 40
        + (d["vwap_z"] >= VWAP_Z_ENTRY).astype(float) * 20
        + (d["rvol"] >= RVOL_MIN).astype(float) * 15
        + (d["rsi"] >= RSI_OB).astype(float) * 15
        + rev_dn.astype(float) * 10
    )
    d["long_sig_b"] = d["long_hard_b"] & (score_b_long >= SCORE_LONG)
    d["short_sig_b"] = d["short_hard_b"] & (score_b_short >= SCORE_LONG)

    # Playbook C — rare BREAKOUT (must clear range + buffer, strong vol)
    prior_high = d["high"].rolling(BREAKOUT_LOOKBACK).max().shift(1)
    prior_low = d["low"].rolling(BREAKOUT_LOOKBACK).min().shift(1)
    buf = BREAKOUT_BUFFER_ATR * d["atr"]
    d["long_sig_c"] = (
        ENABLE_BREAKOUT_C
        & d["daily_up"].fillna(False)
        & (~d["vol_shock"])
        & (d["adx"] >= ADX_BREAKOUT)
        & (d["htf_up"] == 1)
        & (d["ema21"] > d["ema50"])
        & (d["close"] > prior_high + buf)
        & (d["close"] > d["open"])
        & (d["rvol"] >= RVOL_BREAK)
        & (d["kf_vel"] > 0)
    )
    d["short_sig_c"] = (
        ENABLE_BREAKOUT_C
        & d["daily_dn"].fillna(False)
        & (~d["vol_shock"])
        & (d["adx"] >= ADX_BREAKOUT)
        & (d["htf_dn"] == 1)
        & (d["ema21"] < d["ema50"])
        & (d["close"] < prior_low - buf)
        & (d["close"] < d["open"])
        & (d["rvol"] >= RVOL_BREAK)
        & (d["kf_vel"] < 0)
    )

    # Side filter + TS momentum (Moskowitz) — engine already enforces one open trade
    allow_long = pd.Series(True, index=d.index)
    allow_short = pd.Series(True, index=d.index)
    if TRADE_MODE == "long_only":
        allow_short[:] = False
    elif TRADE_MODE == "short_only":
        allow_long[:] = False
    elif TRADE_MODE == "auto":
        allow_long = d["daily_up"].fillna(False)
        allow_short = d["daily_dn"].fillna(False)

    if "tsmom_up" in d.columns:
        allow_long = allow_long & d["tsmom_up"].fillna(False)
        allow_short = allow_short & d["tsmom_dn"].fillna(False)

    d["long_sig_a"] = d["long_sig_a"] & allow_long
    d["long_sig_b"] = d["long_sig_b"] & allow_long
    d["long_sig_c"] = d["long_sig_c"] & allow_long
    d["short_sig_a"] = d["short_sig_a"] & allow_short
    d["short_sig_b"] = d["short_sig_b"] & allow_short
    if "short_hard_s" in d.columns:
        d["short_hard_s"] = d["short_hard_s"] & allow_short
    d["short_sig_c"] = d["short_sig_c"] & allow_short

    return d


@dataclass
class Trade:
    side: str
    playbook: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    qty: float = 0.0
    pnl: float = 0.0
    reason: str = ""


@dataclass
class Engine:
    equity: float = STARTING_CAPITAL
    cash: float = STARTING_CAPITAL
    pos: int = 0  # 0 flat, 1 longA, 2 shortA, 3 longB, 4 shortB, 5 longC, 6 shortC
    entry_bar: int = -1
    entry_price: float = 0.0
    stop: float = 0.0
    tp: float = math.nan
    trail: float = math.nan
    qty: float = 0.0
    last_exit_bar: int = -10_000
    playbook: str = ""
    risk_frac: float = RISK_PER_TRADE
    trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)

    def fee(self, notional: float) -> float:
        return notional * (FEE_BPS_ONE_WAY / 10_000.0)

    def slip_price(self, price: float, side: str, is_entry: bool) -> float:
        half = (SLIPPAGE_BPS / 2.0) / 10_000.0
        if side == "long":
            return price * (1 + half) if is_entry else price * (1 - half)
        return price * (1 - half) if is_entry else price * (1 + half)

    def size_from_stop(self, entry: float, stop: float, risk_frac: float | None = None) -> float:
        rf = RISK_PER_TRADE if risk_frac is None else risk_frac
        risk_cash = self.equity * rf
        stop_dist = abs(entry - stop)
        if stop_dist <= 0:
            return 0.0
        qty = risk_cash / stop_dist
        max_qty = (self.equity * MAX_POSITION_FRAC) / entry
        return max(0.0, min(qty, max_qty))

    def mark_equity(self, price: float) -> float:
        if self.pos in (1, 3, 5):
            return self.cash + self.qty * price
        if self.pos in (2, 4, 6):
            return self.cash - self.qty * price
        return self.cash


def run_backtest(d: pd.DataFrame) -> Engine:
    eng = Engine()
    idx = d.index
    highs = d["high"].values
    lows = d["low"].values
    closes = d["close"].values
    atrs = d["atr"].values
    vwap_z = d["vwap_z"].values
    long_a = d["long_sig_a"].fillna(False).values
    short_a = d["short_sig_a"].fillna(False).values
    long_b = d["long_sig_b"].fillna(False).values
    short_b = d["short_sig_b"].fillna(False).values
    long_c = d["long_sig_c"].fillna(False).values if "long_sig_c" in d.columns else np.zeros(len(d), dtype=bool)
    short_c = d["short_sig_c"].fillna(False).values if "short_sig_c" in d.columns else np.zeros(len(d), dtype=bool)

    for i in range(len(d)):
        if np.isnan(atrs[i]) or atrs[i] <= 0:
            eng.equity_curve.append(eng.mark_equity(closes[i]))
            continue

        if eng.pos != 0:
            held = i - eng.entry_bar
            can_exit = held >= MIN_HOLD_BARS
            exit_px = None
            reason = ""

            if eng.pos == 1 and can_exit:
                if highs[i] >= eng.entry_price + BE_AT_R * atrs[i]:
                    eng.stop = max(eng.stop, eng.entry_price)
                tp_a = eng.entry_price + ATR_TP_TREND * atrs[eng.entry_bar]
                if USE_FIXED_TP_A and highs[i] >= tp_a:
                    exit_px, reason = tp_a, "A long TP"
                else:
                    eng.trail = max(
                        eng.trail if not math.isnan(eng.trail) else closes[i] - ATR_TRAIL_TREND * atrs[i],
                        closes[i] - ATR_TRAIL_TREND * atrs[i],
                    )
                    eng.stop = max(eng.stop, eng.trail)
                    if lows[i] <= eng.stop:
                        exit_px, reason = eng.stop, "A long stop/trail"
                    elif held >= MAX_HOLD_TREND:
                        exit_px, reason = closes[i], "A long time"

            elif eng.pos == 2 and can_exit:
                if lows[i] <= eng.entry_price - BE_AT_R * atrs[i]:
                    eng.stop = min(eng.stop, eng.entry_price)
                tp_a = eng.entry_price - ATR_TP_TREND * atrs[eng.entry_bar]
                if USE_FIXED_TP_A and lows[i] <= tp_a:
                    exit_px, reason = tp_a, "A short TP"
                else:
                    eng.trail = min(
                        eng.trail if not math.isnan(eng.trail) else closes[i] + ATR_TRAIL_TREND * atrs[i],
                        closes[i] + ATR_TRAIL_TREND * atrs[i],
                    )
                    eng.stop = min(eng.stop, eng.trail)
                    if highs[i] >= eng.stop:
                        exit_px, reason = eng.stop, "A short stop/trail"
                    elif held >= MAX_HOLD_TREND:
                        exit_px, reason = closes[i], "A short time"

            elif eng.pos == 3 and can_exit:
                z_hit = (not np.isnan(vwap_z[i])) and (vwap_z[i] >= -VWAP_Z_EXIT)
                if lows[i] <= eng.stop:
                    exit_px, reason = eng.stop, "B long SL"
                elif (not math.isnan(eng.tp) and highs[i] >= eng.tp) or z_hit:
                    exit_px, reason = (eng.tp if not math.isnan(eng.tp) and highs[i] >= eng.tp else closes[i]), "B long TP"
                elif held >= MAX_HOLD_RANGE:
                    exit_px, reason = closes[i], "B long time"

            elif eng.pos == 4 and can_exit:
                z_hit = (not np.isnan(vwap_z[i])) and (vwap_z[i] <= VWAP_Z_EXIT)
                if highs[i] >= eng.stop:
                    exit_px, reason = eng.stop, "B short SL"
                elif (not math.isnan(eng.tp) and lows[i] <= eng.tp) or z_hit:
                    exit_px, reason = (eng.tp if not math.isnan(eng.tp) and lows[i] <= eng.tp else closes[i]), "B short TP"
                elif held >= MAX_HOLD_RANGE:
                    exit_px, reason = closes[i], "B short time"

            elif eng.pos == 5 and can_exit:  # long breakout C
                if highs[i] >= eng.entry_price + BE_AT_R * atrs[i]:
                    eng.stop = max(eng.stop, eng.entry_price)
                tp_c = eng.entry_price + ATR_TP_BREAK * atrs[eng.entry_bar]
                if highs[i] >= tp_c:
                    exit_px, reason = tp_c, "C long TP breakout"
                else:
                    eng.trail = max(
                        eng.trail if not math.isnan(eng.trail) else closes[i] - ATR_TRAIL_BREAK * atrs[i],
                        closes[i] - ATR_TRAIL_BREAK * atrs[i],
                    )
                    eng.stop = max(eng.stop, eng.trail)
                    if lows[i] <= eng.stop:
                        exit_px, reason = eng.stop, "C long trail"
                    elif held >= MAX_HOLD_BREAK:
                        exit_px, reason = closes[i], "C long time"

            elif eng.pos == 6 and can_exit:  # short breakout C
                if lows[i] <= eng.entry_price - BE_AT_R * atrs[i]:
                    eng.stop = min(eng.stop, eng.entry_price)
                tp_c = eng.entry_price - ATR_TP_BREAK * atrs[eng.entry_bar]
                if lows[i] <= tp_c:
                    exit_px, reason = tp_c, "C short TP breakout"
                else:
                    eng.trail = min(
                        eng.trail if not math.isnan(eng.trail) else closes[i] + ATR_TRAIL_BREAK * atrs[i],
                        closes[i] + ATR_TRAIL_BREAK * atrs[i],
                    )
                    eng.stop = min(eng.stop, eng.trail)
                    if highs[i] >= eng.stop:
                        exit_px, reason = eng.stop, "C short trail"
                    elif held >= MAX_HOLD_BREAK:
                        exit_px, reason = closes[i], "C short time"

            if exit_px is not None:
                side = "long" if eng.pos in (1, 3, 5) else "short"
                px = eng.slip_price(float(exit_px), side, is_entry=False)
                fee = eng.fee(eng.qty * px)
                if side == "long":
                    proceeds = eng.qty * px
                    pnl = proceeds - eng.qty * eng.entry_price - fee - eng.fee(eng.qty * eng.entry_price)
                    eng.cash = eng.cash + proceeds - fee
                else:
                    pnl = eng.qty * (eng.entry_price - px) - fee - eng.fee(eng.qty * eng.entry_price)
                    eng.cash = eng.cash - eng.qty * px - fee
                t = eng.trades[-1]
                t.exit_time = idx[i]
                t.exit_price = px
                t.pnl = pnl
                t.reason = reason
                eng.equity = eng.cash
                eng.pos = 0
                eng.qty = 0.0
                eng.last_exit_bar = i
                eng.tp = math.nan
                eng.trail = math.nan
                eng.risk_frac = RISK_PER_TRADE

        # cooldown: shorter after breakouts to catch series of moves
        cd = COOLDOWN_BREAK if eng.playbook == "C" else COOLDOWN_BARS
        can_trade = eng.pos == 0 and (i - eng.last_exit_bar >= cd)
        if can_trade:
            side = None
            play = None
            stop = tp = trail = math.nan
            risk_frac = RISK_PER_TRADE
            prefer_short = bool(d["daily_dn"].iloc[i]) if "daily_dn" in d.columns else False
            prefer_long = bool(d["daily_up"].iloc[i]) if "daily_up" in d.columns else False

            def try_long_c():
                nonlocal side, play, stop, trail, risk_frac
                if long_c[i]:
                    side, play = "long", "C"
                    stop = closes[i] - ATR_SL_BREAK * atrs[i]
                    trail = closes[i] - ATR_TRAIL_BREAK * atrs[i]
                    risk_frac = RISK_BREAKOUT
                    eng.pos = 5
                    return True
                return False

            def try_short_c():
                nonlocal side, play, stop, trail, risk_frac
                if short_c[i]:
                    side, play = "short", "C"
                    stop = closes[i] + ATR_SL_BREAK * atrs[i]
                    trail = closes[i] + ATR_TRAIL_BREAK * atrs[i]
                    risk_frac = RISK_BREAKOUT
                    eng.pos = 6
                    return True
                return False

            def try_long_a():
                nonlocal side, play, stop, trail, risk_frac
                if long_a[i]:
                    side, play = "long", "A"
                    stop = closes[i] - ATR_SL_TREND * atrs[i]
                    trail = closes[i] - ATR_TRAIL_TREND * atrs[i]
                    risk_frac = RISK_PER_TRADE
                    eng.pos = 1
                    return True
                return False

            def try_short_a():
                nonlocal side, play, stop, trail, risk_frac
                if short_a[i]:
                    side, play = "short", "A"
                    stop = closes[i] + ATR_SL_TREND * atrs[i]
                    trail = closes[i] + ATR_TRAIL_TREND * atrs[i]
                    risk_frac = RISK_PER_TRADE
                    eng.pos = 2
                    return True
                return False

            def try_long_b():
                nonlocal side, play, stop, tp, risk_frac
                if long_b[i]:
                    side, play = "long", "B"
                    stop = closes[i] - ATR_SL_RANGE * atrs[i]
                    tp = closes[i] + ATR_TP_RANGE * atrs[i]
                    risk_frac = RISK_PER_TRADE
                    eng.pos = 3
                    return True
                return False

            def try_short_b():
                nonlocal side, play, stop, tp, risk_frac
                if short_b[i]:
                    side, play = "short", "B"
                    stop = closes[i] + ATR_SL_RANGE * atrs[i]
                    tp = closes[i] - ATR_TP_RANGE * atrs[i]
                    risk_frac = RISK_PER_TRADE
                    eng.pos = 4
                    return True
                return False

            # Breakout C first (higher return path), then A/B
            if prefer_short and not prefer_long:
                try_short_c() or try_short_a() or try_short_b() or try_long_c() or try_long_a() or try_long_b()
            elif prefer_long and not prefer_short:
                try_long_c() or try_long_a() or try_long_b() or try_short_c() or try_short_a() or try_short_b()
            else:
                if short_c[i] or short_a[i] or short_b[i]:
                    try_short_c() or try_short_a() or try_short_b()
                else:
                    try_long_c() or try_long_a() or try_long_b()

            if side is not None:
                entry = eng.slip_price(float(closes[i]), side, is_entry=True)
                qty = eng.size_from_stop(entry, float(stop), risk_frac)
                if qty <= 0:
                    eng.pos = 0
                else:
                    fee = eng.fee(qty * entry)
                    if side == "long":
                        cost = qty * entry + fee
                        if cost > eng.cash:
                            qty = max(0.0, (eng.cash - fee) / entry)
                            fee = eng.fee(qty * entry)
                            cost = qty * entry + fee
                        eng.cash -= cost
                    else:
                        eng.cash += qty * entry - fee
                    eng.entry_bar = i
                    eng.entry_price = entry
                    eng.stop = float(stop)
                    eng.tp = float(tp) if not (isinstance(tp, float) and math.isnan(tp)) else math.nan
                    eng.trail = float(trail) if not (isinstance(trail, float) and math.isnan(trail)) else math.nan
                    eng.qty = qty
                    eng.playbook = play
                    eng.risk_frac = risk_frac
                    eng.equity = eng.mark_equity(entry)
                    eng.trades.append(
                        Trade(
                            side=side,
                            playbook=play,
                            entry_time=idx[i],
                            entry_price=entry,
                            qty=qty,
                        )
                    )

        eng.equity = eng.mark_equity(closes[i])
        eng.equity_curve.append(eng.equity)

    if eng.pos != 0 and eng.trades:
        px = eng.slip_price(float(closes[-1]), "long" if eng.pos in (1, 3, 5) else "short", is_entry=False)
        fee = eng.fee(eng.qty * px)
        t = eng.trades[-1]
        if eng.pos in (1, 3, 5):
            eng.cash += eng.qty * px - fee
            t.pnl = eng.qty * (px - eng.entry_price) - fee - eng.fee(eng.qty * eng.entry_price)
        else:
            eng.cash -= eng.qty * px + fee
            t.pnl = eng.qty * (eng.entry_price - px) - fee - eng.fee(eng.qty * eng.entry_price)
        t.exit_time = idx[-1]
        t.exit_price = px
        t.reason = "EOD flatten"
        eng.pos = 0
        eng.equity = eng.cash
        eng.equity_curve[-1] = eng.equity

    return eng


def summarize(eng: Engine, d: pd.DataFrame) -> None:
    start = STARTING_CAPITAL
    end = eng.equity
    ret = (end / start - 1.0) * 100.0
    closed = [t for t in eng.trades if t.exit_time is not None]
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    pnls = np.array([t.pnl for t in closed]) if closed else np.array([0.0])
    eq = pd.Series(eng.equity_curve, index=d.index[: len(eng.equity_curve)])
    dd = (eq / eq.cummax() - 1.0).min() * 100.0
    bh_start = d["close"].iloc[50]
    bh_end = d["close"].iloc[-1]
    bh_ret = (bh_end / bh_start - 1.0) * 100.0

    print("=" * 60)
    print("BTC Regime Gate v2.6 — BACKTEST (VPIN + regime)")
    print("=" * 60)
    print(f"Trade mode:        {TRADE_MODE}")
    if USE_VPIN_FILTER:
        print("VPIN filter:       ON (stand aside when flow toxic)")
    else:
        print("VPIN filter:       OFF")
    if USE_ML_FILTER:
        print(f"ML filter:         ON  thr_long={ML_THR_LONG:.2f} thr_short={ML_THR_SHORT:.2f}")
        print(f"ML model:          {ML_MODEL_PATH}")
    else:
        print("ML filter:         OFF")
    print(f"Bars (15m):        {len(d):,}")
    print(f"Period:            {d.index[0]} → {d.index[-1]}")
    print(f"Starting capital:  ${start:,.2f}")
    print(f"Ending equity:     ${end:,.2f}")
    print(f"Net profit:        ${end - start:,.2f}")
    print(f"Return:            {ret:.2f}%")
    print(f"Buy & hold (approx from bar 50): {bh_ret:.2f}%")
    print(f"Max drawdown:      {dd:.2f}%")
    print(f"Total trades:      {len(closed)}")
    print(f"Wins / Losses:     {len(wins)} / {len(losses)}")
    if closed:
        wr = 100.0 * len(wins) / len(closed)
        print(f"Win rate:          {wr:.1f}%")
        print(f"Avg trade PnL:     ${pnls.mean():,.2f}")
        avg_w = np.mean([t.pnl for t in wins]) if wins else 0.0
        avg_l = np.mean([t.pnl for t in losses]) if losses else 0.0
        print(f"Avg WIN:           ${avg_w:,.2f}")
        print(f"Avg LOSS:          ${avg_l:,.2f}")
        if avg_l != 0:
            print(f"|Avg win / Avg loss|: {abs(avg_w / avg_l):.2f}x")
        print(f"Best trade:        ${pnls.max():,.2f}")
        print(f"Worst trade:       ${pnls.min():,.2f}")
        by_pb = {}
        for t in closed:
            by_pb.setdefault(t.playbook, []).append(t.pnl)
        for pb, vals in by_pb.items():
            print(f"Playbook {pb}: n={len(vals)}, PnL=${sum(vals):,.2f}")
        longs = [t for t in closed if t.side == "long"]
        shorts = [t for t in closed if t.side == "short"]
        print(f"LONG  (buy low → sell high):  n={len(longs)}, PnL=${sum(t.pnl for t in longs):,.2f}")
        print(f"SHORT (sell high → buy low):  n={len(shorts)}, PnL=${sum(t.pnl for t in shorts):,.2f}")
    print("=" * 60)


def run_on_dataframe(feats: pd.DataFrame):
    """Run engine + return summary dict."""
    eng = run_backtest(feats)
    closed = [t for t in eng.trades if t.exit_time is not None]
    wins = [t for t in closed if t.pnl > 0]
    losses = [t for t in closed if t.pnl <= 0]
    eq = pd.Series(eng.equity_curve, index=feats.index[: len(eng.equity_curve)])
    dd = float((eq / eq.cummax() - 1.0).min() * 100.0) if len(eq) else 0.0
    bh = float((feats["close"].iloc[-1] / feats["close"].iloc[0] - 1.0) * 100.0)
    return {
        "end": eng.equity,
        "ret": (eng.equity / STARTING_CAPITAL - 1.0) * 100.0,
        "n": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "wr": (100.0 * len(wins) / len(closed)) if closed else 0.0,
        "dd": dd,
        "bh": bh,
        "eng": eng,
        "closed": closed,
    }


def compare_modes(df15, df1h, df1d) -> None:
    """Test all trade modes on the loaded dataset (for new-data experiments)."""
    global TRADE_MODE
    print("\n" + "=" * 70)
    tags = []
    if USE_VPIN_FILTER:
        tags.append("VPIN")
    if USE_ML_FILTER:
        tags.append("ML")
    ml_tag = (" + " + "+".join(tags)) if tags else ""
    print(f"COMPARE MODES{ml_tag} on this dataset (start $100,000)")
    print("=" * 70)
    print(f"{'mode':12s} {'end equity':>12s} {'return':>8s} {'n':>5s} {'WR%':>6s} {'maxDD%':>8s} {'B&H%':>8s}")
    ml_proba = score_ml_proba(df15, ML_MODEL_PATH) if USE_ML_FILTER else None
    best = None
    for mode in ("short_only", "long_only", "auto", "both"):
        TRADE_MODE = mode
        feats = prepare_features(df15, df1h, df1d, ml_proba=ml_proba)
        s = run_on_dataframe(feats)
        print(
            f"{mode:12s} ${s['end']:10,.2f} {s['ret']:7.2f}% {s['n']:5d} {s['wr']:6.1f} {s['dd']:7.2f} {s['bh']:7.2f}"
        )
        if best is None or s["ret"] > best[1]:
            best = (mode, s["ret"], s["end"])
    print("=" * 70)
    if best:
        print(f"Best mode on THIS file: {best[0]} → ${best[2]:,.2f} ({best[1]:+.2f}%)")
    print("=" * 70)


def prepare_features(
    df15: pd.DataFrame,
    df1h: pd.DataFrame,
    df1d: pd.DataFrame,
    ml_proba: pd.Series | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    feats = build_features(df15, df1h, df1d).iloc[250:].copy()
    if USE_VPIN_FILTER:
        from vpin_volume import apply_vpin_filter

        feats = apply_vpin_filter(feats, df15)
    if USE_ML_FILTER:
        feats = apply_ml_filter(
            feats, df15, ML_MODEL_PATH, ML_THR_LONG, ML_THR_SHORT, proba=ml_proba
        )
    if year is not None:
        feats = feats[feats.index.year == year].copy()
        if len(feats) < 500:
            raise ValueError(f"Too few bars for year {year}: {len(feats)}")
    return feats


def run_years(df15, df1h, df1d, years: list[int]) -> pd.DataFrame:
    """Run the active strategy independently on each calendar year (fresh $100k each)."""
    tags = []
    if USE_VPIN_FILTER:
        tags.append("VPIN")
    if USE_ML_FILTER:
        tags.append("ML")
    tag = "+".join(tags) if tags else "plain"
    print("\n" + "=" * 78)
    print(f"YEARLY RUNS — mode={TRADE_MODE} | filters={tag} | start $100,000 each year")
    print("=" * 78)
    print(
        f"{'year':>6} {'end equity':>12} {'profit':>12} {'return':>8} {'n':>5} "
        f"{'WR%':>6} {'maxDD%':>8} {'B&H%':>8}"
    )
    ml_proba = score_ml_proba(df15, ML_MODEL_PATH) if USE_ML_FILTER else None
    # build once (VPIN/ML on full history), then slice years
    base = prepare_features(df15, df1h, df1d, ml_proba=ml_proba, year=None)
    rows = []
    for y in years:
        feats = base[base.index.year == y].copy()
        if len(feats) < 500:
            print(f"{y:>6}  SKIP (too few bars: {len(feats)})")
            continue
        s = run_on_dataframe(feats)
        profit = s["end"] - STARTING_CAPITAL
        print(
            f"{y:6d} ${s['end']:10,.2f} ${profit:10,.2f} {s['ret']:7.2f}% {s['n']:5d} "
            f"{s['wr']:6.1f} {s['dd']:7.2f} {s['bh']:7.2f}"
        )
        rows.append(
            {
                "year": y,
                "mode": TRADE_MODE,
                "vpin": USE_VPIN_FILTER,
                "ml": USE_ML_FILTER,
                "end_equity": s["end"],
                "profit_usd": profit,
                "return_pct": s["ret"],
                "trades": s["n"],
                "win_rate": s["wr"],
                "max_dd_pct": s["dd"],
                "buy_hold_pct": s["bh"],
            }
        )
    print("=" * 78)
    out = pd.DataFrame(rows)
    if len(out):
        print(
            f"SUM profit across years: ${out['profit_usd'].sum():+,.2f}  "
            f"| avg year ret: {out['return_pct'].mean():+.2f}%  "
            f"| years beaten B&H: {int((out['return_pct'] > out['buy_hold_pct']).sum())}/{len(out)}"
        )
        path = os.path.join(os.getcwd(), "backtest_yearly_summary.csv")
        out.to_csv(path, index=False)
        print("Saved →", path)
    print("=" * 78)
    return out


def main():
    global DATA_PATH, TRADE_MODE, USE_ML_FILTER, ML_THR_LONG, ML_THR_SHORT, ML_MODEL_PATH, USE_VPIN_FILTER

    parser = argparse.ArgumentParser(description="BTC Regime Gate backtest (Excel or 15m CSV)")
    parser.add_argument("--data", type=str, default=None, help="Path to BTCUSDT_*.xlsx or *.csv")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["short_only", "long_only", "auto", "both"],
        help="Trade mode (default: config TRADE_MODE)",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run all modes and print comparison table (good for new data)",
    )
    parser.add_argument(
        "--ml",
        action="store_true",
        help="Enable ML pattern filter (models_btc15m/btc15m_model.joblib)",
    )
    parser.add_argument(
        "--ml-model",
        type=str,
        default=DEFAULT_ML_MODEL,
        help="Path to joblib model bundle",
    )
    parser.add_argument("--ml-thr-long", type=float, default=ML_THR_LONG, help="Min P(up) for longs")
    parser.add_argument("--ml-thr-short", type=float, default=ML_THR_SHORT, help="Max P(up) for shorts")
    parser.add_argument(
        "--vpin",
        action="store_true",
        default=None,
        help="Enable VPIN toxicity filter (default ON in v2.6)",
    )
    parser.add_argument(
        "--no-vpin",
        action="store_true",
        help="Disable VPIN toxicity filter",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=None,
        help="Optional: keep only this calendar year after features",
    )
    parser.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma years to run separately, e.g. 2019,2020,2021,2022,2024,2025",
    )
    args, _unknown = parser.parse_known_args()

    if args.mode:
        TRADE_MODE = args.mode
    USE_ML_FILTER = bool(args.ml)
    ML_MODEL_PATH = args.ml_model
    ML_THR_LONG = float(args.ml_thr_long)
    ML_THR_SHORT = float(args.ml_thr_short)
    if args.no_vpin:
        USE_VPIN_FILTER = False
    elif args.vpin:
        USE_VPIN_FILTER = True
    # else keep module default (True)

    DATA_PATH = resolve_data_path(args.data)
    print("Loading", DATA_PATH)
    raw = load_ohlcv(DATA_PATH)
    print(f"Raw bars: {len(raw):,}")
    print(f"Period: {raw.index[0]} → {raw.index[-1]}")
    df15, df1h, df1d = prepare_timeframes(raw)
    print(f"15m bars: {len(df15):,}")

    if args.years:
        years = [int(x.strip()) for x in args.years.split(",") if x.strip()]
        run_years(df15, df1h, df1d, years)
        return

    if args.compare:
        compare_modes(df15, df1h, df1d)
        return

    extras = []
    if USE_VPIN_FILTER:
        extras.append("VPIN")
    if USE_ML_FILTER:
        extras.append("ML")
    print(f"Running single mode={TRADE_MODE}" + ((" + " + "+".join(extras)) if extras else ""))
    feats = prepare_features(df15, df1h, df1d, year=args.year)
    if args.year is not None:
        print(f"Filtered to year {args.year}: {len(feats):,} bars")
    eng = run_backtest(feats)
    summarize(eng, feats)

    rows = [
        {
            "side": t.side,
            "playbook": t.playbook,
            "entry_time": t.entry_time,
            "entry_price": t.entry_price,
            "exit_time": t.exit_time,
            "exit_price": t.exit_price,
            "qty": t.qty,
            "pnl": t.pnl,
            "reason": t.reason,
        }
        for t in eng.trades
        if t.exit_time is not None
    ]
    out = pd.DataFrame(rows)
    out_path = os.path.join(os.getcwd(), "backtest_trades.csv")
    out.to_csv(out_path, index=False)
    print("Trades saved →", out_path)

    eq = pd.Series(eng.equity_curve, index=feats.index[: len(eng.equity_curve)])
    eq_path = os.path.join(os.getcwd(), "backtest_equity.csv")
    eq.to_csv(eq_path, header=["equity"])
    print("Equity curve →", eq_path)


if __name__ == "__main__":
    main()
