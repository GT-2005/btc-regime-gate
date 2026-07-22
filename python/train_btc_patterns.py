"""
BTC Pattern Learner — train on full 15m history (2017+)
Rich feature set + walk-forward LightGBM, purged splits, baselines.

Data: /Users/ganeshtilekar/BTC_DATA/BTCUSDT_15m_All.csv
"""

from __future__ import annotations

import json
import os
import warnings
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb

    # verify native lib loads
    _ = lgb.__version__
    HAS_LGB = True
except Exception:
    HAS_LGB = False
    from sklearn.ensemble import HistGradientBoostingClassifier

# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH = "/Users/ganeshtilekar/BTC_DATA/BTCUSDT_15m_All.csv"
OUT_DIR = "/Users/ganeshtilekar/College Project/python/models_btc15m"
STARTING_CAPITAL = 100_000.0
ATR_TP = 2.0
ATR_SL = 1.0
MAX_HOLD = 32
EMBARGO = 16
TRAIN_YEARS_MIN = 2

FEATURE_COLS: list[str] = []


def load_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None)
    df.columns = [
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
    ot = df["open_time"].astype("float64").to_numpy()
    ms = ot.copy()
    ms = np.where(ms > 1e14, ms / 1000.0, ms)
    ms = np.where(ms > 1e14, ms / 1000.0, ms)
    df["timestamp"] = pd.to_datetime(ms, unit="ms", utc=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp")
    df = df[(df["timestamp"] >= "2017-08-01") & (df["timestamp"] <= "2026-12-31")]
    for c in ["open", "high", "low", "close", "volume", "trades", "taker_buy_base"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])
    df = df[(df["high"] >= df["low"]) & (df["close"] > 0) & (df["volume"] >= 0)]
    return df.set_index("timestamp")[["open", "high", "low", "close", "volume", "trades", "taker_buy_base"]]


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
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_w = tr.ewm(alpha=1 / n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr_w.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / atr_w.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean()


def macd(close: pd.Series):
    e12 = ema(close, 12)
    e26 = ema(close, 26)
    line = e12 - e26
    signal = ema(line, 9)
    return line, signal, line - signal


def stoch(df: pd.DataFrame, n: int = 14):
    ll = df["low"].rolling(n).min()
    hh = df["high"].rolling(n).max()
    k = 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)
    d = k.rolling(3).mean()
    return k, d


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Wide pattern feature set — trend, momentum, vol, volume, candles, HTF."""
    d = df.copy()
    c = d["close"]
    h, l, o, v = d["high"], d["low"], d["open"], d["volume"]

    # returns / momentum
    for n in (1, 2, 3, 5, 8, 13, 21, 34, 55, 89):
        d[f"ret_{n}"] = c.pct_change(n)
        d[f"logret_{n}"] = np.log(c / c.shift(n))

    # moving averages + slopes
    for n in (8, 13, 21, 34, 55, 89, 144, 200):
        d[f"ema_{n}"] = ema(c, n)
        d[f"sma_{n}"] = sma(c, n)
        d[f"dist_ema_{n}"] = (c - d[f"ema_{n}"]) / c
        d[f"ema_slope_{n}"] = d[f"ema_{n}"].pct_change(3)

    d["ema_stack_bull"] = (
        (d["ema_8"] > d["ema_21"]).astype(float)
        + (d["ema_21"] > d["ema_55"]).astype(float)
        + (d["ema_55"] > d["ema_144"]).astype(float)
    )
    d["ema_stack_bear"] = (
        (d["ema_8"] < d["ema_21"]).astype(float)
        + (d["ema_21"] < d["ema_55"]).astype(float)
        + (d["ema_55"] < d["ema_144"]).astype(float)
    )

    # volatility
    d["atr_14"] = atr(d, 14)
    d["atr_28"] = atr(d, 28)
    d["atr_pct"] = d["atr_14"] / c
    d["atr_ratio"] = d["atr_14"] / sma(d["atr_14"], 50)
    d["hv_20"] = d["ret_1"].rolling(20).std()
    d["hv_50"] = d["ret_1"].rolling(50).std()
    d["bb_mid"] = sma(c, 20)
    d["bb_std"] = c.rolling(20).std()
    d["bb_upper"] = d["bb_mid"] + 2 * d["bb_std"]
    d["bb_lower"] = d["bb_mid"] - 2 * d["bb_std"]
    d["bb_pctb"] = (c - d["bb_lower"]) / (d["bb_upper"] - d["bb_lower"]).replace(0, np.nan)
    d["bb_width"] = (d["bb_upper"] - d["bb_lower"]) / d["bb_mid"].replace(0, np.nan)

    # oscillators
    d["rsi_7"] = rsi(c, 7)
    d["rsi_14"] = rsi(c, 14)
    d["rsi_21"] = rsi(c, 21)
    macd_l, macd_s, macd_h = macd(c)
    d["macd"] = macd_l
    d["macd_signal"] = macd_s
    d["macd_hist"] = macd_h
    k, s = stoch(d, 14)
    d["stoch_k"] = k
    d["stoch_d"] = s
    d["adx_14"] = adx(d, 14)
    # CCI (vectorized)
    tp = (h + l + c) / 3.0
    tp_sma = sma(tp, 20)
    tp_mad = (tp - tp_sma).abs().rolling(20).mean()
    d["cci_20"] = (tp - tp_sma) / (0.015 * tp_mad.replace(0, np.nan))

    # volume / flow
    d["rvol_20"] = v / sma(v, 20).replace(0, np.nan)
    d["rvol_50"] = v / sma(v, 50).replace(0, np.nan)
    d["trades"] = d["trades"].astype(float)
    d["taker_buy_ratio"] = d["taker_buy_base"] / v.replace(0, np.nan)
    d["obv_proxy"] = (np.sign(c.diff()) * v).fillna(0).cumsum()
    d["obv_slope"] = d["obv_proxy"].pct_change(10).replace([np.inf, -np.inf], np.nan)
    typ = tp
    vol_sum = v.rolling(48).sum()
    d["roll_vwap"] = (typ * v).rolling(48).sum() / vol_sum.replace(0, np.nan)
    d["vwap_z"] = (c - d["roll_vwap"]) / (c - d["roll_vwap"]).rolling(48).std().replace(0, np.nan)

    # candle structure / patterns
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    d["body_pct"] = body / rng
    d["upper_wick"] = (h - np.maximum(c, o)) / rng
    d["lower_wick"] = (np.minimum(c, o) - l) / rng
    d["bull_candle"] = (c > o).astype(float)
    d["bear_candle"] = (c < o).astype(float)
    d["doji"] = (body / rng < 0.1).astype(float)
    d["hammer"] = ((d["lower_wick"] > 0.6) & (d["body_pct"] < 0.3)).astype(float)
    d["shooting_star"] = ((d["upper_wick"] > 0.6) & (d["body_pct"] < 0.3)).astype(float)
    d["bull_engulf"] = ((c > o) & (c.shift(1) < o.shift(1)) & (c >= o.shift(1)) & (o <= c.shift(1))).astype(float)
    d["bear_engulf"] = ((c < o) & (c.shift(1) > o.shift(1)) & (c <= o.shift(1)) & (o >= c.shift(1))).astype(float)
    d["inside_bar"] = ((h <= h.shift(1)) & (l >= l.shift(1))).astype(float)
    d["outside_bar"] = ((h >= h.shift(1)) & (l <= l.shift(1))).astype(float)
    d["gap_up"] = (l > h.shift(1)).astype(float)
    d["gap_down"] = (h < l.shift(1)).astype(float)

    # range breakout distances
    for n in (12, 24, 48, 96):
        d[f"dist_hh_{n}"] = (c - h.rolling(n).max().shift(1)) / d["atr_14"]
        d[f"dist_ll_{n}"] = (c - l.rolling(n).min().shift(1)) / d["atr_14"]

    # higher timeframe context (no lookahead: shift HTF by 1)
    for rule, prefix in [("1h", "h1"), ("4h", "h4"), ("1D", "d1")]:
        htf = (
            d.resample(rule)
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna()
        )
        htf[f"{prefix}_ema50"] = ema(htf["close"], 50)
        htf[f"{prefix}_ema200"] = ema(htf["close"], 200)
        htf[f"{prefix}_trend"] = (htf[f"{prefix}_ema50"] > htf[f"{prefix}_ema200"]).astype(float)
        htf[f"{prefix}_ret"] = htf["close"].pct_change(1)
        htf[f"{prefix}_rsi"] = rsi(htf["close"], 14)
        cols = [f"{prefix}_ema50", f"{prefix}_ema200", f"{prefix}_trend", f"{prefix}_ret", f"{prefix}_rsi"]
        merged = htf[cols].shift(1).reindex(d.index, method="ffill")
        for col in cols:
            d[col] = merged[col]

    # calendar / session (UTC)
    d["hour"] = d.index.hour
    d["dow"] = d.index.dayofweek
    d["month"] = d.index.month
    d["asia"] = ((d["hour"] >= 0) & (d["hour"] < 8)).astype(float)
    d["london"] = ((d["hour"] >= 7) & (d["hour"] < 16)).astype(float)
    d["ny"] = ((d["hour"] >= 13) & (d["hour"] < 22)).astype(float)

    # regime proxies
    d["trend_regime"] = (d["adx_14"] >= 25).astype(float)
    d["range_regime"] = (d["adx_14"] <= 20).astype(float)
    d["vol_shock"] = (d["atr_ratio"] >= 2.5).astype(float)

    return d


def triple_barrier_labels(df: pd.DataFrame, tp_mult=ATR_TP, sl_mult=ATR_SL, max_hold=MAX_HOLD) -> pd.Series:
    """+1 if TP first, -1 if SL first, 0 if time (mapped to binary later)."""
    close = df["close"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    atrv = df["atr_14"].to_numpy()
    n = len(df)
    y = np.full(n, np.nan)
    for i in range(n - max_hold - 1):
        a = atrv[i]
        if not np.isfinite(a) or a <= 0:
            continue
        up = close[i] + tp_mult * a
        dn = close[i] - sl_mult * a
        label = 0.0
        for j in range(1, max_hold + 1):
            if high[i + j] >= up:
                label = 1.0
                break
            if low[i + j] <= dn:
                label = -1.0
                break
        y[i] = label
    return pd.Series(y, index=df.index, name="tb_label")


def select_feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trades",
        "taker_buy_base",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "roll_vwap",
        "obv_proxy",
        "tb_label",
        "target",
    }
    cols = []
    for c in df.columns:
        if c in drop:
            continue
        if c.startswith("ema_") and c.count("_") == 1 and c.split("_")[1].isdigit():
            # keep dist/slope; raw ema level less useful across years
            if not c.startswith("ema_slope") and not c.startswith("dist_"):
                # keep stack features only among emas — skip raw level emas
                if c.startswith("ema_") and c[4:].isdigit():
                    continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


@dataclass
class FoldResult:
    year: int
    auc: float
    acc: float
    n_test: int
    long_prec: float
    feature_top: list


def make_model():
    if HAS_LGB:
        return lgb.LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=48,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=50,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    return HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
    )


def walk_forward_by_year(df: pd.DataFrame, feature_cols: list[str]) -> tuple[list[FoldResult], object]:
    df = df.dropna(subset=feature_cols + ["target"]).copy()
    years = sorted(df.index.year.unique())
    results = []
    last_model = None
    importances = pd.Series(0.0, index=feature_cols)

    print("\nWalk-forward by year (train past → test next year)")
    print("=" * 70)
    for i, y in enumerate(years):
        if i < TRAIN_YEARS_MIN:
            continue
        train = df[df.index.year < y]
        test = df[df.index.year == y]
        if len(train) < 5000 or len(test) < 1000:
            continue
        # embargo: drop last EMBARGO bars of train overlapping label horizon
        if len(train) > EMBARGO:
            train = train.iloc[:-EMBARGO]

        Xtr, ytr = train[feature_cols], train["target"]
        Xte, yte = test[feature_cols], test["target"]
        # balance check
        if ytr.nunique() < 2 or yte.nunique() < 2:
            continue

        model = make_model()
        model.fit(Xtr, ytr)
        proba = model.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.55).astype(int)
        auc = roc_auc_score(yte, proba)
        acc = accuracy_score(yte, pred)
        long_mask = pred == 1
        long_prec = float(yte[long_mask].mean()) if long_mask.sum() else 0.0

        if HAS_LGB and hasattr(model, "feature_importances_"):
            importances += pd.Series(model.feature_importances_, index=feature_cols)
        elif hasattr(model, "feature_importances_"):
            importances += pd.Series(model.feature_importances_, index=feature_cols)

        top = importances.sort_values(ascending=False).head(15)
        fr = FoldResult(y, float(auc), float(acc), int(len(yte)), float(long_prec), list(top.index))
        results.append(fr)
        last_model = model
        print(f"{y}: AUC={auc:.3f} ACC={acc:.3f} n={len(yte):,} long_prec@0.55={long_prec:.3f}")

    return results, last_model, importances


def simple_rule_backtest(df: pd.DataFrame, proba: pd.Series, thr_long=0.60, thr_short=0.40) -> dict:
    """Use model probs as long/short gate with ATR exits — quick economic check."""
    cash = STARTING_CAPITAL
    equity = STARTING_CAPITAL
    pos = 0
    qty = 0.0
    entry = 0.0
    stop = 0.0
    tp = 0.0
    entry_i = -1
    trades = []
    eq_curve = []

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr_14"].values
    probs = proba.reindex(df.index).fillna(0.5).values
    idx = df.index

    for i in range(len(df)):
        if not np.isfinite(atrs[i]) or atrs[i] <= 0:
            eq_curve.append(equity)
            continue
        if pos == 1:
            if lows[i] <= stop or highs[i] >= tp or (i - entry_i) >= MAX_HOLD:
                px = stop if lows[i] <= stop else (tp if highs[i] >= tp else closes[i])
                pnl = qty * (px - entry)
                cash += qty * px
                trades.append(pnl)
                pos = 0
                qty = 0.0
        elif pos == -1:
            if highs[i] >= stop or lows[i] <= tp or (i - entry_i) >= MAX_HOLD:
                px = stop if highs[i] >= stop else (tp if lows[i] <= tp else closes[i])
                pnl = qty * (entry - px)
                cash -= qty * px
                trades.append(pnl)
                pos = 0
                qty = 0.0

        if pos == 0 and i > 200:
            if probs[i] >= thr_long:
                entry = closes[i]
                stop = entry - ATR_SL * atrs[i]
                tp = entry + ATR_TP * atrs[i]
                risk = equity * 0.01
                qty = risk / max(entry - stop, 1e-9)
                qty = min(qty, (equity * 0.2) / entry)
                cash -= qty * entry
                pos = 1
                entry_i = i
            elif probs[i] <= thr_short:
                entry = closes[i]
                stop = entry + ATR_SL * atrs[i]
                tp = entry - ATR_TP * atrs[i]
                risk = equity * 0.01
                qty = risk / max(stop - entry, 1e-9)
                qty = min(qty, (equity * 0.2) / entry)
                cash += qty * entry
                pos = -1
                entry_i = i

        if pos == 1:
            equity = cash + qty * closes[i]
        elif pos == -1:
            equity = cash - qty * closes[i]
        else:
            equity = cash
        eq_curve.append(equity)

    rets = np.array(trades) if trades else np.array([0.0])
    eq = pd.Series(eq_curve)
    dd = float((eq / eq.cummax() - 1).min() * 100)
    return {
        "end_equity": float(equity),
        "return_pct": float((equity / STARTING_CAPITAL - 1) * 100),
        "trades": int(len(trades)),
        "win_rate": float((rets > 0).mean() * 100) if len(trades) else 0.0,
        "max_dd_pct": dd,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Loading", DATA_PATH)
    raw = load_clean(DATA_PATH)
    print(f"Bars: {len(raw):,} | {raw.index.min()} → {raw.index.max()}")

    print("Building features (this can take a few minutes)…")
    feats = build_features(raw)
    print("Labeling triple-barrier…")
    feats["tb_label"] = triple_barrier_labels(feats)
    # binary: win TP path vs not
    feats["target"] = (feats["tb_label"] > 0).astype(float)
    feats.loc[feats["tb_label"].isna(), "target"] = np.nan

    feature_cols = select_feature_columns(feats)
    global FEATURE_COLS
    FEATURE_COLS = feature_cols
    print(f"Feature count: {len(feature_cols)}")

    # save decimated feature matrix (CSV; no pyarrow needed)
    feats[feature_cols + ["target", "tb_label"]].iloc[::50].to_csv(
        os.path.join(OUT_DIR, "features_decimated.csv")
    )

    results, model, importances = walk_forward_by_year(feats, feature_cols)

    # final model on all but last year
    years = sorted(feats.dropna(subset=["target"]).index.year.unique())
    last_year = years[-1]
    train_final = feats.dropna(subset=feature_cols + ["target"])
    train_final = train_final[train_final.index.year < last_year]
    test_final = feats.dropna(subset=feature_cols + ["target"])
    test_final = test_final[test_final.index.year == last_year]

    final_model = make_model()
    final_model.fit(train_final[feature_cols], train_final["target"])
    joblib.dump(
        {"model": final_model, "features": feature_cols, "has_lgb": HAS_LGB},
        os.path.join(OUT_DIR, "btc15m_model.joblib"),
    )

    proba = pd.Series(final_model.predict_proba(test_final[feature_cols])[:, 1], index=test_final.index)
    bt = simple_rule_backtest(test_final, proba)
    print("\nOOS economic check on", last_year)
    print(bt)

    # full-history OOS probs via expanding WF for report
    all_proba = pd.Series(index=feats.index, dtype=float)
    for fr in results:
        y = fr.year
        tr = feats[(feats.index.year < y)].dropna(subset=feature_cols + ["target"])
        te = feats[(feats.index.year == y)].dropna(subset=feature_cols + ["target"])
        if len(tr) < 1000 or len(te) < 500:
            continue
        tr = tr.iloc[:-EMBARGO] if len(tr) > EMBARGO else tr
        m = make_model()
        m.fit(tr[feature_cols], tr["target"])
        all_proba.loc[te.index] = m.predict_proba(te[feature_cols])[:, 1]

    # year-by-year backtest using WF probs
    year_stats = []
    for y in years:
        if y not in [fr.year for fr in results]:
            continue
        sl = feats[feats.index.year == y].dropna(subset=["atr_14"])
        pr = all_proba.reindex(sl.index)
        if pr.notna().sum() < 500:
            continue
        stats = simple_rule_backtest(sl, pr)
        stats["year"] = y
        year_stats.append(stats)
        print(f"Year {y}: ret={stats['return_pct']:.2f}% trades={stats['trades']} WR={stats['win_rate']:.1f}% DD={stats['max_dd_pct']:.2f}%")

    imp = importances.sort_values(ascending=False)
    imp.head(40).to_csv(os.path.join(OUT_DIR, "feature_importance_top40.csv"))
    print("\nTop 20 patterns / features:")
    print(imp.head(20))

    report = {
        "data_path": DATA_PATH,
        "bars": int(len(raw)),
        "start": str(raw.index.min()),
        "end": str(raw.index.max()),
        "n_features": len(feature_cols),
        "model": "LightGBM" if HAS_LGB else "HistGradientBoosting",
        "folds": [fr.__dict__ for fr in results],
        "last_year_backtest": bt,
        "year_stats": year_stats,
        "top_features": list(imp.head(30).index),
    }
    with open(os.path.join(OUT_DIR, "train_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print("\nSaved →", OUT_DIR)
    print("  btc15m_model.joblib")
    print("  feature_importance_top40.csv")
    print("  train_report.json")


if __name__ == "__main__":
    main()
