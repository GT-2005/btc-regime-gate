"""
ML Pattern Strategy — train on history BEFORE 2025, test full year 2025.

Honest OOS: model never sees 2025 labels/prices during training.
Strategy: enter long when P(up) high, short when P(up) low; ATR stop/TP; fees+slip.

  DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib python3 backtest_ml_2025.py
  python3 backtest_ml_2025.py --mode short_only --risk 0.02
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import joblib
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from train_btc_patterns import (  # noqa: E402
    HAS_LGB,
    build_features,
    load_clean,
    make_model,
    select_feature_columns,
    triple_barrier_labels,
)

DATA_PATH = "/Users/ganeshtilekar/BTC_DATA/BTCUSDT_15m_All.csv"
OUT_DIR = os.path.join(_HERE, "models_btc15m")
TEST_YEAR = 2025
STARTING_CAPITAL = 100_000.0
RISK_PER_TRADE = 0.02
MAX_POSITION_FRAC = 0.25
SLIPPAGE_BPS = 9.0
FEE_BPS = 4.0
ATR_SL = 1.0
ATR_TP = 3.0
MAX_HOLD = 64
COOLDOWN = 8
THR_LONG = 0.60
THR_SHORT = 0.40
EMBARGO = 16


def fee(notional: float) -> float:
    return notional * (FEE_BPS / 10_000.0)


def slip(price: float, side: str, is_entry: bool) -> float:
    half = SLIPPAGE_BPS / 10_000.0 / 2.0
    if side == "long":
        return price * (1 + half) if is_entry else price * (1 - half)
    return price * (1 - half) if is_entry else price * (1 + half)


def run_ml_strategy(
    df: pd.DataFrame,
    proba: pd.Series,
    mode: str = "both",
    thr_long: float = THR_LONG,
    thr_short: float = THR_SHORT,
    risk: float = RISK_PER_TRADE,
) -> dict:
    cash = STARTING_CAPITAL
    equity = STARTING_CAPITAL
    pos = 0  # 1 long, -1 short
    qty = 0.0
    entry = 0.0
    stop = 0.0
    tp = 0.0
    entry_i = -1
    last_exit = -10_000
    trades: list[dict] = []
    eq_curve: list[float] = []

    closes = df["close"].to_numpy(float)
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    atrs = df["atr_14"].to_numpy(float)
    probs = proba.reindex(df.index).to_numpy(float)
    idx = df.index

    allow_long = mode in ("both", "long_only", "auto")
    allow_short = mode in ("both", "short_only", "auto")

    for i in range(len(df)):
        a = atrs[i]
        if not np.isfinite(a) or a <= 0 or not np.isfinite(probs[i]):
            eq_curve.append(equity)
            continue

        # manage open position
        if pos == 1:
            hit_sl = lows[i] <= stop
            hit_tp = highs[i] >= tp
            timeout = (i - entry_i) >= MAX_HOLD
            if hit_sl or hit_tp or timeout:
                raw = stop if hit_sl else (tp if hit_tp else closes[i])
                px = slip(float(raw), "long", is_entry=False)
                f = fee(qty * px) + fee(qty * entry)
                pnl = qty * (px - entry) - f
                cash += qty * px - fee(qty * px)
                trades.append(
                    {
                        "side": "long",
                        "entry_time": idx[entry_i],
                        "exit_time": idx[i],
                        "entry": entry,
                        "exit": px,
                        "qty": qty,
                        "pnl": pnl,
                        "reason": "SL" if hit_sl else ("TP" if hit_tp else "time"),
                        "p_up": float(probs[entry_i]),
                    }
                )
                pos, qty = 0, 0.0
                last_exit = i
                equity = cash
        elif pos == -1:
            hit_sl = highs[i] >= stop
            hit_tp = lows[i] <= tp
            timeout = (i - entry_i) >= MAX_HOLD
            if hit_sl or hit_tp or timeout:
                raw = stop if hit_sl else (tp if hit_tp else closes[i])
                px = slip(float(raw), "short", is_entry=False)
                f = fee(qty * px) + fee(qty * entry)
                pnl = qty * (entry - px) - f
                cash -= qty * px + fee(qty * px)
                trades.append(
                    {
                        "side": "short",
                        "entry_time": idx[entry_i],
                        "exit_time": idx[i],
                        "entry": entry,
                        "exit": px,
                        "qty": qty,
                        "pnl": pnl,
                        "reason": "SL" if hit_sl else ("TP" if hit_tp else "time"),
                        "p_up": float(probs[entry_i]),
                    }
                )
                pos, qty = 0, 0.0
                last_exit = i
                equity = cash

        # new entries
        if pos == 0 and i > 250 and (i - last_exit) >= COOLDOWN:
            p = probs[i]
            want_long = allow_long and p >= thr_long
            want_short = allow_short and p <= thr_short
            # auto: prefer short if p very low, else long if high
            if mode == "auto":
                want_long = p >= thr_long
                want_short = p <= thr_short

            if want_short and (not want_long or p < (1 - thr_long)):
                entry = slip(float(closes[i]), "short", is_entry=True)
                stop = entry + ATR_SL * a
                tp = entry - ATR_TP * a
                risk_cash = equity * risk
                stop_dist = max(stop - entry, 1e-9)
                qty = min(risk_cash / stop_dist, (equity * MAX_POSITION_FRAC) / entry)
                if qty > 0:
                    cash += qty * entry - fee(qty * entry)
                    pos, entry_i = -1, i
            elif want_long:
                entry = slip(float(closes[i]), "long", is_entry=True)
                stop = entry - ATR_SL * a
                tp = entry + ATR_TP * a
                risk_cash = equity * risk
                stop_dist = max(entry - stop, 1e-9)
                qty = min(risk_cash / stop_dist, (equity * MAX_POSITION_FRAC) / entry)
                if qty > 0:
                    cost = qty * entry + fee(qty * entry)
                    if cost > cash:
                        qty = max(0.0, (cash - fee(qty * entry)) / entry)
                        cost = qty * entry + fee(qty * entry)
                    if qty > 0:
                        cash -= cost
                        pos, entry_i = 1, i

        if pos == 1:
            equity = cash + qty * closes[i]
        elif pos == -1:
            equity = cash - qty * closes[i]
        else:
            equity = cash
        eq_curve.append(equity)

    # flatten
    if pos != 0 and trades is not None:
        px = slip(float(closes[-1]), "long" if pos == 1 else "short", is_entry=False)
        if pos == 1:
            pnl = qty * (px - entry) - fee(qty * px) - fee(qty * entry)
            cash += qty * px - fee(qty * px)
        else:
            pnl = qty * (entry - px) - fee(qty * px) - fee(qty * entry)
            cash -= qty * px + fee(qty * px)
        trades.append(
            {
                "side": "long" if pos == 1 else "short",
                "entry_time": idx[entry_i],
                "exit_time": idx[-1],
                "entry": entry,
                "exit": px,
                "qty": qty,
                "pnl": pnl,
                "reason": "EOD",
                "p_up": float(probs[entry_i]),
            }
        )
        equity = cash

    pnls = np.array([t["pnl"] for t in trades]) if trades else np.array([0.0])
    eq = pd.Series(eq_curve)
    dd = float((eq / eq.cummax() - 1.0).min() * 100.0) if len(eq) else 0.0
    bh = float((closes[-1] / closes[0] - 1.0) * 100.0)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    return {
        "end_equity": float(equity),
        "return_pct": float((equity / STARTING_CAPITAL - 1.0) * 100.0),
        "profit_usd": float(equity - STARTING_CAPITAL),
        "trades": int(len(trades)),
        "win_rate": float((pnls > 0).mean() * 100.0) if len(trades) else 0.0,
        "max_dd_pct": dd,
        "buy_hold_pct": bh,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "long_pnl": float(sum(t["pnl"] for t in trades if t["side"] == "long")),
        "short_pnl": float(sum(t["pnl"] for t in trades if t["side"] == "short")),
        "n_long": int(sum(1 for t in trades if t["side"] == "long")),
        "n_short": int(sum(1 for t in trades if t["side"] == "short")),
        "trades_list": trades,
        "equity_curve": eq_curve,
    }


def main():
    parser = argparse.ArgumentParser(description="Train pre-2025 ML strategy, test on 2025")
    parser.add_argument("--data", default=DATA_PATH)
    parser.add_argument("--year", type=int, default=TEST_YEAR)
    parser.add_argument(
        "--mode",
        default="both",
        choices=["both", "short_only", "long_only", "auto"],
    )
    parser.add_argument("--risk", type=float, default=RISK_PER_TRADE)
    parser.add_argument("--thr-long", type=float, default=THR_LONG)
    parser.add_argument("--thr-short", type=float, default=THR_SHORT)
    parser.add_argument("--compare", action="store_true", help="Run all modes on test year")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    year = args.year

    print("=" * 70)
    print(f"ML PATTERN STRATEGY — train < {year}, test = {year} (true OOS)")
    print("=" * 70)
    print("Loading", args.data)
    raw = load_clean(args.data)
    print(f"Bars: {len(raw):,} | {raw.index.min()} → {raw.index.max()}")

    print("Building features…")
    feats = build_features(raw)
    print("Labeling (train years only use these labels)…")
    feats["tb_label"] = triple_barrier_labels(feats)
    feats["target"] = (feats["tb_label"] > 0).astype(float)
    feats.loc[feats["tb_label"].isna(), "target"] = np.nan

    feature_cols = select_feature_columns(feats)
    print(f"Features: {len(feature_cols)} | model: {'LightGBM' if HAS_LGB else 'HistGBM'}")

    train = feats[(feats.index.year < year)].dropna(subset=feature_cols + ["target"])
    if len(train) > EMBARGO:
        train = train.iloc[:-EMBARGO]
    test = feats[feats.index.year == year].copy()
    test_x = test.dropna(subset=feature_cols)

    print(f"Train bars: {len(train):,} (years < {year})")
    print(f"Test bars:  {len(test_x):,} ({year})")
    if len(train) < 20_000 or len(test_x) < 5_000:
        raise SystemExit("Not enough train/test bars")

    print("Training model on pre-test history…")
    model = make_model()
    model.fit(train[feature_cols], train["target"])
    model_path = os.path.join(OUT_DIR, f"btc15m_model_pre{year}.joblib")
    joblib.dump({"model": model, "features": feature_cols, "train_end_year": year - 1}, model_path)
    print("Saved", model_path)

    Xte = test_x[feature_cols].replace([np.inf, -np.inf], np.nan)
    Xte = Xte.fillna(train[feature_cols].median(numeric_only=True))
    proba = pd.Series(model.predict_proba(Xte)[:, 1], index=test_x.index, name="ml_p_up")
    test = test.loc[test_x.index].copy()
    test["ml_p_up"] = proba

    modes = ["both", "short_only", "long_only", "auto"] if args.compare else [args.mode]
    rows = []
    best = None
    for mode in modes:
        stats = run_ml_strategy(
            test,
            proba,
            mode=mode,
            thr_long=args.thr_long,
            thr_short=args.thr_short,
            risk=args.risk,
        )
        rows.append((mode, stats))
        tag = f"{mode:12s}"
        print(
            f"{tag}  end=${stats['end_equity']:,.2f}  "
            f"profit=${stats['profit_usd']:+,.2f}  "
            f"ret={stats['return_pct']:+.2f}%  "
            f"trades={stats['trades']:4d}  WR={stats['win_rate']:.1f}%  "
            f"DD={stats['max_dd_pct']:.2f}%  B&H={stats['buy_hold_pct']:.2f}%"
        )
        if best is None or stats["return_pct"] > best[1]["return_pct"]:
            best = (mode, stats)

    mode, stats = best if args.compare else rows[0]
    print("\n" + "=" * 70)
    print(f"SELECTED RESULT — {year} | mode={mode}")
    print("=" * 70)
    print(f"Starting capital:  ${STARTING_CAPITAL:,.2f}")
    print(f"Ending equity:     ${stats['end_equity']:,.2f}")
    print(f"Profit:            ${stats['profit_usd']:+,.2f}")
    print(f"Return:            {stats['return_pct']:+.2f}%")
    print(f"Buy & hold {year}:   {stats['buy_hold_pct']:+.2f}%")
    print(f"Max drawdown:      {stats['max_dd_pct']:.2f}%")
    print(f"Trades:            {stats['trades']}  (WR {stats['win_rate']:.1f}%)")
    print(f"Longs:  n={stats['n_long']}  PnL=${stats['long_pnl']:+,.2f}")
    print(f"Shorts: n={stats['n_short']}  PnL=${stats['short_pnl']:+,.2f}")
    if stats["avg_loss"] != 0:
        print(f"|Avg win / Avg loss|: {abs(stats['avg_win'] / stats['avg_loss']):.2f}x")
    print(f"Risk/trade:        {args.risk*100:.1f}% | SL={ATR_SL} ATR TP={ATR_TP} ATR")
    print(f"Thresholds:        long P>={args.thr_long}  short P<={args.thr_short}")
    print("=" * 70)

    tdf = pd.DataFrame(stats["trades_list"])
    t_path = os.path.join(OUT_DIR, f"ml_strategy_trades_{year}.csv")
    tdf.to_csv(t_path, index=False)
    eq = pd.Series(stats["equity_curve"], index=test.index[: len(stats["equity_curve"])])
    eq_path = os.path.join(OUT_DIR, f"ml_strategy_equity_{year}.csv")
    eq.to_csv(eq_path, header=["equity"])

    report = {
        "test_year": year,
        "train_years": f"<{year}",
        "mode": mode,
        "model": "LightGBM" if HAS_LGB else "HistGBM",
        "n_features": len(feature_cols),
        "train_bars": int(len(train)),
        "test_bars": int(len(test_x)),
        "starting_capital": STARTING_CAPITAL,
        "end_equity": stats["end_equity"],
        "profit_usd": stats["profit_usd"],
        "return_pct": stats["return_pct"],
        "buy_hold_pct": stats["buy_hold_pct"],
        "max_dd_pct": stats["max_dd_pct"],
        "trades": stats["trades"],
        "win_rate": stats["win_rate"],
        "n_long": stats["n_long"],
        "n_short": stats["n_short"],
        "long_pnl": stats["long_pnl"],
        "short_pnl": stats["short_pnl"],
        "thr_long": args.thr_long,
        "thr_short": args.thr_short,
        "risk": args.risk,
        "compare": {m: {k: v for k, v in s.items() if k not in ("trades_list", "equity_curve")} for m, s in rows},
    }
    r_path = os.path.join(OUT_DIR, f"ml_strategy_report_{year}.json")
    with open(r_path, "w") as f:
        json.dump(report, f, indent=2)

    print("Saved:")
    print(" ", model_path)
    print(" ", t_path)
    print(" ", eq_path)
    print(" ", r_path)


if __name__ == "__main__":
    main()
