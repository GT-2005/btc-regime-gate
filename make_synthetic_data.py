#!/usr/bin/env python3
"""
make_synthetic_data.py — generates a fake OHLCV file for smoke-testing.

IMPORTANT: this is NOT market data. It is a regime-switching random walk
designed to exercise every branch of the pipeline. Backtest numbers produced
on it are meaningless as evidence about anything. Its only job is to let you
confirm the code runs before you point it at real candles.

    python make_synthetic_data.py --rows 12000 --out data/synthetic_sample.xlsx
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def generate(n: int = 12000, seed: int = 7, start_price: float = 30000.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # three regimes with different drift and volatility, switching slowly
    drift = np.array([-0.00035, 0.0, 0.00045])
    vol = np.array([0.011, 0.005, 0.008])
    trans = np.array([
        [0.988, 0.010, 0.002],
        [0.008, 0.984, 0.008],
        [0.002, 0.010, 0.988],
    ])

    state = 1
    states = np.empty(n, dtype=int)
    rets = np.empty(n)

    for t in range(n):
        state = rng.choice(3, p=trans[state])
        states[t] = state
        # fat tails: mostly normal, occasionally a jump
        shock = rng.normal(0, vol[state])
        if rng.random() < 0.004:
            shock *= rng.uniform(3, 7) * rng.choice([-1, 1])
        rets[t] = drift[state] + shock

    close = start_price * np.exp(np.cumsum(rets))

    # build believable OHLC around each close
    spread = np.abs(rng.normal(0, 0.0035, n)) * close
    open_ = np.empty(n)
    open_[0] = start_price
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) + spread * rng.uniform(0.2, 1.0, n)
    low = np.minimum(open_, close) - spread * rng.uniform(0.2, 1.0, n)

    # volume correlates with absolute return, as it does in reality
    base_vol = 900 + 240 * np.abs(rets) / (np.abs(rets).mean() + 1e-12)
    volume = base_vol * rng.lognormal(0, 0.42, n)

    dates = pd.date_range("2023-01-01", periods=n, freq="h", tz=None)

    return pd.DataFrame({
        "date": dates,
        "open": np.round(open_, 2),
        "high": np.round(high, 2),
        "low": np.round(low, 2),
        "close": np.round(close, 2),
        "volume": np.round(volume, 4),
    })


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="data/synthetic_sample.xlsx")
    a = ap.parse_args()

    df = generate(a.rows, a.seed)
    if a.out.endswith(".csv"):
        df.to_csv(a.out, index=False)
    else:
        df.to_excel(a.out, index=False)

    print(f"Wrote {len(df):,} synthetic bars to {a.out}")
    print(f"Price ran {df['close'].iloc[0]:,.0f} -> {df['close'].iloc[-1]:,.0f}")
    print("\nReminder: this is generated data, not real candles.")
