#!/usr/bin/env python3
"""
fetch_data.py — pull OHLCV candles from Binance's public API.

No account, no API key, no signature. The klines endpoint is public.

    python fetch_data.py                                  # 2 years of BTC/USDT 1h
    python fetch_data.py --symbol ETHUSDT --interval 4h
    python fetch_data.py --start 2021-01-01 --out data/btc_long.xlsx

If Binance is unreachable from where you are, use the bulk archive instead:
https://data.binance.vision  →  spot/monthly/klines/BTCUSDT/1h/
Those are plain CSV zips and need no code at all.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://api.binance.com/api/v3/klines"
LIMIT = 1000                       # max candles per request Binance allows

# how many milliseconds one candle spans, per interval
SPAN_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
    "1h": 3_600_000, "2h": 7_200_000, "4h": 14_400_000, "6h": 21_600_000,
    "8h": 28_800_000, "12h": 43_200_000, "1d": 86_400_000,
}


def fetch(symbol: str, interval: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    """Page through the endpoint until we reach end_ms."""
    if interval not in SPAN_MS:
        raise SystemExit(f"Unsupported interval {interval!r}. Pick from: {', '.join(SPAN_MS)}")

    rows: list[list] = []
    cursor = start_ms
    span = SPAN_MS[interval]

    while cursor < end_ms:
        try:
            r = requests.get(
                BASE,
                params={"symbol": symbol, "interval": interval,
                        "startTime": cursor, "endTime": end_ms, "limit": LIMIT},
                timeout=20,
            )
        except requests.RequestException as e:
            raise SystemExit(
                f"\nCould not reach Binance: {e}\n\n"
                "If Binance is blocked on your network or in your region, download\n"
                "the monthly CSV archives from https://data.binance.vision instead."
            )

        if r.status_code == 429:
            print("  rate limited, waiting 60s...")
            time.sleep(60)
            continue
        if r.status_code == 451:
            raise SystemExit(
                "\nBinance returned 451 — the API is restricted from your location.\n"
                "Use https://data.binance.vision (plain CSV downloads) instead."
            )
        if r.status_code != 200:
            raise SystemExit(f"\nBinance returned {r.status_code}: {r.text[:200]}")

        batch = r.json()
        if not batch:
            break

        rows.extend(batch)
        cursor = batch[-1][0] + span          # next candle after the last one

        got = datetime.fromtimestamp(batch[-1][0] / 1000, timezone.utc)
        print(f"  {len(rows):>7,} candles   up to {got:%Y-%m-%d %H:%M} UTC", end="\r")

        time.sleep(0.25)                      # stay well inside the rate limit

    print()
    if not rows:
        raise SystemExit("Binance returned nothing. Check the symbol spelling.")

    # Binance kline format: [openTime, open, high, low, close, volume, closeTime, ...]
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["date", "open", "high", "low", "close", "volume"]
    df["date"] = pd.to_datetime(df["date"], unit="ms")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c])

    # oldest first, no duplicates — the loader wants this and so does the model
    return (df.sort_values("date")
              .drop_duplicates(subset="date", keep="last")
              .reset_index(drop=True))


def main() -> int:
    ap = argparse.ArgumentParser(description="Download candles from Binance.")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="1h", help="1m 5m 15m 1h 4h 1d ...")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD (default: 2 years ago)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD (default: today)")
    ap.add_argument("--out", default=None, help="output path (.xlsx or .csv)")
    a = ap.parse_args()

    start = (datetime.strptime(a.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
             if a.start else datetime.now(timezone.utc) - timedelta(days=730))
    end = (datetime.strptime(a.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
           if a.end else datetime.now(timezone.utc))

    out = Path(a.out or f"data/{a.symbol}_{a.interval}.xlsx")
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{a.symbol}  {a.interval}   {start:%Y-%m-%d} -> {end:%Y-%m-%d}\n")
    df = fetch(a.symbol, a.interval,
               int(start.timestamp() * 1000), int(end.timestamp() * 1000))

    if out.suffix.lower() == ".csv":
        df.to_csv(out, index=False)
    else:
        df.to_excel(out, index=False)

    print(f"\nWrote {len(df):,} candles to {out}")
    print(f"  {df['date'].iloc[0]}  ->  {df['date'].iloc[-1]}")
    print(f"  close ran {df['close'].iloc[0]:,.2f} -> {df['close'].iloc[-1]:,.2f}")

    gaps = df["date"].diff().dropna()
    if len(gaps):
        odd = int((gaps > gaps.median() * 1.5).sum())
        print(f"  {odd} gap(s) in the series"
              + ("  (normal — exchanges have maintenance windows)" if odd else ""))

    print(f"\nNext:\n  python run_backtest.py --data {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
