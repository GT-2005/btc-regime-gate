"""
Volume clock + VPIN (bar-level proxy) — Easley / López de Prado / O'Hara.

Papers:
  - The Volume Clock (SSRN 2034858)
  - Flow Toxicity and Liquidity / VPIN (SSRN 1695596)

Note: classic VPIN wants tick trades. Here we use 15m (or 1m) bars with
bulk-volume classification Φ(ΔP/σ) as the paper allows for time bars.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def make_volume_bars(df: pd.DataFrame, bucket_vol: float) -> pd.DataFrame:
    """Aggregate OHLCV into equal-volume bars of size bucket_vol."""
    if bucket_vol <= 0:
        raise ValueError("bucket_vol must be > 0")
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float)
    idx = df.index.to_numpy()

    rows = []
    acc = 0.0
    bo = bh = bl = bc = None
    bt = None
    for i in range(len(df)):
        vi = float(v[i])
        if not np.isfinite(vi) or vi < 0:
            continue
        if bo is None:
            bo, bh, bl, bc, bt = o[i], h[i], l[i], c[i], idx[i]
            acc = 0.0
        bh = max(bh, h[i])
        bl = min(bl, l[i])
        bc = c[i]
        rem = vi
        while rem > 0:
            need = bucket_vol - acc
            take = min(need, rem)
            acc += take
            rem -= take
            if acc + 1e-12 >= bucket_vol:
                rows.append(
                    {
                        "timestamp": idx[i],
                        "open": bo,
                        "high": bh,
                        "low": bl,
                        "close": bc,
                        "volume": bucket_vol,
                        "start": bt,
                    }
                )
                acc = 0.0
                if rem > 0:
                    bo = bh = bl = bc = c[i]
                    bt = idx[i]
                    bh = h[i]
                    bl = l[i]
                else:
                    bo = None
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.set_index("timestamp")


def estimate_bucket_size(df: pd.DataFrame, buckets_per_day: float = 50.0) -> float:
    """Paper default vibe: ~50 volume buckets per average day."""
    daily = df["volume"].resample("1D").sum()
    med = float(daily.replace(0, np.nan).median())
    if not np.isfinite(med) or med <= 0:
        med = float(df["volume"].sum() / max(len(daily), 1))
    return max(med / buckets_per_day, 1e-9)


def compute_vpin_on_bars(
    df: pd.DataFrame,
    buckets_per_day: float = 50.0,
    n_buckets: int = 50,
    sigma_window: int = 50,
) -> pd.DataFrame:
    """
    Bar-level VPIN proxy.
    Returns df copy with: buy_vol, sell_vol, bucket_id, vpin, vpin_hi (toxic).
    """
    d = df.copy()
    v = d["volume"].to_numpy(float)
    c = d["close"].to_numpy(float)
    dp = np.diff(c, prepend=c[0])
    # rolling σ of price changes
    sigma = (
        pd.Series(dp, index=d.index)
        .rolling(sigma_window, min_periods=max(10, sigma_window // 5))
        .std()
        .replace(0, np.nan)
        .to_numpy(float)
    )
    z = np.divide(dp, sigma, out=np.zeros_like(dp), where=np.isfinite(sigma) & (sigma > 0))
    z = np.clip(z, -5.0, 5.0)
    buy_frac = norm.cdf(z)
    buy_vol = v * buy_frac
    sell_vol = v * (1.0 - buy_frac)
    d["buy_vol"] = buy_vol
    d["sell_vol"] = sell_vol
    d["imbalance"] = np.abs(buy_vol - sell_vol)

    V = estimate_bucket_size(d, buckets_per_day=buckets_per_day)
    # assign volume buckets
    bucket_ids = np.full(len(d), -1, dtype=int)
    bucket_imb = []
    acc = 0.0
    imb_acc = 0.0
    b_id = 0
    for i in range(len(d)):
        rem_v = v[i]
        rem_imb = abs(buy_vol[i] - sell_vol[i])
        # distribute proportionally if bar spans multiple buckets
        while rem_v > 1e-12:
            need = V - acc
            take = min(need, rem_v)
            frac = take / rem_v if rem_v > 0 else 0.0
            acc += take
            imb_acc += rem_imb * frac
            rem_v -= take
            rem_imb *= 1.0 - frac
            bucket_ids[i] = b_id
            if acc + 1e-12 >= V:
                bucket_imb.append(imb_acc)
                b_id += 1
                acc = 0.0
                imb_acc = 0.0

    d["bucket_id"] = bucket_ids
    # VPIN at each completed bucket, then map to bars
    if len(bucket_imb) < n_buckets:
        d["vpin"] = np.nan
        d["vpin_hi"] = False
        d.attrs["bucket_vol"] = V
        return d

    imb = np.array(bucket_imb, dtype=float)
    # rolling mean of |OI| / V over n buckets == sum(|OI|)/(n*V)
    roll = pd.Series(imb).rolling(n_buckets).sum() / (n_buckets * V)
    # map bucket end → timestamp: last bar of each bucket
    bucket_end_pos = {}
    for i, bid in enumerate(bucket_ids):
        if bid >= 0:
            bucket_end_pos[bid] = i
    vpin_bar = np.full(len(d), np.nan)
    for bid, val in roll.items():
        if bid in bucket_end_pos and np.isfinite(val):
            vpin_bar[bucket_end_pos[bid]] = float(val)
    d["vpin"] = pd.Series(vpin_bar, index=d.index).ffill()
    # only the most toxic tail (expanding 95th) — stand aside, don't kill the book
    exp_q = d["vpin"].expanding(min_periods=max(n_buckets, 100)).quantile(0.95)
    d["vpin_hi"] = (d["vpin"] >= exp_q) & exp_q.notna()
    d.attrs["bucket_vol"] = V
    return d


def apply_vpin_filter(feats: pd.DataFrame, ohlcv15: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Block new regime entries when VPIN is elevated (toxicity / liquidity stress)."""
    vp = compute_vpin_on_bars(ohlcv15[["open", "high", "low", "close", "volume"]], **kwargs)
    d = feats.copy()
    vpin = vp["vpin"].reindex(d.index).ffill()
    toxic = vp["vpin_hi"].reindex(d.index).fillna(False)
    # align tz
    if d.index.tz is not None and vpin.index.tz is None:
        pass
    d["vpin"] = vpin
    d["vpin_hi"] = toxic
    ok = ~toxic
    before_l = int(d["long_sig_a"].sum() + d["long_sig_b"].sum() + d["long_sig_c"].sum())
    before_s = int(d["short_sig_a"].sum() + d["short_sig_b"].sum() + d["short_sig_c"].sum())
    for col in ("long_sig_a", "long_sig_b", "long_sig_c", "short_sig_a", "short_sig_b", "short_sig_c", "short_hard_s"):
        if col in d.columns:
            d[col] = d[col] & ok
    after_l = int(d["long_sig_a"].sum() + d["long_sig_b"].sum() + d["long_sig_c"].sum())
    after_s = int(d["short_sig_a"].sum() + d["short_sig_b"].sum() + d["short_sig_c"].sum())
    print(
        f"VPIN filter bucket_vol≈{vp.attrs.get('bucket_vol', 0):.2f} | "
        f"long {before_l}→{after_l}, short {before_s}→{after_s} | "
        f"toxic bars={int(toxic.sum()):,}/{len(toxic):,}"
    )
    return d
