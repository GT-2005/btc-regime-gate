"""
latent.backtest — walk-forward validation and trade execution.

The walk-forward loop is the part that makes the numbers mean something.
The data is cut into sequential folds; the model trains on everything up to
a point and is scored only on what comes after. It never sees its own test
set. Train once, test once gives you a number. This gives you a range, and
the range is the honest answer.

Execution is bar-by-bar and intentionally pessimistic:
  - entry and exit both pay commission and slippage
  - a stop that could have been hit inside the bar is assumed hit
  - if both stop and target sit inside the same bar, the stop wins
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .model import StackedEnsemble, safe_features
from .risk import RiskManager


# ─────────────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────────────
def _simulate(df: pd.DataFrame, probs: np.ndarray, cfg: dict,
              rm: RiskManager, verbose: bool = True) -> tuple[pd.DataFrame, list[float]]:
    """
    Walk the test slice bar by bar, opening and closing one position at a time.

    Returns (trades, equity_curve).
    """
    buy_t = float(cfg["signal_buy_thresh"])
    sell_t = float(cfg["signal_sell_thresh"])
    atr_stop = float(cfg.get("atr_stop_mult", 2.4))
    atr_target = float(cfg.get("atr_target_mult", 3.0))
    timeout = int(cfg.get("timeout_bars", 25))
    cool = int(cfg.get("min_bars_between", 3))
    comm = float(cfg.get("commission", 0.0004))
    slip = float(cfg.get("slippage", 0.0002))
    allow_short = bool(cfg.get("allow_shorts", True))

    trades: list[dict] = []
    equity_curve: list[float] = []

    pos = None            # open position, or None
    last_exit = -10**9

    idx = df.index.to_numpy()
    close = df["close"].to_numpy(float)
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
    regime = df["hmm_regime"].to_numpy(int) if "hmm_regime" in df else np.ones(len(df), int)
    volreg = df["vol_regime"].to_numpy(float) if "vol_regime" in df else np.ones(len(df))
    rsi = df["rsi"].to_numpy(float) if "rsi" in df else np.full(len(df), 50.0)
    hurst = df["hurst"].to_numpy(float) if "hurst" in df else np.full(len(df), 0.5)
    dates = df["date"].to_numpy() if "date" in df else idx

    for i in range(len(df)):
        # ── manage an open position first ────────────────────────────
        if pos is not None:
            hit_stop = low[i] <= pos["stop"] if pos["dir"] == 1 else high[i] >= pos["stop"]
            hit_tgt = high[i] >= pos["target"] if pos["dir"] == 1 else low[i] <= pos["target"]
            aged = (i - pos["i"]) >= timeout

            exit_px = reason = None
            if hit_stop:                      # pessimistic: stop wins ties
                exit_px, reason = pos["stop"], "stop"
            elif hit_tgt:
                exit_px, reason = pos["target"], "target"
            elif aged:
                exit_px, reason = close[i], "timeout"
            elif i == len(df) - 1:
                exit_px, reason = close[i], "end of data"

            if exit_px is not None:
                fill = exit_px * (1 - slip * pos["dir"])
                gross = (fill - pos["entry"]) * pos["dir"] / pos["entry"]
                net = gross - 2 * comm
                pnl = net * pos["notional"]
                rm.apply(pnl)

                trades.append({
                    "entry_time": pos["date"], "exit_time": dates[i],
                    "direction": "long" if pos["dir"] == 1 else "short",
                    "entry": round(pos["entry"], 2), "exit": round(fill, 2),
                    "prob": round(pos["prob"], 4),
                    "size_frac": round(pos["frac"], 4),
                    "notional": round(pos["notional"], 2),
                    "regime": pos["regime"],
                    "bars_held": i - pos["i"],
                    "exit_reason": reason,
                    "return_pct": round(net * 100, 4),
                    "pnl": round(pnl, 2),
                    "equity_after": round(rm.equity, 2),
                })
                pos = None
                last_exit = i

        # ── consider a new one ───────────────────────────────────────
        if pos is None and i < len(df) - 1:
            p = probs[i]
            a = atr[i]

            # the six-condition entry gate
            gate_long = (
                p >= buy_t                              # 1. model confident
                and 30 <= rsi[i] <= 70                  # 2. not already stretched
                and np.isfinite(a) and a > 0            # 3. volatility is measurable
                and (i - last_exit) >= cool             # 4. cooldown respected
                and not rm.blocked()                    # 5. drawdown brake off
                and volreg[i] < 3.0                     # 6. not a volatility spike
            )
            gate_short = allow_short and (
                p <= sell_t
                and 30 <= rsi[i] <= 70
                and np.isfinite(a) and a > 0
                and (i - last_exit) >= cool
                and not rm.blocked()
                and volreg[i] < 3.0
            )

            if gate_long or gate_short:
                d = 1 if gate_long else -1
                conf = p if d == 1 else 1 - p

                # trending tape earns a wider target
                tgt_mult = atr_target * (1.15 if hurst[i] > 0.55 else 1.0)
                stop_dist = atr_stop * a
                tgt_dist = tgt_mult * a
                payoff = tgt_dist / stop_dist if stop_dist > 0 else 0.0

                frac = rm.size(conf, payoff, volreg[i])
                frac = rm.risk_capped_size(frac, stop_dist / close[i])

                if frac > 1e-4:
                    entry = close[i] * (1 + slip * d)
                    pos = {
                        "i": i, "dir": d, "entry": entry,
                        "stop": entry - d * stop_dist,
                        "target": entry + d * tgt_dist,
                        "prob": p, "frac": frac,
                        "notional": rm.equity * frac,
                        "regime": int(regime[i]),
                        "date": dates[i],
                    }

        equity_curve.append(rm.equity)

    return pd.DataFrame(trades), equity_curve


# ─────────────────────────────────────────────────────────────────────────
# Walk-forward
# ─────────────────────────────────────────────────────────────────────────
def walk_forward(df: pd.DataFrame, features: list[str], cfg: dict,
                 n_folds: int = 3, verbose: bool = True) -> dict:
    """
    Train and test across sequential folds.

    Fold k trains on everything before its test window and is scored only on
    the window itself. Returns trades, the equity curve, per-fold AUCs and
    the feature importances from the final fold.
    """
    d = df.dropna(subset=["target"]).reset_index(drop=True)
    feats = safe_features(d, features)
    if verbose:
        dropped = set(features) - set(feats)
        print(f"\nUsing {len(feats)} of {len(features)} features"
              + (f" (dropped: {', '.join(sorted(dropped))})" if dropped else ""))

    n = len(d)
    test_size = n // (n_folds + 1)
    if test_size < 100:
        raise ValueError(
            f"Only {n} usable rows. Need roughly {(n_folds + 1) * 100} minimum "
            f"for {n_folds} folds."
        )

    rm = RiskManager(cfg)
    all_trades, all_equity, fold_stats = [], [], []
    importances = None

    for k in range(n_folds):
        train_end = test_size * (k + 1)
        test_end = min(train_end + test_size, n)
        if test_end - train_end < 50:
            break

        train = d.iloc[:train_end]
        test = d.iloc[train_end:test_end].reset_index(drop=True)

        if verbose:
            print(f"\nFold {k + 1}/{n_folds}"
                  f"  train {len(train):>6} rows  |  test {len(test):>6} rows")

        X_tr = train[feats].ffill().fillna(0)
        y_tr = train["target"].astype(int)

        try:
            model = StackedEnsemble().fit(X_tr, y_tr, verbose=verbose)
        except Exception as e:                               # noqa: BLE001
            if verbose:
                print(f"  ! fold skipped: {e}")
            continue

        probs = model.predict_proba(test[feats].ffill().fillna(0))
        equity_before = rm.equity

        # worth knowing before you conclude the engine is broken
        n_signal = int((probs >= cfg["signal_buy_thresh"]).sum()
                       + (probs <= cfg["signal_sell_thresh"]).sum())
        if verbose and n_signal == 0:
            print(f"  no bar crossed the threshold — probabilities ran "
                  f"{probs.min():.3f} to {probs.max():.3f}, gate needs "
                  f">={cfg['signal_buy_thresh']:.2f} or <={cfg['signal_sell_thresh']:.2f}")
            print("    this is the model declining to trade, not a failure")
        trades, curve = _simulate(test, probs, cfg, rm, verbose)

        if not trades.empty:
            trades["fold"] = k + 1
            all_trades.append(trades)
        all_equity.extend(curve)
        importances = model.importances()

        fold_stats.append({
            "fold": k + 1,
            "train_rows": len(train),
            "test_rows": len(test),
            "trades": len(trades),
            "bars_over_threshold": n_signal,
            "prob_min": round(float(probs.min()), 4),
            "prob_max": round(float(probs.max()), 4),
            "auc_xgb": round(model.auc_xgb, 4),
            "auc_lgb": round(model.auc_lgb, 4),
            "auc_stacked": round(model.auc_meta, 4),
            "return_pct": round((rm.equity / equity_before - 1) * 100, 2),
            "equity_end": round(rm.equity, 2),
        })

        if verbose:
            print(f"  {len(trades):>4} trades   equity {equity_before:,.0f} "
                  f"-> {rm.equity:,.0f}  ({(rm.equity / equity_before - 1) * 100:+.2f}%)")

    trades = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    equity = pd.Series(all_equity if all_equity else [rm.start])

    return {
        "trades": trades,
        "equity": equity,
        "folds": pd.DataFrame(fold_stats),
        "importances": importances,
        "features_used": feats,
        "halted_bars": rm.halt_bars,
    }
