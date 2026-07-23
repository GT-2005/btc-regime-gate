"""
latent.metrics — the numbers a sceptical reader would ask for.

Deliberately includes the unflattering ones. Max drawdown, longest losing
run and time under water say more about whether you could actually hold a
system than annual return does.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

BARS_PER_YEAR = 365 * 24          # hourly crypto: it never closes


def _ann(x: float) -> float:
    return x * np.sqrt(BARS_PER_YEAR)


def sharpe(returns: pd.Series) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(_ann(r.mean() / r.std()))


def sortino(returns: pd.Series) -> float:
    """Same idea as Sharpe, but upside surprises are not treated as risk."""
    r = returns.dropna()
    downside = r[r < 0]
    if len(r) < 2 or len(downside) == 0 or downside.std() == 0:
        return 0.0
    return float(_ann(r.mean() / downside.std()))


def max_drawdown(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    peak = equity.cummax()
    return float(((peak - equity) / peak.replace(0, np.nan)).max() or 0.0)


def calmar(equity: pd.Series, returns: pd.Series) -> float:
    """Annualised return divided by the worst drawdown you had to sit through."""
    mdd = max_drawdown(equity)
    if mdd == 0 or len(equity) < 2:
        return 0.0
    total = equity.iloc[-1] / equity.iloc[0] - 1
    years = max(len(returns) / BARS_PER_YEAR, 1e-9)
    annual = (1 + total) ** (1 / years) - 1
    return float(annual / mdd)


def profit_factor(pnls: np.ndarray) -> float:
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def value_at_risk(pnls: np.ndarray, level: float = 0.05) -> float:
    """The loss you should expect to be beaten only 5% of the time."""
    return float(np.percentile(pnls, level * 100)) if len(pnls) else 0.0


def longest_losing_run(pnls: np.ndarray) -> int:
    worst = run = 0
    for p in pnls:
        run = run + 1 if p < 0 else 0
        worst = max(worst, run)
    return int(worst)


def time_under_water(equity: pd.Series) -> float:
    """Fraction of the period spent below a previous equity peak."""
    if len(equity) < 2:
        return 0.0
    return float((equity < equity.cummax()).mean())


def summarise(trades: pd.DataFrame, equity: pd.Series) -> dict:
    """Everything, in one dictionary, ready to write to CSV."""
    if trades.empty:
        return {"trades": 0, "note": "no trades taken"}

    pnl = trades["pnl"].to_numpy()
    rets = equity.pct_change().dropna()
    wins = pnl > 0

    total_return = equity.iloc[-1] / equity.iloc[0] - 1

    return {
        "trades": int(len(trades)),
        "net_return_pct": round(total_return * 100, 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "win_rate_pct": round(float(wins.mean() * 100), 2),
        "avg_win": round(float(pnl[wins].mean()) if wins.any() else 0.0, 2),
        "avg_loss": round(float(pnl[~wins].mean()) if (~wins).any() else 0.0, 2),
        "profit_factor": round(profit_factor(pnl), 3),
        "sharpe": round(sharpe(rets), 3),
        "sortino": round(sortino(rets), 3),
        "calmar": round(calmar(equity, rets), 3),
        "max_drawdown_pct": round(max_drawdown(equity) * 100, 2),
        "var_95": round(value_at_risk(pnl), 2),
        "longest_losing_run": longest_losing_run(pnl),
        "time_under_water_pct": round(time_under_water(equity) * 100, 2),
        "avg_holding_bars": round(float(trades["bars_held"].mean()), 1),
    }


def by_regime(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Performance split by the regime the trade was opened in.

    This is the table worth reading twice. A system that only works in one
    regime is a system waiting for that regime to end.
    """
    if trades.empty or "regime" not in trades.columns:
        return pd.DataFrame()

    names = {0: "bear", 1: "sideways", 2: "bull"}
    rows = []
    for r, g in trades.groupby("regime"):
        pnl = g["pnl"].to_numpy()
        rows.append({
            "regime": names.get(int(r), str(r)),
            "trades": len(g),
            "share_of_trades_pct": round(len(g) / len(trades) * 100, 1),
            "win_rate_pct": round(float((pnl > 0).mean() * 100), 1),
            "avg_pnl": round(float(pnl.mean()), 2),
            "total_pnl": round(float(pnl.sum()), 2),
            "profit_factor": round(profit_factor(pnl), 3),
        })
    return pd.DataFrame(rows).sort_values("regime").reset_index(drop=True)
