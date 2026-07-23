"""
latent.report — writing results to disk.

Produces:
    metrics.csv            headline numbers
    trades.csv             every trade, with the reason it closed
    regime_breakdown.csv   the same metrics split by regime
    folds.csv              per-fold detail
    feature_importance.csv what the model actually leaned on
    equity_curve.png       the picture
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")                       # no display in most environments
import matplotlib.pyplot as plt             # noqa: E402
import pandas as pd                         # noqa: E402

from . import metrics                       # noqa: E402

INK = "#0F1B2D"
ACCENT = "#FF5A1F"
MUTED = "#8A96A8"


def _plot(equity: pd.Series, trades: pd.DataFrame, out: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11, 7), height_ratios=[3, 1], sharex=True
    )
    fig.patch.set_facecolor("white")

    ax1.plot(equity.index, equity.values, color=ACCENT, lw=1.6, label="Equity")
    ax1.fill_between(equity.index, equity.min(), equity.values,
                     color=ACCENT, alpha=0.07)
    ax1.axhline(equity.iloc[0], color=MUTED, lw=1, ls="--", label="Starting capital")
    ax1.set_title("LATENT — walk-forward equity curve (out-of-sample)",
                  fontsize=13, color=INK, loc="left", pad=14)
    ax1.set_ylabel("Equity")
    ax1.legend(frameon=False, fontsize=9)
    ax1.grid(alpha=0.15)

    peak = equity.cummax()
    dd = (equity - peak) / peak * 100
    ax2.fill_between(dd.index, dd.values, 0, color="#C0392B", alpha=0.35)
    ax2.set_ylabel("Drawdown %")
    ax2.set_xlabel("Bar")
    ax2.grid(alpha=0.15)

    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "equity_curve.png", dpi=140)
    plt.close(fig)


def write(results: dict, cfg: dict, out_dir: str | Path = "output",
          verbose: bool = True) -> dict:
    """Write everything to `out_dir` and return the summary dictionary."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    trades: pd.DataFrame = results["trades"]
    equity: pd.Series = results["equity"]

    summary = metrics.summarise(trades, equity)
    summary["halted_bars"] = results.get("halted_bars", 0)
    summary["features_used"] = len(results.get("features_used", []))

    pd.DataFrame([summary]).T.rename(columns={0: "value"}).to_csv(out / "metrics.csv")

    if not trades.empty:
        trades.to_csv(out / "trades.csv", index=False)
        metrics.by_regime(trades).to_csv(out / "regime_breakdown.csv", index=False)

    if results.get("folds") is not None and not results["folds"].empty:
        results["folds"].to_csv(out / "folds.csv", index=False)

    if results.get("importances") is not None:
        results["importances"].to_csv(out / "feature_importance.csv", index=False)

    if len(equity) > 1:
        _plot(equity, trades, out)

    if verbose:
        _print(summary, trades, results)

    return summary


def _print(summary: dict, trades: pd.DataFrame, results: dict) -> None:
    line = "─" * 62
    print(f"\n{line}\nRESULTS — out-of-sample, walk-forward\n{line}")

    if summary.get("trades", 0) == 0:
        print("No trades were taken. Try lowering signal_buy_thresh in config.py,")
        print("or check that your data has enough rows.")
        return

    rows = [
        ("Net return", f"{summary['net_return_pct']:+.2f}%"),
        ("Final equity", f"{summary['final_equity']:,.2f}"),
        ("Trades", f"{summary['trades']}"),
        ("Win rate", f"{summary['win_rate_pct']:.2f}%"),
        ("Profit factor", f"{summary['profit_factor']:.3f}"),
        ("Sharpe", f"{summary['sharpe']:.3f}"),
        ("Sortino", f"{summary['sortino']:.3f}"),
        ("Calmar", f"{summary['calmar']:.3f}"),
        ("Max drawdown", f"{summary['max_drawdown_pct']:.2f}%"),
        ("VaR 95%", f"{summary['var_95']:,.2f}"),
        ("Longest losing run", f"{summary['longest_losing_run']} trades"),
        ("Time under water", f"{summary['time_under_water_pct']:.1f}%"),
        ("Avg holding", f"{summary['avg_holding_bars']:.1f} bars"),
    ]
    for k, v in rows:
        print(f"  {k:<22} {v:>18}")

    reg = metrics.by_regime(trades)
    if not reg.empty:
        print(f"\n{line}\nBY REGIME\n{line}")
        print(reg.to_string(index=False))

    imp = results.get("importances")
    if imp is not None and not imp.empty:
        print(f"\n{line}\nTOP FEATURES\n{line}")
        for _, r in imp.head(8).iterrows():
            bar = "█" * int(r["importance"] * 120)
            print(f"  {r['feature']:<16} {r['importance'] * 100:5.1f}%  {bar}")

    print(f"\n{line}")
    print("These are backtest results on out-of-sample folds, net of modelled")
    print("costs. They are not a live track record and not a forecast.")
    print(line)
