#!/usr/bin/env python3
"""
run_backtest.py — the entry point.

    python run_backtest.py --data data/sample_data.xlsx
    python run_backtest.py --data mine.csv --validate
    python run_backtest.py --data mine.csv --folds 5 --threshold 0.65

Everything it prints, it also writes to output/.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import latent
from config import CFG


BANNER = r"""
 _        _  _____ _____ _   _ _____
| |      / \|_   _| ____| \ | |_   _|    regime-aware BTC/USDT
| |     / _ \ | | |  _| |  \| | | |      research system
| |___ / ___ \| | | |___| |\  | | |
|_____/_/   \_\_| |_____|_| \_| |_|      v{v}
""".format(v=latent.__version__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the LATENT walk-forward backtest.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Results are backtested, not live traded. Read the risk notice in README.md.",
    )
    p.add_argument("--data", required=True,
                   help="path to OHLCV file (.xlsx or .csv)")
    p.add_argument("--out", default="output",
                   help="where to write results (default: output)")
    p.add_argument("--folds", type=int, default=3,
                   help="walk-forward folds (default: 3)")
    p.add_argument("--threshold", type=float, default=None,
                   help="override signal_buy_thresh, e.g. 0.65")
    p.add_argument("--validate", action="store_true",
                   help="check the data and exit without running anything")
    p.add_argument("--quiet", action="store_true", help="less output")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    verbose = not args.quiet
    if verbose:
        print(BANNER)

    cfg = dict(CFG)
    if args.threshold is not None:
        cfg["signal_buy_thresh"] = args.threshold
        cfg["signal_sell_thresh"] = round(1 - args.threshold, 4)
        if verbose:
            print(f"Threshold overridden: long >= {args.threshold}, "
                  f"short <= {cfg['signal_sell_thresh']}\n")

    # ── load ─────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        df = latent.load_data(args.data)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nCould not load your data:\n  {e}\n")
        return 1

    if verbose:
        print(f"Loaded {len(df):,} bars   {df['date'].iloc[0]}  ->  {df['date'].iloc[-1]}")

    problems = latent.validate(df)
    if problems:
        print("\nData check:")
        for p in problems:
            print(f"  ! {p}")
    elif verbose:
        print("Data check: nothing obviously wrong.")

    if args.validate:
        print("\n--validate given, stopping here.")
        return 0 if not problems else 1

    # ── features ─────────────────────────────────────────────────────
    if verbose:
        print("\nBuilding features...")
    df = latent.add_base_features(df)
    df = latent.add_signal_features(df, verbose=verbose)
    df = latent.add_garch_vol(df, verbose=verbose)
    if verbose:
        print(f"  {df.shape[1]} columns ready")

    # ── walk forward ─────────────────────────────────────────────────
    try:
        results = latent.walk_forward(
            df, latent.ALL_FEATURES, cfg, n_folds=args.folds, verbose=verbose
        )
    except ValueError as e:
        print(f"\nCould not run the backtest:\n  {e}\n")
        return 1

    # ── write ────────────────────────────────────────────────────────
    latent.report.write(results, cfg, out_dir=args.out, verbose=verbose)

    if verbose:
        print(f"\nWritten to {Path(args.out).resolve()}")
        print(f"Took {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
