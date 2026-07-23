"""
config.py — every tunable parameter for LATENT lives here.

The point of keeping it in one file: you should be able to change a
threshold, a risk limit, or a data split without reading a single line of
algorithmic code.

Change ONE value, re-run, compare. Changing five at once tells you nothing
about which one mattered.
"""

CFG = {
    # ── Capital and risk ────────────────────────────────────────────
    # Starting portfolio in USD. Every return and drawdown figure is
    # measured against this baseline.
    "starting_capital": 100_000,

    # Hard ceiling on per-trade risk. On a 100k book, no single trade
    # can put more than $1,000 at stake.
    "max_risk_per_trade": 0.01,

    # The emergency brake. Once peak-to-trough drop passes this, the
    # system halts new entries and goes into preservation mode.
    "max_drawdown_limit": 0.15,

    # Concentration cap. Even if Kelly asks for more, no trade takes
    # more than this fraction of capital.
    "max_position_frac": 0.20,

    # ── Signal thresholds ───────────────────────────────────────────
    # The model must be at least this confident before a long is
    # considered. Raise it for fewer, higher-conviction trades.
    "signal_buy_thresh": 0.62,

    # For shorts, probability must drop to this or below.
    # The two thresholds are symmetric around 0.50.
    "signal_sell_thresh": 0.38,
}
