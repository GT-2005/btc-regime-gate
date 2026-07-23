"""
LATENT — a regime-aware BTC/USDT research system.

Kalman filtering for direction, a hidden-Markov model for regime, GARCH for
volatility, a stacked ensemble for the decision, fractional Kelly for the size.

This is a research tool. It is not investment advice, and every published
figure comes from historical backtests rather than live trading.
"""

__version__ = "0.1.0"
__author__ = "Ganesh Tilekar"
__license__ = "MIT"

from .data import load_data, add_base_features, validate, BASE_FEATURES
from .signals import add_signal_features, SIGNAL_FEATURES
from .garch import add_garch_vol, GARCH_FEATURES
from .backtest import walk_forward
from . import metrics, report

ALL_FEATURES = BASE_FEATURES + SIGNAL_FEATURES + GARCH_FEATURES

__all__ = [
    "load_data", "add_base_features", "validate",
    "add_signal_features", "add_garch_vol",
    "walk_forward", "metrics", "report",
    "BASE_FEATURES", "SIGNAL_FEATURES", "GARCH_FEATURES", "ALL_FEATURES",
]
