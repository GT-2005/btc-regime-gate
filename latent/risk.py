"""
latent.risk — deciding how much, and when not to.

Three things sit between a signal and an order:

  1. Fractional Kelly turns a probability and a payoff ratio into a size
  2. GARCH volatility scales that size down when conditions are hot
  3. Hard caps and a drawdown brake override both

The order matters. Kelly is a growth-optimal formula that assumes you know
your edge exactly. You don't. Half Kelly is the usual compromise, and the
caps exist because even half Kelly will happily suggest a position that
ruins you if the probability estimate is wrong.
"""

from __future__ import annotations

import numpy as np


def kelly_fraction(p: float, payoff: float) -> float:
    """
    Full Kelly for a binary bet.

        f* = (p * b - q) / b        where q = 1 - p, b = win/loss ratio

    Returns 0 when the edge is negative — no bet is a valid bet.
    """
    if payoff <= 0 or not np.isfinite(payoff):
        return 0.0
    q = 1.0 - p
    f = (p * payoff - q) / payoff
    return float(max(f, 0.0))


class RiskManager:
    """
    Owns equity, the peak watermark, and the decision to stop trading.

    Usage per bar:
        rm.blocked()                  -> True if the brake is on
        rm.size(prob, payoff, vol)    -> fraction of equity to commit
        rm.apply(pnl)                 -> update equity after a closed trade
    """

    def __init__(self, cfg: dict):
        self.start = float(cfg["starting_capital"])
        self.equity = self.start
        self.peak = self.start

        self.kelly_frac = float(cfg.get("kelly_fraction", 0.5))
        self.max_risk = float(cfg["max_risk_per_trade"])
        self.max_pos = float(cfg["max_position_frac"])
        self.max_dd = float(cfg["max_drawdown_limit"])

        self.halted = False
        self.halt_bars = 0

    # ── state ────────────────────────────────────────────────────────
    @property
    def drawdown(self) -> float:
        return 0.0 if self.peak <= 0 else (self.peak - self.equity) / self.peak

    def blocked(self) -> bool:
        """
        The emergency brake. Once drawdown passes the limit, stop opening
        new positions until equity recovers to within half the limit.

        A system that keeps trading through its worst stretch is how a bad
        month becomes a blown account.
        """
        dd = self.drawdown
        if not self.halted and dd >= self.max_dd:
            self.halted = True
        elif self.halted and dd <= self.max_dd * 0.5:
            self.halted = False
        if self.halted:
            self.halt_bars += 1
        return self.halted

    # ── sizing ───────────────────────────────────────────────────────
    def size(self, prob: float, payoff: float, vol_regime: float = 1.0) -> float:
        """
        Fraction of equity to commit. Zero means don't take the trade.

        prob        model probability for the direction being taken
        payoff      reward-to-risk ratio of the planned trade
        vol_regime  forecast volatility / its long-run average
        """
        f = kelly_fraction(prob, payoff) * self.kelly_frac

        # calm markets get full size, hot markets get cut
        if np.isfinite(vol_regime) and vol_regime > 1.0:
            f /= min(vol_regime, 3.0)

        # the two hard ceilings
        f = min(f, self.max_pos)
        f = min(f, self.max_risk / max(1e-6, 1.0 / max(payoff, 1e-6)))

        return float(np.clip(f, 0.0, self.max_pos))

    def risk_capped_size(self, f: float, stop_dist_frac: float) -> float:
        """
        Second ceiling, expressed in loss terms: if this trade hits its stop,
        it must not cost more than `max_risk_per_trade` of equity.
        """
        if stop_dist_frac <= 0:
            return 0.0
        return float(min(f, self.max_risk / stop_dist_frac, self.max_pos))

    # ── accounting ───────────────────────────────────────────────────
    def apply(self, pnl: float) -> None:
        self.equity += pnl
        self.peak = max(self.peak, self.equity)
