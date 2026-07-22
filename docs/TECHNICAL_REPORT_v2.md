# BTC/USDT Regime-Adaptive Trading System with TradingView Deployment

## Technical Report (Version 2.1)

**Prepared by:**  
Ganesh Tilekar (UI23EC21)  
Department of Electronics and Communication Engineering

**Date:** July 2026  
**Supersedes:** Technical Report Version 2.0  

---

## Abstract

This report presents a two-layer BTC/USDT trading framework revised after underperformance of the earlier multi-gate design. Layer A is a Python research pipeline (features, optional ML boost, purged walk-forward, baselines). Layer B is a TradingView Pine Script v5 Regime Gate Indicator for live charts and alerts.

**Version 2.1 changes the trade engine, not just the packaging.** The system now uses two explicit playbooks (trend pullback and mean-reversion fade), a small set of hard gates, regime-specific exits with positive expectancy geometry, and ML as an optional soft boost rather than a mandatory stack of filters. Primary build remains 15-minute BTC/USDT with 1-hour higher-timeframe alignment.

---

## 1. Introduction

### 1.1 Background

Crypto trends and ranges alternate quickly. Systems that demand many simultaneous confirmations often undertrade or enter late. Systems with stop loss wider than take profit need unrealistically high win rates. Version 2.1 redesigns entries and exits around those failure modes.

### 1.2 Objectives

1. Detect regime (trend vs range vs shock) with simple, robust features first.  
2. Trade **only one playbook per regime** (no mixed rules on the same bar).  
3. Enforce higher-timeframe alignment on trend trades.  
4. Use exits with non-inverted risk/reward (or trailing logic in trends).  
5. Validate with purged walk-forward and mandatory baselines.  
6. Deploy the same playbooks on TradingView (Pine v5).  
7. Keep ML optional: boost or veto, never the only reason to trade.

### 1.3 Design Philosophy

Defensible, testable, improvable. No “ultimate indicator” claims. Success = beat baselines out of sample on risk-adjusted metrics with controlled drawdown, plus a chart layer traders can explain bar by bar.

### 1.4 Why Prior Logic Was Replaced (Honest Diagnosis)

| Old piece (v1.2 / v2.0) | Problem | v2.1 replacement |
|-------------------------|---------|------------------|
| TP 1.5 ATR / SL 2.0 ATR | Reward < risk (R:R ≈ 0.75). Needs win rate ≳ 57% after costs just to break even | Regime-specific exits (see Section 7) |
| 9 hard gates all required | Overfiltering: few trades, late entries, fragile | 4 hard gates + soft score |
| RSI forced into \[30, 70\] on every trade | Blocks trend continuation; fights momentum | RSI only inside **mean-reversion** playbook |
| OHLCV OFI as required confirm | Weak proxy; noisy on 15m | Relative volume (RVOL) soft confirm |
| HMM + Hurst + ADX all hard-aligned | Disagreement kills size/trades; HMM unstable | **ADX primary** regime; HMM/Hurst soft features only |
| Stacked ML required to enter | Easy to overfit; unclear edge vs rules | **Rules first**; ML optional boost/veto |
| Single exit template for all regimes | Trend and range need different hold logic | Playbook A trail / Playbook B fixed TP |

---

## 2. System Architecture

### 2.1 Two-Layer Contract

| Layer | Name | Role | Runtime |
|-------|------|------|---------|
| A | Research Engine | Backtest, metrics, optional ML, baselines | Python 3.x |
| B | Chart Deployment | Same playbooks, score, alerts | Pine Script v5 |

### 2.2 Decision Flow (v2.1)

```
OHLCV bar
  → Vol shock check (hard block)
  → Regime = Trend | Range | Neutral  (ADX + BB width)
  → if Neutral: no trade
  → if Trend: Playbook A (pullback) + HTF gate required
  → if Range: Playbook B (fade to mean) + stretch required
  → Soft score / optional ML boost
  → Size + exit per playbook
```

### 2.3 Product Name

**BTC Regime Gate Indicator (v2.1 playbooks)**

---

## 3. Data and Configuration

### 3.1 Data

| Item | Spec |
|------|------|
| Asset | BTC/USDT |
| Primary TF | **15m** (HTF = 1h) |
| Later | 5m, 30m, 1h |
| Quality | Ascending timestamps; drop duplicate stamps; flag gaps > 2 bars |

| TF | Slippage assumption (round trip) |
|----|----------------------------------|
| 5m | 12 bps |
| 15m | 9 bps |
| 30m | 8 bps |
| 1h | 9 bps |

### 3.2 Canonical Configuration (v2.1)

```
starting_capital         = 100000
max_risk_per_trade       = 0.01
max_drawdown_limit       = 0.15
max_position_frac        = 0.20

# Regime
adx_len                  = 14
adx_trend_thresh         = 25
adx_range_thresh         = 20
bb_width_lookback        = 20
vol_shock_atr_mult       = 2.5   # ATR / ATR_ma > this → no trade

# Playbook A (trend pullback)
ema_fast                 = 21
ema_slow                 = 50
pullback_touch_tol       = 0.15  # fraction of ATR from EMA21
atr_sl_trend             = 1.5
atr_trail_trend          = 2.0
max_hold_trend           = 48    # bars on 15m ≈ 12h

# Playbook B (mean reversion)
vwap_z_entry             = 1.8
vwap_z_exit              = 0.3
rsi_os                   = 30
rsi_ob                   = 70
atr_sl_range             = 1.0
atr_tp_range             = 1.5
max_hold_range           = 16    # bars on 15m ≈ 4h

# Soft score / ML (optional)
score_long_thresh        = 60
score_short_thresh       = 40
min_bars_between_trades  = 4
min_edge_over_cost       = 0.55

# Validation
train_frac               = 0.60
val_frac                 = 0.20
n_walk_forward           = 3
embargo_bars             = 10
```

### 3.3 HTF Map (unchanged intent)

| Trading TF | HTF | Long if | Short if |
|------------|-----|---------|----------|
| 5m | 15m | EMA50 > EMA200 | opposite |
| **15m** | **1h** | EMA50 > EMA200 | opposite |
| 30m | 1h | same | same |
| 1h | 4h | EMA50 > EMA200 | opposite |

HTF series shifted by one HTF bar before use (no peeking).

---

## 4. Feature Engineering (Layer A)

### 4.1 Core Features (always on)

- Returns, EMA21/50, EMA slope  
- ATR(14), ADX(14), Bollinger bandwidth  
- RSI(14), MACD histogram (soft)  
- Session VWAP and VWAP z-score  
- Relative volume: `volume / SMA(volume, 20)`  
- HTF trend flag  

### 4.2 Soft / Research-Only Features

- Kalman velocity (or EMA slope proxy on Pine)  
- Hurst exponent (regime research, not hard gate)  
- HMM state (research soft feature; not required to enter)  
- GARCH vol (sizing research; Pine uses ATR ratio)  

### 4.3 Labels (match playbook exits)

**Trend labels (Playbook A):** path-dependent — stop at 1.5 ATR adverse; success if trailing logic would lock ≥ 1.0 ATR before time stop (or use vertical barrier + MAE/MFE diagnostics).  

**Range labels (Playbook B):** classic triple-barrier with `atr_tp_range = 1.5`, `atr_sl_range = 1.0`, `max_hold_range = 16`.  

Binary training target remains playbook-conditional. Do not mix A and B labels in one undifferentiated model.

### 4.4 Leakage Controls

Past-only rolling windows; HTF lag; purged embargo; OOS-only test metrics.

---

## 5. Regime Detection (Simplified)

### 5.1 Primary Classifier (used for trading)

```
vol_shock = ATR / SMA(ATR, 50) >= vol_shock_atr_mult
if vol_shock: regime = SHOCK → no trade

else if ADX >= 25 and BB_width >= median(BB_width, 100):
    regime = TREND → Playbook A
else if ADX <= 20 and BB_width <= median(BB_width, 100):
    regime = RANGE → Playbook B
else:
    regime = NEUTRAL → no trade
```

### 5.2 What We Demoted

| Tool | Role in v2.1 |
|------|----------------|
| HMM | Soft feature / analysis only |
| Hurst | Soft confirmation of TREND vs RANGE; never sole gate |
| OFI (OHLCV) | Removed from required path |
| GARCH | Optional sizing research; ATR ratio on chart |

### 5.3 Directional Anchor

- **Kalman / EMA slope** confirms direction inside the chosen playbook.  
- Does not select the playbook by itself.

---

## 6. Strategy Playbooks (Core Alpha)

### 6.1 Playbook A — Trend Pullback (HTF aligned)

**When:** `regime == TREND`  
**Idea:** Buy dips in uptrends / sell rallies in downtrends; let winners run.

**Long entry (all hard gates):**

1. HTF trend_up = 1 (1h EMA50 > EMA200 for 15m chart)  
2. LTF EMA21 > EMA50 (structure still up)  
3. Pullback: low or close comes within `0.15 * ATR` of EMA21, then close reclaims above EMA21  
4. RVOL ≥ 0.8 (not a dead bar)  
5. Soft: Kalman/EMA slope ≥ 0; score ≥ 60 (or ML boost if enabled)  

**Short entry:** mirror.

**Exit (Playbook A):**

1. Initial stop: `1.5 * ATR` beyond swing extreme / entry  
2. After +1.0 ATR favorable: move stop to break-even  
3. Trail: Chandelier-style stop at `close − 2.0 * ATR` (long) / mirror for short  
4. Time stop: 48 bars if still open  
5. **No fixed small TP** — avoid cutting trends early  

**Why this replaces old trend logic:** old RSI mid-band gate and fixed 1.5/2.0 barriers capped winners and blocked momentum.

### 6.2 Playbook B — Mean-Reversion Fade

**When:** `regime == RANGE`  
**Idea:** Fade stretched deviations back toward VWAP; tight risk, modest target.

**Long entry (all hard gates):**

1. VWAP z-score ≤ `−1.8`  
2. RSI(14) ≤ 30  
3. Close starts reversing up (close > open or close > prior close)  
4. RVOL ≥ 0.8  
5. Soft: score ≥ 60  

**Short entry:** VWAP z ≥ `+1.8`, RSI ≥ 70, reverse candle, mirror.

**Exit (Playbook B):**

1. Take profit: nearer of (a) VWAP z → `0.3` or (b) `+1.5 * ATR`  
2. Stop loss: `1.0 * ATR`  
3. Time stop: 16 bars  
4. R:R geometric minimum ≈ **1.5 : 1** before costs  

**Why this replaces old universal exits:** range trades should not use trend trailing; they need mean target + tight stop.

### 6.3 Explicit No-Trade States

- SHOCK volatility  
- NEUTRAL regime  
- Trend signal without HTF alignment  
- Range signal without stretch (z and RSI)  
- Within `min_bars_between_trades` cooldown  

---

## 7. Soft Score, Optional ML, and Sizing

### 7.1 Hard vs Soft (important)

**Hard (must pass):** regime playbook gates in Section 6.  
**Soft:** composite score and optional ML — can block weak setups, cannot invent a trade that failed hard gates.

### 7.2 Composite Score Weights (Layer B / rule mode)

| Component | Weight |
|-----------|--------|
| Playbook hard gates satisfied | 40 (pass/fail lump) |
| HTF aligned (A) or stretch quality (B) | 20 |
| RVOL confirmation | 15 |
| Slope / structure quality | 15 |
| Distance to stop vs ATR (clean invalidation) | 10 |
| **Total** | **100** |

Long bias if score ≥ 60; short if ≤ 40 after side is chosen by playbook.

### 7.3 Optional ML (Layer A)

- Train **separate** models per playbook (A vs B), never one blob model.  
- Labels match that playbook’s barriers.  
- LightGBM or XGBoost + calibration.  
- Use as: `allow trade if p_cal >= 0.55` **in addition to** hard gates (veto), or small size boost if `p_cal >= 0.65`.  
- Stacking + dual-model agreement is optional research, not required for v2.1 MVP.

### 7.4 Position Sizing

- Risk `1%` of equity per trade at stop distance  
- Cap position at `20%` of equity  
- In SHOCK or after 2 consecutive losses: optional size × 0.5  
- Fractional Kelly only as an upper bound research check, not live default  

### 7.5 Cost-Aware Filter

Keep expected-move vs fee test using **playbook TP/SL geometry** (for B use 1.5/1.0; for A use conservative assumed capture of 1.5 ATR trail).

---

## 8. Validation Protocol

### 8.1 Method

- 60% / 20% / 20% with 3 purged walk-forward folds  
- `embargo_bars = 10`  
- Report separately for Playbook A, Playbook B, and combined  

### 8.2 Mandatory Baselines

1. Buy and hold BTC  
2. EMA50/200 crossover  
3. MACD signal crossover  
4. RSI 30/70 mean-reversion  

**Accept TF only if** combined system beats ≥ 3 of 4 on Sharpe **and** max drawdown OOS.

### 8.3 Diagnostics

- Val vs test metric gap  
- Deflated Sharpe Ratio with trial count  
- Fold stability of profit factor / win rate  
- **Per-playbook** trade count (avoid a “winning” combo that is only one playbook lucky)  

### 8.4 Python ↔ Pine Agreement

Rule-mode signals on 15m must be compared to Pine outputs on the same window before trusting alerts.

---

## 9. TradingView Layer (Layer B)

### 9.1 Plots / Outputs

1. Regime background (TREND / RANGE / NEUTRAL / SHOCK)  
2. Playbook A / B entry markers  
3. Stop / trail / TP reference lines  
4. Score pane (0–100)  
5. Alerts: `Long A`, `Short A`, `Long B`, `Short B`, `Exit`  

### 9.2 Pine Proxies

| Research | Pine |
|----------|------|
| Kalman | EMA slope |
| GARCH shock | ATR / SMA(ATR) |
| HMM | not used on chart |
| ML p_cal | soft score |

### 9.3 Non-Claims

No on-chart XGBoost. Score ≠ calibrated probability. Chart ≠ proof of edge without OOS research.

---

## 10. Performance Reporting Template

| TF | Playbook | OOS Ret | Sharpe | Max DD | PF | Trades | vs Baselines | Pine Agree % |
|----|----------|---------|--------|--------|----|--------|--------------|--------------|
| 15m | A | | | | | | | |
| 15m | B | | | | | | | |
| 15m | Combined | | | | | | | |

Transparency: date range, fees, spot vs perp, trial count for DSR, list of changed parameters from this report.

---

## 11. Implementation Roadmap

| Phase | Deliverable | Exit criterion |
|-------|-------------|----------------|
| P0 | Report v2.1 + strategy lock | Approved |
| P1 | Pine MVP: regime + Playbooks A/B + alerts | Works on BTCUSDT 15m |
| P2 | Python rule backtest A/B + baselines | Tables filled |
| P3 | Optional ML veto/boost per playbook | OOS lift vs rules alone |
| P4 | Multi-TF | Cross-TF table |
| P5 | README + paper-trade checklist | Resume-ready |

---

## 12. Limitations

1. Regime switches can whip around threshold boundaries (use hysteresis later if needed).  
2. VWAP on 24/7 crypto needs a defined anchor (rolling session or 24h VWAP — document choice in code).  
3. Trailing stops in Playbook A are path-dependent and sensitive to gap/wick behavior.  
4. 5m remains cost-sensitive; do not promote 5m without fee stress tests.  
5. Past underperformance can repeat if parameters are overfit to a short window — baselines and DSR remain mandatory.

---

## 13. Conclusion

Version 2.1 keeps the two-layer project shape but replaces the weak trade engine: inverted stops, over-gating, and one-size-fits-all exits. The new core is **regime → playbook → hard gates → soft score → playbook-specific exit**. That is simpler to implement, easier to debug when performance fails, and more honest for a resume / final-year narrative.

---

## 14. Future Work

1. Regime hysteresis (enter TREND at 25, exit at 22)  
2. CPCV validation  
3. Drift detection on score thresholds  
4. True microstructure features if order-book data becomes available  
5. Exchange paper-trade bot consuming the same signal CSV as Pine alerts  

---

## References (Core Set)

1. Wilder, J.W. (1978). New Concepts in Technical Trading Systems.  
2. Moskowitz, T., Ooi, Y.H., Pedersen, L.H. (2012). Time Series Momentum.  
3. Lopez de Prado, M. (2018). Advances in Financial Machine Learning.  
4. Bailey, D.H. and Lopez de Prado, M. (2014). Deflated Sharpe Ratio.  
5. Kalman, R.E. (1960). A New Approach to Linear Filtering and Prediction Problems.  
6. Chen, T. and Guestrin, C. (2016). XGBoost.  
7. Ke, G. et al. (2017). LightGBM.  
8. TradingView. Pine Script v5 Language Reference Manual.  

---

## Document Control

| Version | Summary |
|---------|---------|
| 1.2 | Research pipeline: triple-barrier, HTF, ML stack, purged WF |
| 2.0 | Added TradingView layer; 15m-first |
| **2.1** | **Replaced underperforming engine:** dual playbooks, fixed R:R / trail exits, fewer hard gates, demoted HMM/OFI/RSI-universal, ML optional |

**End of Report**
