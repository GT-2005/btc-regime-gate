# Project Report

## Bitcoin (BTC/USDT) Regime-Adaptive Trading System  
### With Research Pipeline, Machine Learning Filters, and TradingView Deployment

---

| Field | Details |
|-------|---------|
| **Student** | Ganesh Tilekar |
| **Roll No.** | UI23EC21 |
| **Department** | Electronics and Communication Engineering |
| **Project type** | Final-year / research-oriented trading system |
| **Version** | 2.7 (July 2026) |
| **Market / TF** | BTCUSDT, 15-minute bars (+ 1h / daily context) |

---

## Abstract

This project designs, implements, and evaluates a **regime-adaptive Bitcoin trading system**. Market state is classified as trend, range, or volatility shock using ADX and ATR. Two playbooks fire accordingly: **trend pullback (A)** and **mean-reversion fade (B)**. The system enforces **one open position at a time**, asymmetric risk/reward (stop 1 ATR, take-profit 3 ATR on trend trades), and optional research filters drawn from academic literature (time-series momentum, VPIN / volume clock, walk-forward ML).

A **Python backtest engine** validates rules on Binance-style 15m history (2017–2026). A **TradingView Pine Script v6 indicator** (`BTC Regime Gate v2.7`) deploys the same logic for charting and alerts. Results are reported honestly: the best stable configuration (`short_only`) achieves roughly **+1%** on 2025 starting from $100,000 while buy-and-hold was negative, and about **+$1,000** summed across several calendar years—**not** a guaranteed 15% system. Experiments that chased higher returns (breakouts, ML spam entries) destroyed capital and were rejected.

---

## 1. Introduction

### 1.1 Motivation

Bitcoin alternates between strong trends and choppy ranges. Naive indicators either overtrade or enter late. Many retail systems also invert risk/reward (wide stop, tight target), which requires unrealistically high win rates after fees.

### 1.2 Objectives

1. Detect regime (trend / range / shock) with robust, explainable features.  
2. Trade **one playbook per regime** with clear entry and exit rules.  
3. Allow **only one long or one short** at a time (no signal spam).  
4. Backtest with fees, slippage, and year-by-year evaluation.  
5. Optionally use ML / microstructure filters without replacing the rule core.  
6. Deploy the same logic on TradingView for live monitoring.  
7. Document failures as carefully as successes (scientific honesty).

### 1.3 Scope

- **In scope:** Spot-style BTCUSDT OHLCV, 15m research, Pine indicator, offline backtests.  
- **Out of scope:** Guaranteed returns, live brokerage execution, tick-level HFT, options VRP portfolios.

---

## 2. Literature Survey (Selected)

Papers were triaged into **KEEP** (used in design) and **NOT TAKEN** (theory / wrong market / too heavy for this stack).

### 2.1 KEEP (implemented or guiding design)

| # | Paper | Use in this project |
|---|--------|---------------------|
| 1 | Easley, López de Prado, O’Hara — *The Volume Clock* | Volume-time thinking; volume buckets |
| 2 | Easley, López de Prado, O’Hara — *Flow Toxicity / VPIN* | Optional toxicity filter (`--vpin`) |
| 3 | Bailey & López de Prado — *Deflated Sharpe Ratio* | Avoid fake performance after many trials |
| 4 | Bailey et al. — *Probability of Backtest Overfitting* | Holdout alone is not enough |
| 5 | López de Prado — *Advances in Financial Machine Learning* | Triple-barrier labels, walk-forward, meta-label path |
| 6 | Carr & López de Prado — *Optimal Trading Rules without Backtesting* | Caution on over-tuning stops via search |
| 7 | Moskowitz, Ooi, Pedersen — *Time Series Momentum* | Side filter: trade with ~60-day return sign |
| 8 | Baz, Harvey et al. — *Dissecting carry/momentum/value* | Prefer **time-series** (single asset) over cross-section |
| 9 | Quant note — *Hyperparameters are trading opinions* | Treat windows/thresholds as risk beliefs |

### 2.2 NOT TAKEN (examples)

CAPM (Sharpe), Fama–French cross-section, Markowitz portfolio theory, Kyle / Glosten–Milgrom (theory only), Bitcoin whitepaper (protocol), FinRL / deep RL libraries (overfit risk, deferred), Polymarket studies (wrong market), HFT queue/latency notes (not tradable here).

---

## 3. System Architecture

### 3.1 Two-layer design

| Layer | Component | Role |
|-------|-----------|------|
| **A — Research** | Python (`backtest_regime_gate.py`, `train_btc_patterns.py`, `vpin_volume.py`) | Features, backtest, ML, yearly stats |
| **B — Deployment** | Pine Script v6 (`BTC_Regime_Gate_v27.pine`) | Chart markers, position state, alerts |

### 3.2 Decision flow

```
15m OHLCV bar
  → Volatility shock? → BLOCK
  → Regime: TREND / RANGE / NEUTRAL (ADX primary)
  → TS momentum side filter (optional, default ON)
  → Playbook A (trend pullback) or B (VWAP fade)
  → If FLAT + cooldown → open ONE position
  → Manage SL / TP / trail / time stop
  → EXIT (close long) or COVER (close short) → FLAT
```

### 3.3 Position discipline (critical)

| Side | Open | Close |
|------|------|--------|
| Long | **BUY** | **EXIT** (sell) |
| Short | **SELL** | **COVER** (buy back) |

No second BUY while long; no second SELL while short. New entry only after flat + cooldown.

---

## 4. Methodology

### 4.1 Data

- Source: Binance-style BTCUSDT **15-minute** candles (`BTCUSDT_15m_All.csv`).  
- Span used: approximately **2017-08 → 2026-06** (~310k bars).  
- Resampling: 1h and 1D for higher-timeframe bias and TS momentum.

### 4.2 Regime detection

- **Trend:** ADX ≥ 30 and no ATR shock.  
- **Range:** ADX ≤ 18 and no shock.  
- **Shock:** ATR / ATR_MA ≥ 2.5 → no new trades.  
- HTF bias: 1h EMA50 vs EMA200 (lagged).  
- Daily bias (Auto mode): price vs daily EMA50 / EMA200.

### 4.3 Playbook A — Trend pullback

- Structure: EMA21 vs EMA50 separation.  
- Entry: pullback to EMA + reclaim candle + RVOL + HTF align.  
- Exit: SL 1 ATR, TP 3 ATR, breakeven + trail, max hold ~96 bars.

### 4.4 Playbook B — Mean reversion

- Range regime + extreme rolling VWAP z-score + RSI extreme + reversal candle.  
- Exit: tighter SL/TP and/or z-score mean return, short max hold.

### 4.5 Playbook C — Breakout (disabled)

Tested; produced large losses (~−15% in stress tests). **Kept OFF.**

### 4.6 Research add-ons

| Module | Default | Role |
|--------|---------|------|
| Time-series momentum (~60d) | ON | Long only if momentum up; short if down |
| VPIN (bar-level proxy) | OFF (`--vpin`) | Stand aside when flow looks toxic |
| LightGBM pattern model | OFF (`--ml`) | Optional soft gate; walk-forward trained |
| Pure ML entries | Rejected | Overtrading; large losses OOS |

### 4.7 Risk and costs (backtest)

| Parameter | Value |
|-----------|--------|
| Starting capital (per run) | $100,000 |
| Risk per trade | 2% of equity |
| Max position fraction | 25% |
| Slippage | ~9 bps round-trip style |
| Fee | ~4 bps one way |
| Trade modes | `short_only`, `long_only`, `auto`, `both` |

---

## 5. Implementation

### 5.1 Repository layout (main files)

| Path | Description |
|------|-------------|
| `pine/BTC_Regime_Gate_v27.pine` | Live indicator (one-position FSM) |
| `python/backtest_regime_gate.py` | Main backtest + yearly runner |
| `python/train_btc_patterns.py` | Feature build + LightGBM walk-forward |
| `python/vpin_volume.py` | Volume buckets + VPIN proxy |
| `python/backtest_ml_2025.py` | Honest OOS ML strategy test |
| `python/models_btc15m/` | Models, importance, yearly CSVs |
| `STRATEGY.md` | Locked strategy notes |
| `docs/TECHNICAL_REPORT_v2.md` | Earlier detailed design notes |

### 5.2 How to run backtests

```bash
# Single year
python3 python/backtest_regime_gate.py \
  --data /path/to/BTCUSDT_15m_All.csv \
  --mode short_only --year 2025

# Multiple years (fresh $100k each)
python3 python/backtest_regime_gate.py \
  --data /path/to/BTCUSDT_15m_All.csv \
  --mode short_only \
  --years 2019,2020,2021,2022,2024,2025
```

### 5.3 TradingView deployment

1. Open Pine Editor → paste `BTC_Regime_Gate_v27.pine` → Add to chart (BTC 15m).  
2. Remove older v2.1 / v2.3 indicators.  
3. Verify table **Pos** = FLAT / LONG / SHORT and markers BUY→EXIT or SELL→COVER.

---

## 6. Experiments and Results

All figures below use **$100,000** starting capital, fees/slippage on, breakout C off, unless noted.

### 6.1 Year-by-year — `short_only` (plain rules)

| Year | Profit (approx.) | Return | Trades | Buy & hold (approx.) |
|------|------------------|--------|--------|----------------------|
| 2019 | −$417 | −0.42% | 42 | +95% |
| 2020 | +$1,460 | +1.46% | 35 | +303% |
| 2021 | +$404 | +0.40% | 28 | +61% |
| 2022 | −$1,369 | −1.37% | 53 | −64% |
| 2024 | −$182 | −0.18% | 24 | +120% |
| 2025 | +$1,122 | +1.12% | 31 | −6% |
| **Sum** | **~$+1,018** | | | |

Interpretation: edge is **small but real vs bad years for buy-and-hold (e.g. 2025)**; it does **not** capture full bull-market upside (2019–2021, 2024).

### 6.2 With TS momentum (v2.7)

On 2025 `short_only`, a smoke test after adding TS momentum showed about **+1.36%** with fewer trades (19)—directionally consistent with filtering against the trend sign.

### 6.3 Failed / weaker variants (documented)

| Variant | Outcome | Decision |
|---------|---------|----------|
| Breakout playbook C | Large loss (~−15% class) | Disabled |
| Pure ML probability entries | Catastrophic OOS (~−50% class) | Rejected |
| Aggressive VPIN (blocked most bars) | Almost no trades | Softened; opt-in only |
| Auto mode (longs + shorts) | Often worse than short_only | Prefer short_only / careful Auto |

### 6.4 Machine learning summary

- ~116 engineered features; LightGBM walk-forward by year.  
- OOS AUC ~0.51–0.55 (weak discrimination).  
- Best use: **optional filter**, not primary signal generator.  
- Model trained including 2025 is **not** a valid 2025 test; pre-2025 model used for honest checks.

---

## 7. Discussion

### 7.1 Why not 15%?

Academic and practitioner papers (Deflated Sharpe, backtest overfitting) warn that scanning many rules invents lucky history. On fee-aware BTC 15m, a **~1% annual edge with low drawdown** can still beat buy-and-hold in a down year, but is far from a marketing “15% printer.” Higher returns would need a different edge (e.g. strong multi-year trend capture on higher timeframes, or leverage—which multiplies risk).

### 7.2 Why one position?

Multiple stacked SELLs without COVER inflate risk and confuse evaluation. One ticket matches risk budgeting (2% per trade) and makes the Pine chart auditable.

### 7.3 Engineering contribution

The project is not only an indicator: it is a **full research loop**—data hygiene, regime rules, ablation of failed ideas, optional ML/VPIN, and chart deployment with matching position state.

---

## 8. Conclusion

This work delivered a **complete, honest BTC regime-gate trading project**:

1. Rule-based playbooks A/B with ADX regimes and HTF gates.  
2. Strict **one-position** lifecycle (BUY/EXIT and SELL/COVER).  
3. Research-backed filters (TS momentum; optional VPIN/ML).  
4. Python backtests across multiple years with costs.  
5. TradingView v2.7 indicator for live monitoring.  
6. Clear rejection of overfitting paths that looked profitable only in hindsight.

**Final recommended configuration for evaluation:** `short_only`, SL 1 ATR / TP 3 ATR, 2% risk, TS momentum ON, breakout C OFF, VPIN/ML optional.

---

## 9. Future work

1. Meta-labeling (AFML): rules propose trade → ML only accepts/rejects.  
2. Report Deflated Sharpe with explicit trial count.  
3. Higher-timeframe (4h/1d) trend module for bull years.  
4. Tick-level VPIN if trade data becomes available.  
5. Paper trading / exchange API (out of current academic scope).

---

## 10. References (selected)

1. Easley, D., López de Prado, M., & O’Hara, M. (2012). *The Volume Clock.* Journal of Portfolio Management / SSRN 2034858.  
2. Easley, D., López de Prado, M., & O’Hara, M. (2012). *Flow Toxicity and Liquidity in a High-Frequency World.* Review of Financial Studies / SSRN 1695596.  
3. Bailey, D. H., & López de Prado, M. (2014). *The Deflated Sharpe Ratio.* SSRN 2460551.  
4. Bailey, D. H., et al. *The Probability of Backtest Overfitting.* SSRN 2326253.  
5. López de Prado, M. *Advances in Financial Machine Learning.* Wiley / related SSRN materials.  
6. Carr, P., & López de Prado, M. (2014). *Determining Optimal Trading Rules without Backtesting.* arXiv:1408.1159.  
7. Moskowitz, T. J., Ooi, Y. H., & Pedersen, L. H. (2012). *Time series momentum.* Journal of Financial Economics.  
8. Baz, J., Granger, N., Harvey, C. R., Le Roux, N., & Rattray, S. *Dissecting Investment Strategies in the Cross Section and Time Series.* SSRN 2695101.  
9. Nakamoto, S. (2008). *Bitcoin: A Peer-to-Peer Electronic Cash System.* (Background only.)

---

## Appendix A — Glossary

| Term | Meaning |
|------|---------|
| ADX | Average Directional Index (trend strength) |
| ATR | Average True Range (volatility / stop sizing) |
| RVOL | Volume vs its recent average |
| VPIN | Volume-Synchronized Probability of Informed Trading |
| TS momentum | Sign of an asset’s own past return |
| OOS | Out-of-sample (data not used to fit rules/model) |
| COVER | Buy back to close a short |

## Appendix B — Declaration

The numerical results in this report are from historical simulation. Past performance does not guarantee future results. This work is for **academic / educational** purposes and is not investment advice.

---

**End of Report**
