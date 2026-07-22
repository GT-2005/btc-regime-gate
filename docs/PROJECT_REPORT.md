# BTC/USDT INSTITUTIONAL TRADING SYSTEM
## Complete Code & Parameter Deep-Dive  
### Technical Report — Version 2.0 (Improved)

**10 Hedge Fund Algorithms · 30+ Features · XGBoost + LightGBM · Kalman · HMM · GARCH · Kelly Criterion Sizing · Walk-Forward Validated · TradingView Deployment**

---

**Author**  
Ganesh Tilekar

**Date:** July 2026  
**Version:** 2.0 (Improved)

---

## TABLE OF CONTENTS

1. Introduction and System Overview  
2. Software Dependencies and Data Pipeline  
3. Feature Engineering  
4. Kalman Filter  
5. Hidden Markov Model  
6. GARCH(1,1) Volatility Model  
7. Hurst Exponent  
8. Order Flow Imbalance  
9. VWAP Deviation and Z-Score  
10. Ensemble Machine Learning  
11. Walk-Forward Optimization  
12. Kelly Criterion and Position Sizing  
13. Entry and Exit Logic  
14. Risk Metrics and Backtest Reporting  
15. System Improvements in Version 2.0 *(new)*  
16. TradingView Deployment *(new)*  
17. Conclusion  
18. References  

---

## Chapter 1: Introduction and System Overview

### 1.1 Background

Over the past decade, cryptocurrency markets have drawn serious attention from institutional investors, and the BTC/USDT pair sits at the centre of that interest. These markets run around the clock, they swing hard, and their microstructure still carries enough inefficiency to reward well-designed systematic strategies. Capturing those rewards demands a multi-layered system that brings together signal processing, statistical modelling, machine learning, and disciplined risk management.

This report documents an institutional-grade BTC/USDT trading system built around ten distinct hedge-fund-style algorithms, more than thirty engineered features, an ensemble of XGBoost and LightGBM with meta-logistic stacking, and a position sizing framework grounded in the Kelly Criterion. Version **2.0** keeps this same architecture and **improves** it with research-backed filters, corrected risk/reward geometry, one-position discipline, longer-history validation, and a TradingView chart layer — without replacing the original multi-algorithm design.

### 1.2 Objectives

- Build a noise-filtered, regime-aware signal pipeline using Kalman Filters for directional estimation, Hidden Markov Models for regime classification, and GARCH for dynamic volatility forecasting.  
- Engineer a rich feature set of dimensionless, scale-invariant indicators from raw OHLCV bars.  
- Train a stacked ensemble (XGBoost + LightGBM + Meta-Logistic Regression) that outputs calibrated trade probabilities.  
- Validate predictions through walk-forward optimisation with no future leakage.  
- Size positions using fractional Kelly Criterion with hard risk caps and drawdown circuit breakers.  
- Report Sharpe, Sortino, Calmar, maximum drawdown, profit factor, and related metrics.  
- **(v2.0)** Enforce one open position at a time; improve exit R:R; add optional VPIN / time-series momentum gates; deploy matching logic on TradingView.

### 1.3 System Architecture

At a high level the system is a five-stage pipeline. Raw data enters on the left; sized trade decisions exit on the right. No stage may look ahead into future data.

| Layer | Key Components | What It Does |
|-------|----------------|--------------|
| **Data Layer** | OHLCV CSV/Excel, date sort, `add_base_features()` | Ingests Binance-style bars and builds 30+ indicator columns |
| **Signal Layer** | Kalman, HMM, Hurst, OFI, VWAP, Z-Score | Directional, regime, and statistical signals |
| **ML Layer** | XGBoost, LightGBM, Meta-Logistic, Walk-Forward | Calibrated probability per bar |
| **Risk Layer** | Kelly, GARCH, drawdown control, position cap | How much capital to allocate; block when risk is high |
| **Execution Layer** | Multi-condition entry gate, ATR TP/SL, time stop, fees | Order simulation with realistic costs |

**v2.0 addition:** a parallel **Chart Layer** (Pine Script) mirrors playbook / regime decisions for live monitoring and alerts.

If any single algorithm fails (for example HMM does not converge), the pipeline degrades gracefully rather than crashing.

---

## Chapter 2: Software Dependencies and Data Pipeline

### 2.1 Python Libraries

| Library | Category | Role |
|---------|----------|------|
| NumPy | Numeric | Array math across every algorithm |
| Pandas | Data | DataFrames, rolling windows, feature engineering |
| SciPy | Stats / Opt | HMM log-space maths; GARCH MLE |
| XGBoost | ML | Gradient-boosted trees (base model A) |
| LightGBM | ML | Histogram leaf-wise trees (base model B) |
| scikit-learn | ML | Meta-logistic, scaling, AUC |
| joblib | IO | Persist trained models |
| openpyxl | IO | Excel OHLCV (optional) |

### 2.2 Data Loading and Preprocessing

The research stack accepts BTC/USDT OHLCV from Binance-style **Excel (1H)** or **CSV (15m)**. Columns are normalised to `date/open/high/low/close/volume`, timestamps parsed, and rows **sorted ascending**. Sorting is non-negotiable: out-of-order bars silently leak future information into rolling features.

**v2.0 data expansion:** full 15-minute history (~2017–2026, ~310k bars) is used for yearly and walk-forward checks in addition to the original 1H research path.

### 2.3 Global Configuration (CFG)

| Parameter | v1.0 | v2.0 (improved) | What It Controls |
|-----------|------|-----------------|------------------|
| `starting_capital` | 100,000 | 100,000 | Baseline equity |
| `max_risk_per_trade` | 1% | **2%** (research default) | Per-trade risk ceiling |
| `max_drawdown_limit` | 15% | 15% | Halt new entries |
| `max_position_frac` | 20% | 25% | Concentration cap |
| `signal_buy_thresh` | 0.62 | 0.60–0.62 | Long ML probability |
| `signal_sell_thresh` | 0.38 | 0.38–0.40 | Short ML probability |
| `atr_tp_mult` | 1.5 | **3.0** (trend playbook) | Take-profit × ATR |
| `atr_sl_mult` | 2.0 | **1.0** (trend playbook) | Stop-loss × ATR |
| `max_hold_bars` | 25 (1H) | 96 (15m trend) / 16–25 range | Time stop |
| `slippage_bps` | 5 | 9 round-trip style | Fill friction |
| `commission_bps` | 4 | 4 one-way | Fees |
| `one_position_only` | implicit | **hard rule** | No pyramiding |

**Why TP/SL changed:** v1.0 used TP 1.5 ATR / SL 2.0 ATR (reward < risk). That geometry needs an unrealistically high win rate after costs. v2.0 trend exits use **TP 3 / SL 1** so winners can pay for losers.

---

## Chapter 3: Feature Engineering

Feature engineering turns raw OHLCV into inputs the signal and ML layers can learn from. Features are normalised or dimensionless wherever possible.

### 3.1 Returns and Momentum

Percentage returns, log-returns, and multi-horizon momentum (3 / 5 / 10 bars) feed HMM and GARCH and enter the ML vector. EMAs at 9, 21, 50, 200 plus `ema_fast_slow` and `trend_up` capture multi-scale trend.

### 3.2 ATR, RSI, and MACD

ATR (and `atr_pct`) scales stops and targets. RSI(14) and smoothed RSI act as soft filters (avoid extreme stretch). MACD and MACD histogram measure momentum change.

### 3.3 Volume and Candle Structure

`vol_ratio`, `vol_trend`, `body_pct`, wick ratios describe participation and bar anatomy.

### 3.4 Target Column

Supervised target remains causal: whether price improves over a forward horizon (e.g. next 5 bars on 1H research, or triple-barrier / ATR path labels on 15m research). Targets are **never** available to the live backtest loop.

### 3.5 Feature Set (~29+ research features)

| Category | Examples | Count |
|----------|----------|-------|
| Price / Returns | returns, log_ret, momentum3/5/10 | 5 |
| Oscillators | rsi, rsi_smooth, macd, macd_hist | 4 |
| Trend | ema_fast_slow, atr_pct, trend_up | 3 |
| Volume | vol_ratio, vol_trend, body_pct | 3 |
| Kalman | kf_velocity, kf_signal | 2 |
| Order Flow | ofi, ofi_signal, cvd | 3 |
| Statistical | vwap_z, zscore, bb_pct | 3 |
| Regime | hurst, is_trending, is_meanrev, hmm_regime, garch_vol | 5+ |

**v2.0 expansion (optional training path):** up to ~100+ pattern features (multi-horizon returns, HTF RSI/EMA, session flags) for LightGBM research — used as a soft gate, not a replacement for the ten-algorithm core.

---

## Chapter 4: Algorithm 1 — Kalman Filter

### 4.1 Theory

A Kalman Filter is a recursive Bayesian estimator of a hidden state. Here the state is `[price, velocity]`. Predict and update steps blend the model forecast with the noisy close via Kalman gain.

### 4.2 Parameters

| Parameter | Typical value | Effect |
|-----------|---------------|--------|
| `process_var` | 1e-4 | Smaller → smoother, slower |
| `obs_var` | 1e-2 | Larger → trust model more than raw ticks |

### 4.3 Trading use

`kf_velocity > 0` required for longs; `< 0` for shorts — blocks ML signals that fight the noise-filtered trend.

---

## Chapter 5: Algorithm 2 — Hidden Markov Model

### 5.1 Theory

Three hidden regimes (Bear / Sideways / Bull) with Gaussian emissions on returns, fitted by Baum–Welch (EM) in log-space.

### 5.2 Trading use

- **Bull** → prefer trend-following longs  
- **Bear** → prefer trend-following shorts  
- **Sideways** → mean-reversion using VWAP / z-score stretch  

HMM is trained on the early window and applied forward without refitting on future test folds inside a fold.

---

## Chapter 6: Algorithm 3 — GARCH(1,1)

### 6.1 Model

σ²(t) = ω + α ε²(t−1) + β σ²(t−1), with α + β < 1, fitted by maximum likelihood.

### 6.2 Trading use

`garch_vol` feeds Kelly sizing (smaller size when volatility is high). `vol_regime` (vol / rolling mean) above a shock threshold **blocks new entries**.

---

## Chapter 7: Algorithm 4 — Hurst Exponent

Rolling R/S analysis (~100 bars):

- H > 0.55 → trending  
- H < 0.45 → mean-reverting  
- 0.45–0.55 → dead zone (no forced strategy)

Hurst works with HMM to choose trend-follow vs fade.

---

## Chapter 8: Algorithm 5 — Order Flow Imbalance

OHLCV proxy:

`buy_vol = volume × (close − low) / (high − low)`  
`sell_vol = volume × (high − close) / (high − low)`  
`OFI = (buy_vol − sell_vol) / volume`

Confirmation gate: longs need non-negative OFI signal; shorts need non-positive.

---

## Chapter 9: Algorithms 6–7 — VWAP Deviation and Z-Score

Rolling VWAP and `vwap_z` locate price vs institutional fair value. Mean-reversion longs prefer deep negative `vwap_z`; shorts prefer large positive stretch. Bollinger `%B` (`bb_pct`) adds a scale-free stretch measure for ML.

---

## Chapter 10: Algorithm 8 — Ensemble Machine Learning

### 10.1 XGBoost + LightGBM

Two base classifiers (hundreds of trees, shallow depth, low learning rate, row/column subsampling, L1/L2 regularisation).

### 10.2 Meta-learner

Logistic regression trained on **validation** probabilities (not the base training set) to blend the two models into one calibrated probability.

### 10.3 Resilience

`_safe_features()` drops missing columns if HMM/Hurst fail so the system continues in degraded mode.

---

## Chapter 11: Walk-Forward Optimization

Data split example: 60% train / 20% validation / 20% test, with the test segment rolled into multiple expanding folds. Each fold trains a fresh ensemble. Only out-of-sample probabilities are traded. A large gap between validation and test AUC is treated as an overfitting warning.

**v2.0:** additional year-by-year walk-forward on 15m history (train past years → test next year) for LightGBM research models.

---

## Chapter 12: Kelly Criterion and Position Sizing

Fractional Kelly (e.g. 25% of full Kelly) plus hard caps:

1. Fractional Kelly shrink  
2. Max risk per trade (1–2%)  
3. Max position fraction (20–25%)  

GARCH volatility further reduces size in stressed markets.

---

## Chapter 13: Entry and Exit Logic

### 13.1 Long entry (all required in core system)

1. ML probability ≥ buy threshold  
2. Context strategy OK (trend bull setup **or** mean-reversion stretch)  
3. Kalman velocity > 0  
4. OFI confirmation ≥ 0  
5. RSI not at extreme (soft band)  
6. Volatility shock not active  

### 13.2 Short entry

Mirror image with sell threshold and bear / overbought context.

### 13.3 Exits

| Mechanism | v1.0 | v2.0 trend path |
|-----------|------|-----------------|
| Take profit | 1.5 × ATR | **3.0 × ATR** |
| Stop loss | 2.0 × ATR | **1.0 × ATR** (+ BE / trail) |
| Time stop | 25 bars (1H) | Regime-specific max hold |
| Costs | slippage + commission | same philosophy |

### 13.4 One-position rule (v2.0 hard constraint)

- Long cycle: **BUY → EXIT (sell)**  
- Short cycle: **SELL → COVER (buy back)**  
- No second entry until flat + cooldown  

This prevents signal spam and keeps risk accounting meaningful.

---

## Chapter 14: Risk Metrics and Backtest Reporting

| Metric | What it tells you |
|--------|-------------------|
| Sharpe / Sortino | Return per unit of (downside) risk |
| Calmar | Return vs worst drawdown |
| Max drawdown | Deepest equity hole |
| Profit factor | Gross wins / gross losses |
| VaR (95%) | Tail loss context |

Performance is also reviewed by regime (Bull / Sideways / Bear) when HMM labels are available.

### 14.1 Measured results (v2.0 fee-aware yearly checks, short-priority)

Using $100,000 start, fees/slippage on, one position, improved R:R:

| Year | Approx. return | Notes |
|------|----------------|-------|
| 2019 | −0.4% | Missed full bull upside (by design, cautious) |
| 2020 | +1.5% | Positive |
| 2021 | +0.4% | Small positive |
| 2022 | −1.4% | Bear year; still far better than B&H ~−64% path |
| 2024 | −0.2% | Flat-ish |
| 2025 | +1.1% | Beat B&H ~−6% |

**Interpretation:** the system is a **risk-managed research stack**, not a guaranteed +20% product. Version 1.0 *target profile* (+20%, Sharpe 1.4–2.0) remains an aspirational institutional benchmark; Version 2.0 reports **what we actually measured** after costs on expanded data.

### 14.2 Failed experiments (documented)

| Experiment | Result | Action |
|------------|--------|--------|
| Aggressive breakout add-on | Large losses | Disabled |
| Pure ML probability spam entries | Severe OOS losses | Rejected |
| Over-strict VPIN block | Almost no trades | Softened; optional |

---

## Chapter 15: System Improvements in Version 2.0

Version 2.0 **does not replace** the ten-algorithm institutional design. It **adds** the following:

| Improvement | Source / motivation | Effect |
|-------------|---------------------|--------|
| Corrected trend R:R (TP 3 / SL 1) | Expectancy geometry | Winners can fund losers |
| Hard one-position FSM | Execution hygiene | No pyramiding / spam |
| Time-series momentum side filter | Moskowitz et al. | Align side with ~60d return sign |
| Optional VPIN / volume-clock gate | Easley–LdP–O’Hara | Stand aside in toxic flow |
| Longer 15m history + yearly OOS | Data expansion | Harder, more honest tests |
| LightGBM pattern research path | AFML-style labeling | Soft filter, not sole brain |
| TradingView Pine mirror | Deployment | Live visualisation + alerts |
| Literature triage (KEEP / SKIP) | Research process | Avoid wrong tools (CAPM-only, FinRL-first, etc.) |

---

## Chapter 16: TradingView Deployment

| Item | Detail |
|------|--------|
| Script | `pine/BTC_Regime_Gate_v27.pine` |
| Timeframe | BTCUSDT 15m (HTF 1h) |
| Markers | BUY / EXIT · SELL / COVER |
| Table | Regime, HTF, Score, ADX, Pos, Why |
| Rule | Pos = SHORT ⇒ no new SELL until COVER |

This layer is for **monitoring and alerts**. The Python research stack remains the authority for quantitative backtests.

---

## Chapter 17: Conclusion

This report documents a ten-algorithm BTC/USDT trading system: Kalman denoising, HMM regimes, GARCH volatility, Hurst strategy selection, OFI confirmation, VWAP/Z-score dislocation, stacked ML probabilities, walk-forward validation, and Kelly sizing.

Version 2.0 keeps that institutional picture intact and improves it where Version 1.0 was weakest: inverted risk/reward, multi-signal spam, limited chart deployment, and optimistic reporting without enough failed-path documentation. The philosophy remains **quality over quantity** — most bars are skipped; capital is committed only when layered evidence agrees.

**Limitations remain:** OFI is an OHLCV proxy; GARCH understates tails; backtests omit large market impact; measured fee-aware yearly returns are modest. The framework is a solid foundation for further research and engineering work — not a promise of fixed annual profit.

---

## Chapter 18: References

[1] Kalman, R.E. (1960). A New Approach to Linear Filtering and Prediction Problems.  
[2] Baum, L.E. et al. (1970). A Maximization Technique… Markov Chains.  
[3] Rabiner, L.R. (1989). A Tutorial on Hidden Markov Models.  
[4] Bollerslev, T. (1986). Generalized Autoregressive Conditional Heteroskedasticity.  
[5] Engle, R.F. (1982). Autoregressive Conditional Heteroscedasticity.  
[6] Hurst, H.E. (1951). Long-Term Storage Capacity of Reservoirs.  
[7] Mandelbrot, B.B. & Van Ness, J.W. (1968). Fractional Brownian Motions.  
[8] Peters, E.E. (1994). Fractal Market Analysis.  
[9] Chen, T. & Guestrin, C. (2016). XGBoost.  
[10] Ke, G. et al. (2017). LightGBM.  
[11] Wolpert, D.H. (1992). Stacked Generalization.  
[12] Kelly, J.L. (1956). A New Interpretation of Information Rate.  
[13] Thorp, E.O. (2006). The Kelly Criterion…  
[14] Wilder, J.W. (1978). New Concepts in Technical Trading Systems.  
[15] Bollinger, J. (2002). Bollinger on Bollinger Bands.  
[16] Easley, D., López de Prado, M. & O’Hara, M. (2012). Flow Toxicity and Liquidity…  
[17] Easley, D., López de Prado, M. & O’Hara, M. (2012). The Volume Clock.  
[18] López de Prado, M. (2018). Advances in Financial Machine Learning.  
[19] Bailey, D.H. et al. (2014). Pseudo-Mathematics and Financial Charlatanism / Deflated Sharpe.  
[20] Moskowitz, T.J., Ooi, Y.H. & Pedersen, L.H. (2012). Time Series Momentum.  
[21] Carr, P. & López de Prado, M. (2014). Determining Optimal Trading Rules without Backtesting.  

---

## Appendix A — Repository Map

| Path | Role |
|------|------|
| `python/backtest_regime_gate.py` | Fee-aware backtest / yearly runner / improved execution rules |
| `python/train_btc_patterns.py` | Extended feature + LightGBM research |
| `python/vpin_volume.py` | Volume clock / VPIN proxy |
| `pine/BTC_Regime_Gate_v27.pine` | Chart deployment |
| `docs/PROJECT_REPORT.pdf` | This report |
| `STRATEGY.md` | Locked improvement notes |

---

## Appendix B — Declaration

Simulated results only. Past performance does not guarantee future results. This report is for research and portfolio demonstration and is not investment advice.

---

**End of Report — Version 2.0**
