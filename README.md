# BTC Regime Gate

### Regime-adaptive BTC/USDT trading system  
**Python research backtester + TradingView Pine indicator**

| | |
|---|---|
| **Author** | Ganesh Tilekar |
| **Stack** | Python · Pine Script v6 · LightGBM (optional) |
| **Market** | BTCUSDT · 15-minute (+ 1h / daily context) |
| **Version** | 2.7 |
| **Report** | [Project Report (PDF)](docs/PROJECT_REPORT.pdf) · [Markdown](docs/PROJECT_REPORT.md) |

> **Interview one-liner:** I built an institutional-style BTC/USDT system (Kalman, HMM, GARCH, Hurst, OFI, stacked ML, Kelly sizing, walk-forward) and improved it with better risk/reward, one-position discipline, research filters, multi-year fee-aware tests, and a TradingView deployment layer — with honest measured results, not hype.

---

## Why this project exists

Most retail “indicator stacks” fail for three reasons:

1. They treat **trend and range** the same way  
2. They allow **signal spam** (many sells while already short)  
3. They report **optimistic backtests** (no fees, no failed ideas)

This project fixes those with a clear engineering loop: **design → backtest → kill bad ideas → deploy**.

---

## What I built

```text
┌─────────────────────┐     ┌──────────────────────────┐
│  Layer A: Research  │     │  Layer B: Live chart     │
│  Python backtester  │────▶│  TradingView Pine v2.7   │
│  ML / VPIN optional │     │  Alerts + position state  │
└─────────────────────┘     └──────────────────────────┘
            │
            ▼
   Multi-year BTC 15m OHLCV
   Fees + slippage + yearly OOS checks
```

### Core trading logic

| Piece | Behavior |
|-------|----------|
| **Regime** | Trend / Range / Shock using ADX + ATR |
| **Playbook A** | Trend pullback (EMA + higher-timeframe alignment) |
| **Playbook B** | Mean reversion (VWAP z-score + RSI) |
| **Playbook C** | Breakouts — **disabled** after it lost money in tests |
| **Position rule** | **One trade only** — Long: BUY→EXIT · Short: SELL→COVER |
| **Risk** | 2% equity risk · SL 1 ATR · TP 3 ATR (trend) |
| **Filters** | Time-series momentum (default ON) · optional VPIN · optional ML gate |

---

## Results (honest)

Starting capital **$100,000** per year · fees/slippage on · mode **`short_only`**

| Year | Strategy return | Approx. profit | Buy & hold (context) |
|------|-----------------|----------------|----------------------|
| 2019 | −0.4% | −$417 | Strong bull |
| 2020 | **+1.5%** | +$1,460 | Strong bull |
| 2021 | +0.4% | +$404 | Bull |
| 2022 | −1.4% | −$1,369 | Bear (B&H ~−64%) |
| 2024 | −0.2% | −$182 | Strong bull |
| 2025 | **+1.1%** | +$1,122 | B&H ~−6% |
| **Sum** | | **~$+1,000** | |

### How to read this in an interview

- Edge is **small but real** in some years (e.g. 2025 vs buy-and-hold down).  
- It **does not** capture full bull-market upside — by design it is cautious / short-biased.  
- I rejected paths that looked “more profitable” but failed OOS (see below).

### Ideas I tested and rejected

| Idea | Outcome | Decision |
|------|---------|----------|
| Breakout playbook + higher risk | Large losses (~−15% class) | Disabled |
| Pure ML entry spam | Catastrophic OOS losses | Rejected |
| Over-aggressive VPIN filter | Almost no trades | Softened; opt-in only |

This matters more than a fake +15% curve: it shows **research discipline**.

---

## Repository map

```text
btc-regime-gate/
├── README.md                          ← you are here
├── STRATEGY.md                        ← locked rules + lessons
├── docs/
│   ├── PROJECT_REPORT.pdf             ← full write-up for submission
│   ├── PROJECT_REPORT.md
│   └── TECHNICAL_REPORT_v2.md         ← earlier design notes
├── pine/
│   ├── BTC_Regime_Gate_v27.pine       ← ★ use this on TradingView
│   └── BTC_Regime_Gate_v21.pine       ← older reference
└── python/
    ├── backtest_regime_gate.py        ← main engine + --years runner
    ├── train_btc_patterns.py          ← features + LightGBM walk-forward
    ├── vpin_volume.py                 ← volume clock / VPIN proxy
    ├── backtest_ml_2025.py            ← honest OOS ML experiment
    ├── requirements.txt
    ├── BTC_Regime_Gate_Backtest_Colab.ipynb
    └── models_btc15m/                 ← models + yearly CSVs
```

---

## Tech stack

| Area | Tools |
|------|--------|
| Data / backtest | Python 3, pandas, NumPy |
| Indicators / risk | ATR, ADX, EMA, VWAP z, RVOL |
| Optional ML | LightGBM, scikit-learn, joblib |
| Microstructure research | VPIN / volume buckets (bar-level proxy) |
| Charting | TradingView Pine Script v6 |
| Validation | Year-by-year runs, walk-forward training, fee model |

**Research influences (selected):** Volume Clock, VPIN, Time-Series Momentum (Moskowitz et al.), AFML practices (triple-barrier / walk-forward), Deflated Sharpe / backtest-overfitting caution.

---

## Quick start

### 1) Clone & install

```bash
git clone https://github.com/GT-2005/btc-regime-gate.git
cd btc-regime-gate
python3 -m pip install -r python/requirements.txt
```

macOS + LightGBM (if needed):

```bash
brew install libomp
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib
```

### 2) Data (not in git — file is large)

Use your own Binance-style BTCUSDT **15m** CSV, then:

```bash
python3 python/backtest_regime_gate.py \
  --data /path/to/BTCUSDT_15m_All.csv \
  --mode short_only \
  --years 2019,2020,2021,2022,2024,2025
```

Useful flags:

| Flag | Meaning |
|------|---------|
| `--mode short_only\|auto\|long_only\|both` | Side policy |
| `--years 2019,2025` | Fresh $100k per year |
| `--vpin` | Toxicity stand-aside filter |
| `--ml` | Optional ML probability gate |
| `--compare` | Compare all modes on one dataset |

### 3) TradingView (live view)

1. Open [Pine Editor](https://www.tradingview.com/)  
2. Paste [`pine/BTC_Regime_Gate_v27.pine`](pine/BTC_Regime_Gate_v27.pine)  
3. Add to **BTCUSDT · 15 minutes**  
4. Check the on-chart table: **Pos = FLAT / LONG / SHORT**  
5. Valid short cycle: **SELL → COVER** (never stacked SELLs while already short)

---

## Design decisions (good interview answers)

**Q: Why one position only?**  
So risk is defined (2% per trade), exits are clear, and the chart matches the backtest. Pyramiding hides drawdowns.

**Q: Why short_only as default?**  
On the tested sample it was the most stable dollar outcome vs mixed long/short. Auto mode often diluted results.

**Q: Why keep ML optional?**  
Walk-forward AUC was weak (~0.51–0.55). Rules carry the edge; ML is a soft filter, not the brain.

**Q: Did you overfit for 15%?**  
No. Papers on backtest overfitting / Deflated Sharpe guided me to prefer small honest edges over optimized fairy tales.

---

## Project report

For the full technical write-up (architecture, algorithms, methodology, results, references):

- **PDF:** [`docs/PROJECT_REPORT.pdf`](docs/PROJECT_REPORT.pdf)  
- **Source:** [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md)

---

## Disclaimer

This repository is for **research and portfolio demonstration**. Historical simulations are not live trading results. Cryptocurrency trading can lead to substantial losses. Nothing here is investment advice.

---

## License

MIT — see [`LICENSE`](LICENSE)
