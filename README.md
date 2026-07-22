# BTC Regime Gate

**Student:** Ganesh Tilekar (UI23EC21)  
**Topic:** Bitcoin (BTC/USDT) regime-adaptive trading system with Python backtests and TradingView deployment  

> Educational / research project. Not financial advice. Past backtests ≠ future profits.

---

## What this repo contains

| Path | Description |
|------|-------------|
| [`docs/PROJECT_REPORT.pdf`](docs/PROJECT_REPORT.pdf) | Full project report (PDF) |
| [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md) | Same report in Markdown |
| [`pine/BTC_Regime_Gate_v27.pine`](pine/BTC_Regime_Gate_v27.pine) | TradingView indicator (one position only) |
| [`python/backtest_regime_gate.py`](python/backtest_regime_gate.py) | Main backtest engine |
| [`python/train_btc_patterns.py`](python/train_btc_patterns.py) | ML feature + LightGBM training |
| [`python/vpin_volume.py`](python/vpin_volume.py) | Volume clock / VPIN proxy |
| [`python/models_btc15m/`](python/models_btc15m/) | Trained models + yearly result CSVs |
| [`STRATEGY.md`](STRATEGY.md) | Locked strategy notes (honest results) |

---

## Strategy (v2.7) in one minute

- **Regimes:** Trend / Range / Shock (ADX + ATR)
- **Playbook A:** Trend pullback (EMA + HTF)
- **Playbook B:** Mean-reversion (VWAP z + RSI)
- **Playbook C:** Breakout — **disabled** (lost money in tests)
- **Position rule:** Only **one** long **or** short at a time  
  - Long: BUY → EXIT  
  - Short: SELL → COVER
- **Risk:** 2% per trade, SL 1 ATR / TP 3 ATR (trend)
- **Extras:** Time-series momentum filter (ON); optional `--vpin`, `--ml`

**Best tested stack:** `short_only` — about **+1%** on 2025 vs buy-and-hold ~−6% (not a 15% system).

---

## Setup

```bash
git clone <your-repo-url>
cd "College Project"   # or your folder name
python3 -m pip install -r python/requirements.txt
```

### Data (not in GitHub — too large)

Place your 15m BTCUSDT CSV locally, e.g.:

`/Users/<you>/BTC_DATA/BTCUSDT_15m_All.csv`

Or pass any path:

```bash
python3 python/backtest_regime_gate.py \
  --data /path/to/BTCUSDT_15m_All.csv \
  --mode short_only \
  --years 2019,2020,2021,2022,2024,2025
```

### TradingView

1. Open Pine Editor  
2. Paste [`pine/BTC_Regime_Gate_v27.pine`](pine/BTC_Regime_Gate_v27.pine)  
3. Add to **BTCUSDT 15m** chart  
4. Remove older Regime Gate versions  

---

## Requirements

See [`python/requirements.txt`](python/requirements.txt).

Main packages: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `joblib`, `scipy`, `openpyxl`.

On macOS, LightGBM may need:

```bash
brew install libomp
export DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib
```

---

## Research notes

KEEP papers used in design (volume clock, VPIN, Deflated Sharpe, AFML ideas, time-series momentum, etc.) are listed in the project report. Raw copyrighted PDFs are **not** uploaded.

---

## Disclaimer

Simulated results only. Crypto trading involves substantial risk of loss.
