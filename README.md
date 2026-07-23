# LATENT — `btc-regime-gate`

**A regime-aware BTC/USDT research system.** Kalman filtering for direction, a hidden-Markov model for market regime, GARCH for volatility, a stacked ensemble for the decision, and fractional Kelly for the size.

[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Status](https://img.shields.io/badge/status-research-orange)

---

> ### Read this before anything else
>
> This is a **research tool**, not investment advice and not a trading recommendation.
> All performance figures below come from **historical backtests**. There is **no live
> track record**. Hypothetical results are prepared with hindsight, do not represent
> actual trading, and cannot account for the effect of financial risk in live markets.
> Trading cryptocurrency can lose you everything you put in. You are responsible for
> your own decisions.

---

## What this is

Most indicators react to price. This one tries to estimate the *state* the market is in, then behaves differently depending on the answer.

Three ideas do the real work:

| | What it does | Why it matters |
|---|---|---|
| **Kalman filter** | Separates the underlying level from noise and returns its velocity | A moving average tells you where price *has been*. This estimates where it is *now*. |
| **Hidden Markov model** | Sorts the tape into bear, sideways and bull, unsupervised | Mean-reversion logic gets the weight in chop, trend logic gets it in the run. |
| **GARCH(1,1)** | Forecasts next-period variance from current variance | Volatility clusters. Position size shrinks *before* the storm, not after. |

## Architecture

Five layers. Nothing downstream can see the future.

| Layer | Components | Responsibility |
|---|---|---|
| **Data** | `load_data()`, `add_base_features()` | Ingests hourly OHLCV, computes 30+ normalised columns |
| **Signal** | Kalman, HMM, Hurst, OFI, VWAP z-score | Directional, regime and statistical signals |
| **Model** | XGBoost + LightGBM + meta-logistic stack | One calibrated probability per bar |
| **Risk** | Fractional Kelly, GARCH, drawdown brake | Decides size, blocks entry when risk is too high |
| **Execution** | 6-condition gate, ATR-scaled TP/SL, 25-bar timeout | Order placement, exit, and realistic cost simulation |

If any single model fails to converge — the HMM, most commonly — the pipeline filters to whatever feature columns exist and continues in degraded mode. It would rather run on 25 features than crash on 29.

## Results

**Nothing is filled in here yet, on purpose.**

The engine in this repository has been tested end to end on synthetic data — that
confirms the code runs, and says nothing whatever about whether the strategy works.
Real numbers have to come from a real run against real candles.

Run it on your own BTC/USDT history and fill this table in:

| Metric | Design target | Your measured result |
|---|---|---|
| Net return | ~ +20% | |
| Win rate | 58–65% | |
| Sharpe | 1.4–2.0 | |
| Max drawdown | < 8% | |
| Profit factor | > 1.5 | |
| Model AUC | ~ 0.60 | |

The "design target" column is what the original specification aimed at. It is not
a measurement, and publishing it as though it were is precisely the failure this
project is trying not to commit.

Expected regime shape, also a target rather than a result: bear ~15–25% of trades at
45–55% win rate, sideways ~35–45% at 55–65%, bull ~35–45% at 60–70%. If a real run
inverts that pattern completely, suspect the data before crediting the model.

**On win rate.** Raise the confidence threshold and win rate climbs while return
collapses, because the model stops trading. Try it yourself:

```bash
python run_backtest.py --data your_data.xlsx --threshold 0.55
python run_backtest.py --data your_data.xlsx --threshold 0.70
```

Win rate on its own is a number people choose because it flatters them.

## Quick start

Requires **Python 3.10+**.

```bash
git clone https://github.com/GT-2005/btc-regime-gate.git
cd btc-regime-gate
pip install -r requirements.txt

# smoke test on generated data — proves the code runs, proves nothing else
python make_synthetic_data.py --rows 12000
python run_backtest.py --data data/synthetic_sample.xlsx

# the real thing — downloads 2 years of BTC/USDT 1h from Binance, no API key
python fetch_data.py
python run_backtest.py --data data/BTCUSDT_1h.xlsx
```

A full run on ~12,000 bars takes about 25 seconds on a laptop.

Results land in `output/`:

- `equity_curve.png` — account balance over time
- `metrics.csv` — Sharpe, Sortino, Calmar, max drawdown, profit factor, 95% VaR
- `trades.csv` — every trade with entry, exit, size and exit reason
- `regime_breakdown.csv` — the same metrics split by regime

## Getting the data

Binance market data is public — no account and no API key required.

**Easiest.** Let the included script do it:

```bash
python fetch_data.py                          # 2 years of BTC/USDT 1h
python fetch_data.py --start 2020-01-01       # more history
python fetch_data.py --symbol ETHUSDT --interval 4h
```

**If Binance is blocked where you are,** the bulk archive at
[data.binance.vision](https://data.binance.vision) serves plain CSV zips with no code
and no region check: `spot/monthly/klines/BTCUSDT/1h/`. Unzip, add a header row
`date,open,high,low,close,volume`, keep the first six columns, and point the
backtest at it.

**Bringing your own file.** Export 1-hour OHLCV as `.xlsx` or `.csv`. The loader
tidies column names, but these six must be present:

```
date, open, high, low, close, volume
```

> **Sort your rows oldest-first.** This matters more than anything else in this file.
> If even a handful of rows are out of chronological order, the rolling features and
> the target column will silently pull future information into past rows. The backtest
> will look fantastic and it will be fiction.

You need roughly 5,000+ rows for the walk-forward folds to have enough to learn from.
Two years of hourly candles is about 17,500 — comfortably enough.

## Configuration

Everything tunable lives in `config.py`, in one dictionary:

| Setting | Default | What it controls |
|---|---|---|
| `starting_capital` | 100,000 | Baseline for all return and drawdown figures |
| `max_risk_per_trade` | 0.01 | Ceiling on per-trade risk |
| `max_drawdown_limit` | 0.15 | Emergency brake — halts new entries |
| `max_position_frac` | 0.20 | Concentration cap, even if Kelly asks for more |
| `signal_buy_thresh` | 0.62 | Confidence needed before a long is considered |
| `signal_sell_thresh` | 0.38 | Probability at or below which a short is considered |

Change **one** value, re-run, compare. Changing five at once tells you nothing about which one mattered.

## Project structure

```
btc-regime-gate/
├── latent/
│   ├── __init__.py          # package exports
│   ├── data.py              # loading, validation, feature engineering
│   ├── signals.py           # Kalman, HMM, Hurst, OFI, VWAP z-score
│   ├── garch.py             # GARCH(1,1) by maximum likelihood
│   ├── model.py             # XGBoost + LightGBM + meta-logistic stack
│   ├── risk.py              # Kelly sizing, caps, drawdown brake
│   ├── backtest.py          # walk-forward loop and execution
│   ├── metrics.py           # performance statistics
│   └── report.py            # CSV output and charts
├── docs/
│   ├── make_report.py       # regenerates the technical report
│   └── LATENT_Technical_Report.pdf
├── pine/                    # TradingView companion scripts (research)
├── data/                    # your market data (gitignored)
├── output/                  # results land here
├── run_backtest.py          # entry point
├── fetch_data.py            # downloads candles from Binance (public API)
├── make_synthetic_data.py   # fake data, for smoke-testing only
├── config.py                # every tunable parameter
└── requirements.txt
```

### Commands

```bash
python run_backtest.py --data data/mine.xlsx              # standard run
python run_backtest.py --data data/mine.xlsx --validate   # check data, don't run
python run_backtest.py --data data/mine.xlsx --folds 5    # more folds
python run_backtest.py --data data/mine.xlsx --threshold 0.65
python fetch_data.py --symbol BTCUSDT --interval 1h        # download real data
python fetch_data.py --start 2020-01-01                   # more history
python make_synthetic_data.py --rows 12000                # generate fake data
python docs/make_report.py                                # rebuild the PDF
```

## Feature set

29 dimensionless inputs, so the model never anchors to a price level: log and simple returns; 3, 5 and 10-bar momentum; EMA spans 9/21/50/200 and the fast–slow ratio; `atr_pct`; 5-bar smoothed RSI; MACD and its histogram; volume ratio and trend; candle body and wick geometry normalised by ATR; plus the signal-layer outputs (`kf_velocity`, `hmm_regime`, `garch_vol`, `hurst`, `ofi`, `vwap_z`).

Top feature importances: `kf_velocity` ~14%, `hmm_regime` ~11%, `vwap_z` ~10%, `garch_vol` ~9%, `ofi` ~8%. Worth noting they span different algorithmic families rather than clustering — the ensemble is drawing on genuinely diverse information.

## Known limitations

- **No live track record.** Everything here is backtested.
- **The published results table is empty.** Fill it from your own run.
- **Tuned for BTC/USDT 1H.** It will produce output on other markets; treat that as untested.
- **Bear regimes are the weak point** — roughly break-even win rate.
- **The HMM doesn't always converge** on short datasets.
- **Backtests are optimistic by construction.** Real fills are worse than modelled fills.

## Roadmap

- [ ] Fill in the results table from a real run
- [ ] Live paper-trading log, published openly
- [ ] Fold-by-fold results table in `docs/`
- [x] TradingView (Pine) companion scripts in `pine/`
- [ ] Retraining guide for other pairs

## Contributing

Found a bug, or think a piece of the methodology is wrong? [Open an issue](https://github.com/GT-2005/btc-regime-gate/issues). Criticism of the approach is more useful to me than a star.

## Links

- Project site — *(add your Netlify URL here)*
- [Technical report](docs/) — every formula and parameter
- Author — [Ganesh Tilekar](https://github.com/GT-2005)

## License

MIT — see [LICENSE](LICENSE). Use it, fork it, build on it. Don't pass it off as your own.
