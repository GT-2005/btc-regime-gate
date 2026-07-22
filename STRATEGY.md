# Strategy status v2.7

## Position rule (hard)
- **Only 1 open trade** at a time (long **or** short).
- Long: **BUY** entry → hold → **EXIT** (sell).
- Short: **SELL** entry → hold → **COVER** (buy back).
- No new entry until flat + cooldown (default 16 bars on Pine).
- Pine: use `pine/BTC_Regime_Gate_v27.pine` (remove old v2.3 from chart).

## What we tried for higher returns
1. Breakout Playbook C + higher risk → **lost ~15%**
2. Pure ML spam entries → **lost ~50%+**
3. Aggressive VPIN block → too few trades; optional `--vpin` only
4. Restored proven A/B + **2% risk** + short_only

## Research folded in (KEEP papers)
| Idea | How we use it |
|------|----------------|
| Time series momentum | Side filter: long only if ~60d return > 0; short if < 0 |
| VPIN / volume clock | Optional `--vpin` toxicity stand-aside |
| AFML / Deflated Sharpe | Walk-forward + honesty; don’t chase fake 15% |
| One-position discipline | Matches OTR / hyperparameter caution |

## Current defaults
| Setting | Value |
|---------|--------|
| Mode | **short_only** (or Auto) |
| SL / TP | 1 ATR / 3 ATR |
| Risk | **2%** per trade |
| Breakout C | **OFF** |
| TS momentum | **ON** |
| VPIN | OFF (opt-in) |
| ML gate | OFF (opt-in) |

## Honest results ($100k / year, plain short_only)
Best stack still ~**+1%** on 2025 vs B&H −6%; sum across 2019–22+24–25 ~**+$1k**. Not 15%.

## TradingView
1. Remove old “BTC Regime Gate v2.3”
2. Add `BTC_Regime_Gate_v27.pine`
3. You should see **SELL** then later **COVER**, never stacked SELLs with no cover while Pos shows SHORT.
