# Strategy lockfile — v2.7

Use this when explaining the system quickly.

## Position rule
- Max **1** open trade (long **or** short).
- Long: **BUY → EXIT**
- Short: **SELL → COVER**
- New entry only after flat + cooldown.

## Defaults that worked best in tests
| Setting | Value |
|---------|--------|
| Mode | `short_only` |
| SL / TP (trend) | 1 ATR / 3 ATR |
| Risk | 2% per trade |
| Breakout C | OFF |
| TS momentum | ON |
| VPIN / ML | optional |

## Do not resurrect
- Breakout spam / Playbook C  
- Pure ML probability entries  
- Ultra-aggressive VPIN that blocks almost all bars  

## Honest takeaway
Small fee-aware edge in some years; **not** a guaranteed 15% system.
Full numbers: [README](README.md) · [Report PDF](docs/PROJECT_REPORT.pdf)
