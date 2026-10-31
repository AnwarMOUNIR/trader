# S&P 500 Autonomous Paper Trading Cloud Runner

Production-grade automated quantitative trading system running continuously on GitHub Actions cloud infrastructure.

## Strategy Architecture

1. **Universe:** 30 high-liquidity S&P 500 equities.
2. **Macro Trend Gate (Moreira & Muir / AQR):** SPY 200-day Simple Moving Average.
   - SPY > 200 SMA $\to$ Bull regime.
   - SPY < 200 SMA $\to$ Bear regime (100% Cash Lock).
3. **CPPI Dynamic Drawdown Dial (Grossman & Zhou):**
   - Tracks portfolio high-water mark against current account equity.
   - Drawdown $\ge \$6,000 \to$ De-lever gross exposure to 40%.
   - Drawdown $\ge \$8,500 \to$ 100% Cash Lock (protecting against the -$10k loss ceiling).
4. **Stock Selection:** Filters for stocks trading above their 200 SMA and selects the top 8 by 60-day relative momentum.
5. **Position Sizing:** Equal weight ($\approx 12.5\%$ per stock, capped at $15\%$ max notional).

## Cloud Execution via GitHub Actions

- **Schedule:** Runs every 30 minutes during US regular exchange hours (Monday through Friday, 13:30 to 21:00 UTC).
- **Manual Trigger:** Supports `workflow_dispatch` for immediate execution via GitHub Web UI or GitHub CLI (`gh workflow run`).
- **Zero Local Footprint:** Executes entirely on GitHub cloud runners without requiring a local machine to be powered on.
