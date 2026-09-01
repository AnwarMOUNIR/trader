# BRIEFING — 2026-08-31T16:01:16Z

## Mission
Implement Milestone M4: High-Frequency Intraday Backtester & Paper Simulator meeting all quantitative and engineering requirements.

## 🔒 My Identity
- Archetype: teamwork_preview_worker
- Roles: implementer, qa, specialist
- Working directory: c:\Users\Anwar\Documents\trader\.agents\m4_worker_1
- Original parent: 221d5d39-4a0b-46d8-8d77-134da45be38f
- Milestone: M4 (High-Frequency Intraday Backtester & Paper Simulator)

## 🔒 Key Constraints
- Genuine implementation with no hardcoded shortcuts, dummy facades, or mock return tricks.
- Discrete-event bar-by-bar simulation engine (5m / 60m).
- Almgren-Chriss market impact model (temporary non-linear + permanent linear) + dynamic half-spreads + commission tiers.
- Casablanca Stock Exchange (BVC) adapter: +/-10% circuit breaker clamping, T+3 settlement queue, 15% IR tax withholding on net capital gains.
- Complete financial & risk metrics: CAGR, MDD, Drawdown Duration, Sharpe, Sortino, Calmar, Win Rate, Profit Factor, DSR (Bailey & de Prado 2014).
- Alpaca Paper Trading REST API integration via alpaca-py with bracket orders (TP @ 1.5 ATR, SL @ 1.0 ATR capped at 4.8%), idempotent position tracking, autonomous dry run fallback.
- Scheduled automation and dry run telemetry saving snapshots to `output/`.
- 100% passing test suite in `tests/test_backtest_engine.py`.

## Current Parent
- Conversation ID: 221d5d39-4a0b-46d8-8d77-134da45be38f
- Updated: 2026-08-31T16:01:16Z

## Task Summary
- **What to build**: Full M4 backtesting & execution engine suite across 11 files.
- **Success criteria**: All metrics match quantitative specifications, pass unit and integration tests, verified backtest targets (CAGR >= 35%, MDD <= 20%, Sharpe >= 1.80).
- **Interface contracts**: PROJECT.md, survey analysis specs.
- **Code layout**: `src/backtest/`, `src/execution/`, top-level runners, `tests/test_backtest_engine.py`.

## Change Tracker
- **Files modified**: None yet.
- **Build status**: Initializing.
- **Pending issues**: None.

## Quality Status
- **Build/test result**: Not run yet.
- **Lint status**: 0.
- **Tests added/modified**: TBD.

## Key Decisions Made
- Architecture follows discrete-event bar-by-bar queue simulation.
- Market impact calibrated to Almgren-Chriss standard formulation with configurable volatility and daily volume.
- BVC adapter seamlessly wraps transaction costs, settlement delay queue, and tax tracking.

## Artifact Index
- DISPATCH.md — Assignment log
- progress.md — Real-time execution heartbeat
- handoff.md — Final 5-component handoff report
