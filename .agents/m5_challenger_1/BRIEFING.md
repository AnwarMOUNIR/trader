# BRIEFING — 2026-08-31T17:28:45Z

## Mission
Adversarially stress-test the quantitative intraday trading system against extreme volatility, market impact, sentiment inversion, BVC circuit breakers & T+3 settlement, risk management caps, and lookahead leakage.

## 🔒 My Identity
- Archetype: empirical_challenger
- Roles: critic, specialist
- Working directory: c:\Users\Anwar\Documents\trader\.agents\m5_challenger_1
- Original parent: 221d5d39-4a0b-46d8-8d77-134da45be38f
- Milestone: M5
- Instance: 1 of 1

## 🔒 Key Constraints
- Review-only / challenger — empirical verification through test harnesses, do NOT mask bugs, report findings accurately.
- Stress-test extreme volatility regimes (GARCH spike, flash crash, 5-sigma jump events).
- Stress-test high market impact execution (order size > 20% bar volume -> Almgren-Chriss penalty).
- Stress-test sudden news sentiment inversion (+1.0 to -1.0 within 1 bar).
- Stress-test Casablanca BVC limit-down lock days (-10% circuit breaker preventing exit) and cash insolvency checks with T+3 settlement delay.
- Verify risk management caps (single asset <= 34%, gross exposure <= 100%, -4.8% max stop loss) hold unconditionally.
- Check statistical lookahead leakage in features, scaling, or labels.
- Output verdict: APPROVE or REQUEST_CHANGES.
- Write findings to `stress_report.md` and handoff report to `handoff.md`.

## Current Parent
- Conversation ID: 221d5d39-4a0b-46d8-8d77-134da45be38f
- Updated: 2026-08-31T17:28:45Z

## Review Scope
- **Files to review**: `src/` (data, sentiment, models, backtest, execution, signals)
- **Interface contracts**: PROJECT.md
- **Review criteria**: Empirical stress-testing, boundary robustness, lookahead bias, risk cap enforcement

## Attack Surface
- **Hypotheses tested**: TBD
- **Vulnerabilities found**: TBD
- **Untested angles**: TBD

## Loaded Skills
- None

## Key Decisions Made
- Will write modular empirical stress testing harnesses in `tests/stress/` or standalone test execution scripts and run them directly.

## Artifact Index
- `.agents/m5_challenger_1/DISPATCH.md` — Dispatch record
- `.agents/m5_challenger_1/BRIEFING.md` — Persistent memory
- `.agents/m5_challenger_1/progress.md` — Heartbeat
