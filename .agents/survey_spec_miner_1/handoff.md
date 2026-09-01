# Handoff Report: Specification Mining & Formal Requirements

**Agent:** `survey_spec_miner_1` (teamwork_preview_spec_miner)  
**Recipient:** `orchestrator_1` (conversation ID: `221d5d39-4a0b-46d8-8d77-134da45be38f`)  
**Timestamp:** 2026-08-31T15:26:00Z  
**Type:** Hard Handoff (Task Complete)  

---

## 1. Observation

1. **User Request & Baseline Requirements (`.agents/ORIGINAL_REQUEST.md`)**:
   - Lines 5-8: Intraday algorithmic trading system operating on 5–60 minute cycles, integrating real-time RSS news crawling & NLP sentiment, guided by a multi-horizon deep neural network trained on 30 years of daily data + recent intraday data.
   - Lines 12-15 (R1): Dual-resolution data ingestion pipeline (30y daily US equities via yfinance/free APIs + high-res 5m–60m intraday data for last 60–730 days).
   - Lines 16-19 (R2): RSS news harvester (Yahoo Finance, Google News) + local NLP sentiment analyzer (VADER / FinBERT / HuggingFace) outputting numerical sentiment scores.
   - Lines 20-26 (R3): Multi-horizon deep neural network (LSTM / GRU / TCN) utilizing macro features, intraday microstructure features, and real-time sentiment features to output BUY/HOLD/SELL decisions for a dual US/Moroccan universe.
   - Lines 27-31 (R4): Discrete-event intraday backtester (5m–60m) with trading frictions (bid-ask spread, broker commissions, Almgren-Chriss market impact) and Alpaca Paper Trader connector.
   - Lines 40-46 (Acceptance Criteria): Net annualized return (CAGR) $\ge 35\%$, Maximum Drawdown (MDD) $\le 20\%$, Sharpe Ratio $\ge 1.80$, pytest suite passes 100%, and live paper trading dry-run cycle executes successfully.

2. **Existing Codebase State**:
   - `src/data/loader.py`: Ingests US and BVC daily market data via `yfinance` with fallback deterministic statistical simulation generator.
   - `src/signals/whale_detector.py`: Computes Volume Z-Score, Wyckoff VSA (Absorption, Spring, Upthrust), Order Flow Imbalance (OFI), and Isolation Forest ML anomaly scores.
   - `src/portfolio/allocator.py`: Implements inverse-variance risk parity with volatility scaling, single asset cap ($\le 34\%$), and momentum crash protection.
   - `src/execution/simulator.py`: Implements Almgren-Chriss linear temporary and permanent impact, half bid-ask spread, broker commission, Moroccan 15% IR tax withholding, and $T+3$ settlement queues.
   - `src/backtest/engine.py` & `src/backtest/metrics.py`: Simulates portfolio accounting, trailing stops, performance metric calculations (CAGR, MDD, Sharpe, Sortino, Calmar, Win Rate).
   - `src/execution/alpaca_trader.py`: Authenticates with Alpaca Paper Trading REST API, fetches bars, calculates targets, places bracket orders with server-side stop-loss, and operates in autonomous dry-run mode when unauthenticated.
   - `src/validation/monte_carlo.py`, `src/validation/cpcv.py`, `src/validation/deflated_sharpe.py`: Statistical validation engines.
   - `tests/test_system.py`: 7 unit/integration tests covering loader, signals, allocator, execution, metrics, Monte Carlo, CPCV, and Deflated Sharpe.
   - Executed `pytest tests/test_system.py`: Verified 7/7 tests pass in 4.51s (Exit code 0).

---

## 2. Logic Chain

1. **From Requirements to Mathematical Specification**:
   - The Almgren-Chriss framework requires balancing temporary impact $\eta(v)$ (liquidity penalty on immediate fills) against permanent impact $\gamma(v)$ (information leakage shifting equilibrium mid-price). For discrete 5m–60m simulation, this translates directly to participation rate $\rho_t = |\Delta q_t| / V_{\text{bar}, t}$ and effective fill price $P_{\text{exec}} = P_{\text{mid}} (1 \pm (\epsilon + \eta \rho + \gamma \rho + C_{\text{comm}}))$.
   - The acceptance criteria require $\text{CAGR} \ge 35\%$, $\text{MDD} \le 20\%$, and $\text{Sharpe} \ge 1.80$. The annualization factor $M$ must accurately correspond to the simulation bar frequency ($M = 19,656$ for 5m, $M = 1,638$ for 60m, $M = 252$ for daily).
   - The Multi-Horizon DNN requires simultaneously predicting multi-step forward returns ($h \in \{3, 12, 78\}$ bars). Using volatility-adaptive triple barriers $\pm \theta_t = \pm \beta \cdot \text{ATR}_t / P_t$, returns map into discrete classes (BUY, HOLD, SELL). To address severe class imbalance and optimize risk-adjusted returns, the objective function combines Focal Cross-Entropy loss, Multi-Horizon Huber loss, and Differential Sharpe policy reward loss.

2. **From API Contracts to Architecture**:
   - The dual-resolution pipeline integrates daily 30-year feeds with intraday (5m–60m) feeds. Because the universe spans US and Casablanca markets, timestamps must be strictly normalized to UTC, with explicit handling of US Daylight Saving Time and Moroccan Ramadan time changes.
   - The RSS feed engine ingests Yahoo Finance and Google News XML feeds, parses text, and evaluates sentiment via local zero-credential NLP models (VADER / FinBERT). The resulting article scores are continuously aggregated into hourly/daily features using exponential time decay ($t_{1/2} = 6 \text{ hours}$).
   - The live connector interfaces with Alpaca Paper Trading REST endpoints (`/v2/orders`, `/v2/account`) using bracket orders with -4.8% hard stop losses. If API keys are absent, the connector gracefully degrades to an autonomous dry-run engine that generates and logs full simulated execution telemetry without crashing.

---

## 3. Caveats

1. **Intraday Data Availability Limits on Free Tier**:
   - `yfinance` limits 5-minute intraday historical data to the last 60 calendar days and 60-minute data to the last 730 days. For backtesting longer historical windows at 5m resolution, data must either be continuously accumulated into local Parquet caches or supplemented with historical archive datasets / Alpaca Market Data feeds.
2. **Moroccan Casablanca Market Microstructure**:
   - BVC public feeds provide 15-minute delayed quotes and daily bulletins. Real-time sub-second order book data requires institutional FIX gateways (e.g. ICE / local brokers). In development and backtesting, the high-fidelity generative engine enforces Casablanca microstructure constraints ($\pm 10\%$ daily limit clamps, $T+3$ cash settlement, 15% IR tax).
3. **NLP Latency & Hardware Constraints**:
   - FinBERT transformer models require GPU/CPU inference time. For fast real-time 5m polling cycles, lightweight caching and headline-level sentiment parsing are recommended to keep execution latency under 500ms per bar.

---

## 4. Conclusion

The formal specification mining is complete. All mathematical models (Almgren-Chriss, CAGR/MDD/Sharpe formulas, multi-horizon focal/Sharpe loss), API and data contracts (dual-resolution ingestion, RSS/NLP schema, Alpaca Paper Trader connector), 28 discrete features, 12 edge cases, failure recovery protocols, and verification acceptance criteria have been comprehensively documented in `c:\Users\Anwar\Documents\trader\.agents\survey_spec_miner_1\analysis.md`.

The project is fully specified and ready for the next engineering and implementation phases.

---

## 5. Verification Method

1. **Review Detailed Specification**:
   - Inspect `c:\Users\Anwar\Documents\trader\.agents\survey_spec_miner_1\analysis.md` for mathematical formulations, data contracts, feature matrix, and edge cases.
2. **Run System Test Suite**:
   ```powershell
   pytest tests/test_system.py -v
   ```
   *Expected Output:* 7 passed in $< 5$ seconds with exit code 0.
3. **Execute Live Scanner & Dry-Run Verification**:
   ```powershell
   python scan_current_market.py
   python run_live_paper.py
   ```
   *Expected Output:* Outputs live alpha scores, target portfolio allocations, and simulated order logs to `output/current_signals_snapshot.json`.
