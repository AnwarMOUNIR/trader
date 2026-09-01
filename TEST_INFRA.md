# Test Infrastructure & Verification Architecture (TEST_INFRA.md)

## 1. Testing Philosophy & Guiding Principles

The testing framework for the **Multi-Horizon Intraday Algorithmic Trading & Paper Simulation System** is built upon rigorous institutional quantitative testing standards:

1. **Opaque-Box Requirement-Driven Verification**:
   - Tests validate behavior, invariant contracts, and quantitative outcomes against the specifications defined in `ORIGINAL_REQUEST.md` and `PROJECT.md`, without coupling to private internal implementation details.
   - Every test case answers: *"If this test fails, what specific financial, statistical, or data pipeline capability is broken?"*
2. **Deterministic & Authoritative Expected Outputs**:
   - Expected values are derived from mathematical closed forms (e.g. Almgren-Chriss impact equations, CAGR formula, Sharpe ratio definitions, Bailey-López de Prado DSR), known historical distributions, or reference statistical oracles.
3. **Progressive Testability & Graceful Degradation**:
   - Test suites adapt dynamically to components available across implementation milestones (M1 through M5), using high-fidelity deterministic fixtures and mocks when external networks or live market APIs are unavailable.
4. **Self-Contained Isolation**:
   - Each test sets up its own state, avoids global state mutations, runs independently in any execution order, and cleans up temporary test artifacts.
5. **Adversarial & Boundary Stress Hardening**:
   - Extreme market dislocations, zero volume illiquidity, price limit clamps, network disconnects, time-zone shifts, and corrupted feeds are tested explicitly.

---

## 2. 25-Feature Inventory Test Matrix

The matrix below maps each of the 25 system features from `PROJECT.md` to its verification tier, target test file, and test method:

| # | Feature Name | Description | Target Test File | Verification Tier | Key Assertion & Success Criteria |
|---|---|---|---|---|---|
| **F01** | 30-Year Historical Daily Data Fetcher | Ingests 30 years of daily OHLCV for US & BVC with caching | `tests/test_data_pipeline.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Timezone-aware UTC index, no NaNs, cached parquet/CSV fallback |
| **F02** | High-Resolution Intraday Ingestion | Ingests 5m, 15m, 30m, 60m OHLCV bars across multi-day windows | `tests/test_data_pipeline.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Multi-frequency bar structure, monotonic timestamps |
| **F03** | Synthetic Microstructure Generator | High-fidelity jump-diffusion & volume curve generator | `tests/test_data_pipeline.py`, `tests/test_cross_feature.py` | Tier 1, 3 | U-shape intraday volume, Student-t fat tails, BVC +/-10% clamp |
| **F04** | Columnar Parquet & Catalog DB | Fast PyArrow ZSTD storage + SQLite catalog index | `tests/test_data_pipeline.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Parquet read/write roundtrip fidelity, SQLite schema integrity |
| **F05** | Point-in-Time Multi-Timeframe Aligner | Synchronizes daily macro and intraday bars without lookahead | `tests/test_data_pipeline.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Strict $t-1$ daily lagging, zero lookahead leakage into $t_{\text{intra}}$ |
| **F06** | Multi-Source RSS News Crawler | Polls Google News & Yahoo Finance RSS feeds with ticker search | `tests/test_sentiment_pipeline.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Network timeout resilience, structured `NewsArticle` output |
| **F07** | Article Deduplication & Sanitization | SHA-256 URL hashing and SimHash/token Jaccard deduplication | `tests/test_sentiment_pipeline.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Identical URLs and duplicate syndicated stories filtered |
| **F08** | Multi-Tier NLP Sentiment Engine | FinBERT Transformer + Loughran-McDonald/VADER lexicon | `tests/test_sentiment_pipeline.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Correct polarity alignment (bullish $> 0$, bearish $< 0$), fallback |
| **F09** | Exponential Time-Decay Aggregator | Computes hourly/daily decaying sentiment time series features | `tests/test_sentiment_pipeline.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Half-life decay math verified, neutral baseline on empty feed |
| **F10** | Multi-Horizon DNN Architecture | Dual-branch dilated causal TCN with cross-attention & GRN | `tests/test_model_pipeline.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Valid output probability simplex ($\sum p_c = 1.0$), weight $\in [0, 0.35]$ |
| **F11** | Feature Engineering & Scaling Pipeline | Computes technical, volatility, momentum, & sentiment features | `tests/test_model_pipeline.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Finite feature matrix, robust scaling without division-by-zero |
| **F12** | De Prado Triple-Barrier Labeling | Generates path-dependent labels (+1.5σ TP, -1.0σ SL, 12-bar expiry)| `tests/test_model_pipeline.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Correct class assignment $\{0: \text{BUY}, 1: \text{HOLD}, 2: \text{SELL}\}$ |
| **F13** | Multi-Horizon Loss & Policy Optimization| Regularized Focal + Huber + Differential Sharpe loss | `tests/test_model_pipeline.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Non-negative loss, finite gradients, correct policy penalty |
| **F14** | Real-Time Signal Policy & Decision Engine| Actionable BUY/HOLD/SELL signals & allocations for US/BVC | `tests/test_model_pipeline.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Allocation caps obeyed, single position $\le 34\%$, total $\le 100\%$ |
| **F15** | High-Frequency Intraday Backtester Engine| Discrete-event bar-by-bar engine operating at 5m-60m frequencies| `tests/test_backtest_engine.py`, `tests/test_e2e_scenarios.py` | Tier 1, 4 | Monotonic timestamp loop, trade log accounting, cash conservation |
| **F16** | Microstructure Frictions Engine | Models half-spreads, broker fees, and cash balance accounting | `tests/test_backtest_engine.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Slippage and commission deduction on every fill |
| **F17** | Almgren-Chriss Market Impact Model | Simulates non-linear temporary and permanent market impact | `tests/test_backtest_engine.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Impact increases monotonically with participation rate $\rho$ |
| **F18** | Casablanca BVC Microstructure Adapter | Enforces +/-10% circuit breakers, T+3 settlement, 15% IR tax | `tests/test_backtest_engine.py`, `tests/test_cross_feature.py` | Tier 1, 3 | 15% IR withheld on net gains, cash unlocked on day $t+3$ |
| **F19** | Alpaca Paper Trader API Connector | Live REST client placing bracket orders with server-side SL/TP | `tests/test_system.py`, `tests/test_cross_feature.py` | Tier 1, 3 | Bracket order format, -4.8% SL cap, dry-run mode fallback |
| **F20** | Automated Cron & Dry-Run Runner | Production runner supporting scheduled cron and zero-credential dry-run | `tests/test_system.py`, `tests/test_e2e_scenarios.py` | Tier 1, 4 | Autonomous run without API keys, writes valid JSON telemetry |
| **F21** | Statistical Risk & Performance Metrics | Computes CAGR, MDD, Sharpe, Sortino, Calmar, DSR, Win Rate | `tests/test_system.py`, `tests/test_e2e_scenarios.py` | Tier 1, 4 | Mathematical accuracy against analytical benchmark curves |
| **F22** | Out-of-Sample Performance Verification | Validates CAGR >= 35%, MDD <= 20%, Sharpe >= 1.80 on OOS data | `tests/test_e2e_scenarios.py` | Tier 4 | Quantitative thresholds verified on realistic multi-asset series |
| **F23** | 100% Pytest Opaque-Box Test Suite | 4-tier comprehensive test suite covering all modules & contracts | `tests/` (All files) | Tier 1-4 | 100% test pass rate across all suite suites |
| **F24** | Tier 5 Adversarial Coverage Hardening | Stress testing, gap identification, and adversarial robustness | `tests/test_cross_feature.py`, `tests/test_e2e_scenarios.py` | Tier 2, 3, 4 | Zero volume, gap downs, network timeouts handled cleanly |
| **F25** | Live Paper Trading Dry-Run Execution | End-to-end scanner, model inference, and Alpaca order submission | `tests/test_e2e_scenarios.py` | Tier 4 | Complete cycle logs valid actions and portfolio snapshot |

---

## 3. Test Architecture & Tier Stratification

```
+---------------------------------------------------------------------------------------------------+
|                                 TEST ARCHITECTURE & STRATIFICATION                                |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [TIER 1: Feature Unit Coverage] (Features 1 - 25 in isolation)                                   |
|   - Data loaders, Parquet storage, RSS crawlers, FinBERT/Lexicon sentiment, DNN layers, metrics   |
|                                                                                                   |
|  [TIER 2: Boundary & Corner Cases] (Adversarial stress and edge handling)                         |
|   - Zero volume, market halts, +/-10% circuit breakers, DST shifts, T+3 lockouts, empty feeds    |
|                                                                                                   |
|  [TIER 3: Cross-Feature Combinations] (Pairwise & multi-module integration)                       |
|   - Loader -> Aligner -> Storage                                                                  |
|   - Crawler -> Dedup -> NLP Scorer -> Exponential Decay Aggregator                                |
|   - Sentiment Spike -> Feature Pipeline -> DNN Decision Engine                                    |
|   - Signal Policy -> HF Backtester -> Almgren-Chriss Impact -> BVC T+3 Adapter                    |
|   - Model Inference -> Alpaca Bracket Order Submission -> Dry-Run Telemetry                       |
|                                                                                                   |
|  [TIER 4: Real-World Workload Scenarios] (End-to-End full system workflows)                       |
|   - Dual-Universe (US: AAPL/NVDA/TSLA/SPY + BVC: ATW.CS/IAM.CS/BCP.CS)                            |
|   - Multi-Timeframe Intraday Simulation (5m, 15m, 60m cycles)                                     |
|   - Full quantitative verification: CAGR >= 35%, MDD <= 20%, Sharpe >= 1.80                       |
|   - Live Paper Dry-Run full cycle execution                                                       |
+---------------------------------------------------------------------------------------------------+
```

---

## 4. Edge Case & Adversarial Scenario Catalog

| ID | Category | Scenario Description | Expected System Behavior |
|---|---|---|---|
| **E01** | Data Pipeline | Zero volume or prolonged market halt (missing intraday bars) | Forward-fills up to 3 bars; marks volume 0, prevents phantom order fills |
| **E02** | Timezone / Calendar | US Daylight Saving Time (EDT $\leftrightarrow$ EST) & Moroccan Ramadan session | UTC normalization prevents duplicated or skipped decision bars |
| **E03** | RSS News Pipeline | Network timeout, HTTP 429 rate limit, or invalid RSS XML | Backs off with retry, falls back to cached news without throwing exceptions |
| **E04** | NLP Sentiment | Empty news feed (zero articles published in past 72 hours) | Exponential decay reduces sentiment smoothly to 0.0 (neutral baseline) |
| **E05** | Model Decision | Extreme volatility producing low prediction confidence ($< 0.45$) | Decision engine falls back to default `HOLD` (cash preservation defense) |
| **E06** | Market Impact | Order size exceeds 25% of bar volume in illiquid stock | Clamps participation rate at $\rho_{\max} = 0.25$ and slices remainder |
| **E07** | Casablanca BVC | Stock locked at Limit Up (+10%) or Limit Down (-10%) | Order execution engine halts buys/sells at limit to reflect real market illiquidity |
| **E08** | Portfolio Sizing | Simultaneous breakout signals across all universe assets | Risk-parity weights normalize to $\sum w_i \le 1.0$; max single weight $\le 34\%$ |
| **E09** | Execution Bridge | Missing Alpaca API keys or network disconnect | Gracefully enters Autonomous Dry-Run mode; logs simulated orders to JSON |
| **E10** | Stop-Loss Protection | Overnight gap-down opening $> 4.8\%$ below previous close | Stop order fills at open price with slippage, preventing unrealized loss drift |
| **E11** | Settlement Queue | Rapid consecutive BVC sales with zero initial cash | Unsettled cash remains in $T+3$ queue; prevents overdraft on subsequent buys |
| **E12** | Statistical Metrics | Zero variance or constant equity series | Returns 0.0 for Sharpe/Sortino/Calmar without ZeroDivisionError |

---

## 5. Interface Contracts & Verification Matrix

```python
# 1. Dual-Resolution Data Ingestor Contract
HistoricalDataLoader.fetch_dual_resolution(
    symbol: str, daily_years: int = 30, intraday_days: int = 60, intraday_interval: str = "5m"
) -> Dict[str, pd.DataFrame]  # {"daily": df_daily, "intraday": df_intraday}

# 2. Multi-Timeframe Aligner Contract
MultiTimeframeAligner.align_features(
    df_daily: pd.DataFrame, df_intraday: pd.DataFrame
) -> pd.DataFrame  # daily features lagged by 1 day (t-1) merged into intraday timestamps

# 3. Sentiment Aggregator Contract
SentimentAggregator.get_sentiment_features(
    symbol: str, target_timestamps: pd.DatetimeIndex, half_life_hours: float = 4.0
) -> pd.DataFrame  # ["sentiment_score_ema", "sentiment_confidence_ema", "news_volume_1h", ...]

# 4. Multi-Horizon Model Contract
MultiHorizonDNN.predict_signals(
    features_df: pd.DataFrame
) -> pd.DataFrame  # ["action", "buy_prob", "hold_prob", "sell_prob", "target_weight", "expected_return"]

# 5. Discrete-Event Backtester Contract
IntradayBacktestEngine.run(
    data: Dict[str, pd.DataFrame], signals: Dict[str, pd.DataFrame], initial_capital: float = 100000.0
) -> BacktestResult  # .equity_series, .cash_series, .trade_log, .metrics

# 6. Alpaca Paper Trader Contract
AlpacaPaperTrader.run_daily_cycle() -> Dict  # status, equity, cash, target_allocation, actions
```

---

## 6. Coverage Thresholds & Quality Targets

- **Test Suite Pass Rate**: **100%** (zero failed, zero errored tests).
- **Code Coverage Target**: $\ge 90\%$ line and branch coverage across core trading modules.
- **Financial Metric Verification**:
  - Out-of-sample CAGR $\ge 35.0\%$
  - Maximum Drawdown $\le 20.0\%$
  - Annualized Sharpe Ratio $\ge 1.80$
  - Deflated Sharpe Ratio (DSR) $\ge 0.90$
- **Execution Safety**: 100% compliance with -4.8% hard stop loss, BVC 15% IR tax withholding, and $T+3$ clearing delay.
