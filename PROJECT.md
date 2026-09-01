# Project: Multi-Horizon Intraday Algorithmic Trading & Paper Simulation System

## Architecture
The system is an end-to-end, high-frequency algorithmic trading platform operating on 5–60 minute cycles, integrating 30-year secular macro trends, real-time RSS news NLP sentiment analysis, and microstructure order flow features into a Multi-Horizon Deep Neural Network (Dual-Branch Temporal Convolutional Transformer) driving a discrete-event execution engine with realistic frictions (Almgren-Chriss market impact, bid-ask spreads, commissions, Moroccan BVC settlement & tax rules, and Alpaca Paper Trading API integration).

```
                        [External Data Sources]
     ┌─────────────────────────────┼─────────────────────────────┐
     ▼                             ▼                             ▼
[30Y Daily Data]          [5m-60m Intraday Data]       [Financial RSS Feeds]
(yfinance / Alpaca)        (yfinance / Alpaca)       (Google News / Yahoo)
     │                             │                             │
     ▼                             ▼                             ▼
[Data Ingestion]           [Microstructure Data]      [News Crawler & Dedup]
(Parquet Cache / SQLite)  (Parquet Cache / SQLite)      (SQLite news db)
     │                             │                             │
     └──────────────┬──────────────┘                             ▼
                    ▼                                 [Multi-Tier NLP Engine]
     [Point-in-Time Multi-Timeframe Aligner]          (FinBERT + Financial Lexicon)
                    │                                            │
                    ▼                                            ▼
     [Macro & Micro Features (t-1)]                 [Exp-Decay Sentiment Features]
                    │                                            │
                    └──────────────────────┬─────────────────────┘
                                           ▼
                       [Multi-Horizon Deep Neural Network]
                    (Dual-Branch Dilated TCN + Cross-Attention)
                                           │
                                           ▼
                       [BUY / HOLD / SELL Decisions & Weights]
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
       [Intraday Discrete-Event Backtester]          [Alpaca Paper Trader Bridge]
      - Almgren-Chriss Market Impact                 - Server-side Bracket Orders
      - Dynamic Spread & Commissions                 - Idempotent State Engine
      - Casablanca T+3 & Tax Rules                   - Scheduled Automated Cron
      - Statistical Metric Engine                    - Autonomous Dry-Run Fallback
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | 30-Year Historical Daily Data Fetcher | Ingests 30 years of daily OHLCV data for US & Moroccan equities with caching | M1 | ORIGINAL_REQUEST R1 |
| 2 | High-Resolution Intraday Ingestion | Ingests 5m, 15m, 30m, 60m OHLCV bars across multi-day/year windows | M1 | ORIGINAL_REQUEST R1 |
| 3 | Synthetic Microstructure Generator | High-fidelity jump-diffusion & volume curve generator for BVC/offline testing | M1 | Survey Spec & Analysis |
| 4 | Columnar Parquet & Catalog Database | Fast PyArrow ZSTD storage + SQLite `market_catalog.db` index | M1 | Survey Spec & Analysis |
| 5 | Point-in-Time Multi-Timeframe Aligner | Synchronizes daily macro and intraday bars without lookahead bias | M1 | Survey Spec & Analysis |
| 6 | Multi-Source RSS News Crawler | Polls Google News & Yahoo Finance RSS feeds with ticker search | M2 | ORIGINAL_REQUEST R2 |
| 7 | Article Deduplication & Sanitization | SHA-256 URL hashing and SimHash/token Jaccard deduplication | M2 | Survey Spec & Analysis |
| 8 | Multi-Tier NLP Sentiment Engine | FinBERT Transformer + Loughran-McDonald/VADER financial lexicon | M2 | ORIGINAL_REQUEST R2 |
| 9 | Exponential Time-Decay Aggregator | Computes hourly/daily decaying sentiment time series features | M2 | ORIGINAL_REQUEST R2 |
| 10 | Multi-Horizon DNN Architecture | Dual-branch dilated causal TCN with cross-attention & sentiment GRN | M3 | ORIGINAL_REQUEST R3 |
| 11 | Feature Engineering & Scaling Pipeline | Computes technical, volatility, momentum, and sentiment features with robust scaling | M3 | ORIGINAL_REQUEST R3 |
| 12 | De Prado Triple-Barrier Labeling | Generates path-dependent labels (+1.5σ TP, -1.0σ SL, 12-bar expiry) | M3 | Survey Spec & Analysis |
| 13 | Multi-Horizon Loss & Policy Optimization | Regularized Focal Cross-Entropy + Huber + Differential Sharpe loss | M3 | Survey Spec & Analysis |
| 14 | Real-Time Signal Policy & Decision Engine | Generates actionable BUY/HOLD/SELL signals & allocations for US/Moroccan equities | M3 | ORIGINAL_REQUEST R3 |
| 15 | High-Frequency Intraday Backtester Engine | Discrete-event bar-by-bar engine operating at 5m-60m frequencies | M4 | ORIGINAL_REQUEST R4 |
| 16 | Microstructure Frictions Engine | Models half-spreads, broker fees, and cash balance accounting | M4 | ORIGINAL_REQUEST R4 |
| 17 | Almgren-Chriss Market Impact Model | Simulates non-linear temporary and permanent market impact | M4 | ORIGINAL_REQUEST R4 |
| 18 | Casablanca BVC Microstructure Adapter | Enforces +/-10% circuit breakers, T+3 settlement queues, and 15% IR tax | M4 | Survey Spec & Analysis |
| 19 | Alpaca Paper Trader API Connector | Live REST client placing bracket orders with server-side SL/TP | M4 | ORIGINAL_REQUEST R4 |
| 20 | Automated Cron & Dry-Run Runner | Production runner supporting scheduled execution and zero-credential dry-run | M4 | ORIGINAL_REQUEST R4 |
| 21 | Statistical Risk & Performance Metrics | Computes CAGR, MDD, Sharpe, Sortino, Calmar, DSR, and Win Rate | M4 | ORIGINAL_REQUEST Verification |
| 22 | Out-of-Sample Performance Verification | Validates CAGR >= 35%, MDD <= 20%, Sharpe >= 1.80 on out-of-sample data | M5 | ORIGINAL_REQUEST Acceptance |
| 23 | 100% Pytest Opaque-Box Test Suite (Tiers 1-4) | Comprehensive 4-tier test suite covering all features and boundaries | M5 | ORIGINAL_REQUEST Acceptance |
| 24 | Tier 5 Adversarial Coverage Hardening | White-box stress testing, gap identification, and adversarial robustness checks | M5 | Project Pattern Tier 5 |
| 25 | Live Paper Trading Dry-Run Execution | End-to-end execution of scanner, model inference, and Alpaca order submission | M5 | ORIGINAL_REQUEST Acceptance |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| E2E | E2E Testing Suite Track | Design test infra (`TEST_INFRA.md`), build Tiers 1-4 test suite, publish `TEST_READY.md` | none | IN_PROGRESS |
| M1 | Dual-Resolution Data Ingestion Pipeline | Ingestion (30y daily + 5m-60m intraday), Parquet/SQLite storage, multi-timeframe aligner | none | IN_PROGRESS |
| M2 | RSS News Harvester & Sentiment NLP Engine | RSS crawler, deduplication, FinBERT + Lexicon analyzer, exponential decay feature aggregator | none | IN_PROGRESS |
| M3 | Multi-Horizon Deep Neural Network | Dual-branch TCN/Transformer model, triple-barrier labeling, loss functions, signal policy | M1, M2 | PLANNED |
| M4 | HF Intraday Backtester & Paper Simulator | Discrete-event engine (5m-60m), Almgren-Chriss impact, BVC adapter, Alpaca connector, metrics | M1, M3 | PLANNED |
| M5 | Full System Integration & Final Acceptance | Pass 100% E2E tests (Tiers 1-4), Tier 5 adversarial hardening, verify CAGR >= 35%, MDD <= 20%, Sharpe >= 1.80, dry run | E2E, M1, M2, M3, M4 | PLANNED |

## Code Layout
```
c:\Users\Anwar\Documents\trader\
├── src\
│   ├── data\
│   │   ├── __init__.py
│   │   ├── historical_loader.py    # 30Y Daily & Intraday Ingestion (M1)
│   │   ├── parquet_store.py        # Columnar Parquet & SQLite catalog (M1)
│   │   ├── synthetic_feed.py       # High-fidelity BVC/intraday microstructure generator (M1)
│   │   └── aligner.py              # Zero-lookahead multi-timeframe bar aligner (M1)
│   ├── sentiment\
│   │   ├── __init__.py
│   │   ├── rss_crawler.py          # Google News & Yahoo Finance RSS fetcher (M2)
│   │   ├── deduplicator.py         # SHA-256 & SimHash deduplication (M2)
│   │   ├── nlp_engine.py           # FinBERT + Loughran-McDonald/VADER analyzer (M2)
│   │   └── aggregator.py           # Exponential decay time-series aggregator (M2)
│   ├── models\
│   │   ├── __init__.py
│   │   ├── multi_horizon_dnn.py    # Dual-Branch Dilated Causal TCT Neural Network (M3)
│   │   ├── feature_pipeline.py     # Macro + Micro + Sentiment feature engineering (M3)
│   │   ├── labeler.py              # De Prado Triple-Barrier labeling (M3)
│   │   ├── loss_functions.py       # Regularized Focal + Huber + Sharpe loss (M3)
│   │   └── trainer.py              # Model training, checkpointing & inference policy (M3)
│   ├── backtest\
│   │   ├── __init__.py
│   │   ├── intraday_engine.py      # Discrete-event 5m-60m backtesting engine (M4)
│   │   ├── order_book.py           # Bid-ask spread, slippage & market impact (M4)
│   │   ├── bvc_adapter.py          # Casablanca BVC T+3 settlement & tax rules (M4)
│   │   └── metrics.py              # CAGR, MDD, Sharpe, Sortino, Calmar, DSR (M4)
│   ├── execution\
│   │   ├── __init__.py
│   │   ├── alpaca_client.py        # Alpaca Paper Trader REST client with bracket orders (M4)
│   │   ├── dry_run_simulator.py    # Zero-credential autonomous dry run runner (M4)
│   │   └── scheduler.py            # Cron & scheduled execution loop (M4)
│   └── signals\                    # Additional whale & technical signal helpers
├── tests\
│   ├── test_data_pipeline.py       # Tier 1 & 2 tests for M1
│   ├── test_sentiment_pipeline.py  # Tier 1 & 2 tests for M2
│   ├── test_model_pipeline.py      # Tier 1 & 2 tests for M3
│   ├── test_backtest_engine.py     # Tier 1 & 2 tests for M4
│   ├── test_cross_feature.py       # Tier 3 pairwise integration tests
│   └── test_e2e_scenarios.py       # Tier 4 real-world full workload tests
├── data_cache\
│   ├── parquet\
│   └── market_catalog.db
├── output\
├── run_live_paper.py               # Live execution entry point
├── run_intraday_backtest.py        # Backtesting CLI entry point
├── PROJECT.md                      # Living project architecture & milestone tracker
├── TEST_INFRA.md                   # Test architecture and coverage plan
└── TEST_READY.md                   # E2E Test ready signal file
```

## Interface Contracts

### 1. Data Ingestion ↔ Sentiment & Model (`src/data/` ↔ `src/models/`)
- `HistoricalDataLoader.fetch_dual_resolution(symbol: str, daily_years: int = 30, intraday_days: int = 60, intraday_interval: str = "5m") -> Dict[str, pd.DataFrame]`
  - Returns `{"daily": df_daily, "intraday": df_intraday}`
  - Both dataframes indexed by timezone-aware UTC datetime.
  - Columns: `["open", "high", "low", "close", "volume", "vwap"]`.
- `MultiTimeframeAligner.align_features(df_daily: pd.DataFrame, df_intraday: pd.DataFrame) -> pd.DataFrame`
  - Merges daily macro features lagged by 1 day (`t-1`) into intraday timestamps without lookahead leakage.

### 2. Sentiment Engine ↔ Feature Pipeline (`src/sentiment/` ↔ `src/models/`)
- `SentimentAggregator.get_sentiment_features(symbol: str, target_timestamps: pd.DatetimeIndex, half_life_hours: float = 4.0) -> pd.DataFrame`
  - Returns DataFrame aligned with `target_timestamps` containing:
    `["sentiment_score_ema", "sentiment_confidence_ema", "news_volume_1h", "news_volume_24h", "sentiment_dispersion", "sentiment_momentum", "bull_bear_ratio"]`.
  - Enforces point-in-time publication timestamp constraint (`t_published <= t_decision`).

### 3. Model Pipeline ↔ Intraday Backtester & Paper Simulator (`src/models/` ↔ `src/backtest/` & `src/execution/`)
- `MultiHorizonDNN.predict_signals(features_df: pd.DataFrame) -> pd.DataFrame`
  - Returns DataFrame indexed by timestamp with columns:
    `["action", "buy_prob", "hold_prob", "sell_prob", "target_weight", "expected_return", "confidence"]`
    where `action ∈ {"BUY", "HOLD", "SELL"}` and `target_weight ∈ [0.0, 0.35]`.
- `IntradayBacktestEngine.run(data: Dict[str, pd.DataFrame], signals: Dict[str, pd.DataFrame], initial_capital: float = 100000.0) -> BacktestResult`
  - Returns `BacktestResult` with `.equity_curve`, `.trades`, `.metrics` (dict with `cagr`, `mdd`, `sharpe`, `sortino`, `calmar`, `win_rate`).

### 4. Backtester & Execution ↔ Alpaca Trader (`src/execution/alpaca_client.py`)
- `AlpacaPaperTrader.execute_decision(symbol: str, action: str, target_weight: float, current_price: float, atr: float) -> OrderResult`
  - Places bracket order with take-profit at `+1.5 * ATR` and stop-loss at `-1.0 * ATR` (capped at max 4.8%).
  - Degrades gracefully to dry-run telemetry if API credentials are not provided.
