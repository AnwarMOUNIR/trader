# Technical Survey & Architecture Design: Dual-Resolution Ingestion (R1) & Sentiment Pipeline (R2)

**Author:** `survey_explorer_1`  
**Target Milestone:** Milestone 1 (R1 & R2 Architecture & Technical Survey)  
**Date:** 2026-08-31  
**Project Workspace:** `c:\Users\Anwar\Documents\trader`  

---

## 1. Executive Summary & Problem Formulation

The objective of this quantitative system is to build an intraday algorithmic trading platform operating on 5-minute to 60-minute decision cycles, augmented by real-time financial news sentiment extraction and a multi-horizon deep neural network (DNN) trained across 30 years of macro daily data and multi-timeframe intraday microstructure data. The trading universe spans both **US Equities (NYSE/NASDAQ)** and **Moroccan Equities (Casablanca Stock Exchange / BVC)**.

This survey establishes the complete architectural blueprints for:
1. **R1: Dual-Resolution Data Ingestion Pipeline**: Ingestion, normalization, caching, and point-in-time synchronization of 30-year historical daily bars and multi-frequency intraday bars (5m, 15m, 30m, 60m) utilizing Apache Parquet columnar storage and SQLite metadata catalogs.
2. **R2: RSS News Harvester & Sentiment Analyzer**: Live and historical polling of multi-source financial RSS feeds (Google News, Yahoo Finance, Moroccan business news), content deduplication, multi-tier NLP sentiment scoring (FinBERT Transformer + Financial Lexicon Fallback), and continuous exponential time-decay aggregation into structured intraday feature tensors.

---

## 2. Existing Codebase & Environment Audit

### 2.1 Runtime Environment & Verified Dependencies

The system was inspected on the active host environment with the following verified component matrix:

| Package / Tool | Installed Version | Status / Capability | Purpose in Architecture |
|---|---|---|---|
| **Python Interpreter** | `3.14.6` (`AMD64`) | Active (`Python314\python.exe`) | Core runtime |
| **PyTorch (`torch`)** | `2.10.0` | Verified (`torch.nn.LSTM`, CPU tensor ops verified) | Multi-Horizon DNN (R3) & FinBERT inference |
| **Transformers (`transformers`)** | `4.57.1` | Verified (HuggingFace AutoModel / Pipeline) | FinBERT / DistilBERT sentiment extraction |
| **Tokenizers (`tokenizers`)** | `0.22.2` | Verified (Fast Rust-backed tokenization) | NLP batch tokenization |
| **`yfinance`** | `1.6.0` | Verified (30Y daily + 5m intraday confirmed) | Historical US daily & intraday ingestion |
| **`alpaca-py`** | `0.44.0` | Verified (TradingClient, StockHistoricalDataClient) | Live Alpaca paper trading & intraday data |
| **`pyarrow`** | `24.0.0` | Verified (Parquet read/write confirmed) | High-speed columnar storage engine |
| **`pandas`** | `3.0.3` | Verified | Time-series data manipulation |
| **`numpy`** | `2.3.5` | Verified | Numerical linear algebra & vectorized features |
| **`scikit-learn`** | `1.9.0` | Verified (IsolationForest, preprocessing) | Machine learning anomaly detection & scaling |
| **`scipy`** | `1.18.0` | Verified | Statistical hypothesis testing & optimization |
| **`beautifulsoup4` / `lxml`**| `4.15.0` / `6.1.1` | Verified | XML/HTML RSS feed scraping & parsing |
| **`sqlite3`** | Built-in (Python 3.14) | Verified (ACID transactions, memory/disk indexing) | News & market metadata indexing |
| **`pytest`** | `9.1.1` | Verified (7/7 tests passing in 4.74s) | Automated test harness |
| **`uv`** | `0.11.32` | Verified | Package manager |

### 2.2 Existing Codebase Architecture Analysis

The existing repository implements a daily-resolution swing trading system with the following modules:

1. `src/data/loader.py`:
   - Contains `UniverseConfig`, `AssetMarket` (US, BVC), and `MarketDataLoader`.
   - Ingestion: Queries `yfinance` for daily US data (2020-2026) and provides a synthetic geometric Brownian motion generator with Wyckoff whale clusters for BVC equities.
   - Storage: Flat CSV files in `data_cache/`.
   - **Limitations for New Architecture**: CSV-only storage lacks compression and column pruning; daily-only resolution (no intraday 5m-60m ingestion); does not ingest 30-year historical daily depth.

2. `src/signals/whale_detector.py` & `src/signals/insider_scraper.py`:
   - Daily volume Z-scores, RVOL, Wyckoff VSA (absorption, spring, upthrust), OFI normalized by market cap, IsolationForest anomaly detection.
   - Scrapes SEC Form 4 insider transactions via `yfinance`.
   - **Limitations for New Architecture**: Daily-only feature computation; lacks intraday microstructure features and news sentiment feature inputs.

3. `src/portfolio/allocator.py`:
   - Inverse-variance risk parity with conviction multipliers, Daniel & Moskowitz momentum crash protection, and position caps (34% max single asset).

4. `src/execution/simulator.py` & `src/execution/alpaca_trader.py`:
   - `ExecutionModel`: Almgren-Chriss market impact (linear temporary $\eta$ and permanent $\gamma$), half-spread, brokerage commissions, Moroccan 15% IR tax withholding, and T+3 settlement queue.
   - `AlpacaPaperTrader`: Interacts with Alpaca paper trading API for live order execution.

5. `src/backtest/engine.py` & `src/backtest/metrics.py`:
   - Daily step simulation loop. Computes CAGR, Sharpe, Sortino, Calmar, MDD, Win Rate, Profit Factor.

6. `src/validation/`:
   - Block-bootstrap Monte Carlo (`monte_carlo.py`), Combinatorial Purged Cross-Validation (`cpcv.py`), Deflated Sharpe Ratio (`deflated_sharpe.py`).

---

## 3. Architecture of R1: Dual-Resolution Data Ingestion Pipeline

```
+-----------------------------------------------------------------------------------------------+
|                             R1: DUAL-RESOLUTION DATA INGESTION ENGINE                         |
+-----------------------------------------------------------------------------------------------+
|                                                                                               |
|   +--------------------------+  +--------------------------+  +---------------------------+   |
|   |  30-Year Historical Daily|  |   High-Res Intraday      |  |  Microstructure Synthetic |   |
|   |  - yfinance (1995-Pres.) |  |   - yfinance (5m-60m)    |  |  - Intraday U-shape Vol   |   |
|   |  - Alpaca Daily Bars     |  |   - Alpaca Minute Bars   |  |  - BVC 10% Limits Clamping|   |
|   +------------+-------------+  +------------+-------------+  +-------------+-------------+   |
|                |                             |                              |                 |
|                +--------------------+        |        +---------------------+                 |
|                                     v        v        v                                       |
|                         +---------------------------------------+                             |
|                         |    DualUniverse Ingestion Manager     |                             |
|                         |    - US Universe (20 Large-Caps + SPY)|                             |
|                         |    - BVC Universe (12 MASI Leaders)   |                             |
|                         |    - Currency Normalization (USD/MAD) |                             |
|                         +-------------------+-------------------+                             |
|                                             |                                                 |
|                                             v                                                 |
|                         +---------------------------------------+                             |
|                         |       Storage & Indexing Subsystem    |                             |
|                         |  - Apache Parquet (ZSTD Columnar)     |                             |
|                         |  - SQLite Metadata Catalog            |                             |
|                         +-------------------+-------------------+                             |
|                                             |                                                 |
|                                             v                                                 |
|                         +---------------------------------------+                             |
|                         |  MultiTimeframe Point-in-Time Aligner |                             |
|                         |  - Lagged Daily Macro -> Intraday 5m  |                             |
|                         |  - No-Lookahead Window Synchronization|                             |
|                         |  - Resampling (5m -> 15m/30m/60m)     |                             |
|                         +---------------------------------------+                             |
+-----------------------------------------------------------------------------------------------+
```

### 3.1 Dual-Resolution Specifications

1. **Resolution 1 (Macro Daily Timeframe - 30 Years)**:
   - **Range**: `1995-01-01` to Present (~7,800 trading days for mature US assets; max available for newer IPOs/BVC).
   - **Fields**: `[Timestamp (Date), Open, High, Low, Close, Volume, Dividend, Splits, MarketCap, Currency, Market]`.
   - **Use Case**: Macro regime detection (200 SMA, 50 SMA), multi-decade volatility regimes, Daniel-Moskowitz momentum score, long-term feature embeddings for R3 neural network pre-training.

2. **Resolution 2 (Micro Intraday Timeframes - 5m, 15m, 30m, 60m)**:
   - **Range**: Maximum historical depth available via public APIs:
     - 5-Minute (5m): Last 60 days via `yfinance` / historical range via Alpaca / multi-year synthetic replay.
     - 15-Minute (15m): Resampled from 5m or direct query.
     - 30-Minute (30m): Resampled from 5m or direct query.
     - 60-Minute / 1-Hour (60m): Last 730 days via `yfinance` / historical range via Alpaca.
   - **Fields**: `[Timestamp (UTC), Open, High, Low, Close, Volume, VWAP, TradeCount, BidAskSpreadEstimate, OFI_Proxy, Market, Currency]`.
   - **Use Case**: Intraday execution signals, order flow pressure, intraday breakout detection, real-time news sentiment fusion.

### 3.2 Dual-Universe Universe Definition

- **US Equities Universe (20 Growth/Momentum Mega-Caps + Benchmark)**:
  `NVDA`, `AAPL`, `MSFT`, `AMZN`, `META`, `GOOGL`, `TSLA`, `AMD`, `AVGO`, `PLTR`, `SMCI`, `NFLX`, `CRM`, `INTC`, `PANW`, `CRWD`, `NOW`, `UBER`, `ABNB`, `COIN`, and Benchmark `SPY`.
- **Moroccan Equities Universe (12 Liquid Casablanca Stock Exchange Blue-Chips)**:
  `ATW.BVC` (Attijariwafa Bank), `BCP.BVC` (Banque Centrale Populaire), `IAM.BVC` (Maroc Telecom), `BOA.BVC` (Bank of Africa), `CSR.BVC` (Cosumar), `LBV.BVC` (Label Vie), `HPS.BVC` (Hightech Payment Systems), `TQM.BVC` (TotalEnergies Maroc), `TGCC.BVC` (TGCC), `MSA.BVC` (Marsa Maroc), `CIM.BVC` (Ciments du Maroc), `ADH.BVC` (Addoha).
- **Cross-Market Session & Currency Specifications**:
  - Currency: USD (US) vs MAD (BVC). Default FX rate parameter: `mad_usd_rate = 0.10` (1 MAD = 0.10 USD).
  - Trading Hours:
    - US Regular Trading Hours (RTH): 09:30 - 16:00 EST / EDT (13:30 - 20:00 UTC).
    - Casablanca Trading Hours: 09:30 - 15:30 UTC+1 (08:30 - 14:30 UTC).

### 3.3 High-Fidelity Synthetic Microstructure Generator

To ensure continuous, robust backtesting over multi-year intraday horizons and realistic Moroccan market dynamics:
- **Intraday Volume Distribution**: Models the empirical U-shaped intraday volume curve:
  $$V_{t, intra} = V_{daily} \times \left( a(t - t_{mid})^2 + b \right)$$
- **Fat-Tailed Price Innovations**: Student-$t$ distribution ($\nu = 4.5$) with intraday volatility clustering (GARCH(1,1) proxy).
- **Institutional Order Flow Injection**: Wyckoff accumulation / distribution blocks (multi-bar volume surges of 3x-6x average with directional price absorption).
- **Casablanca Circuit Breaker Clamping**: Enforces BVC statutory dynamic price threshold of $\pm 10\%$ daily and $\pm 3\%$ per continuous auction tick.

### 3.4 Columnar Parquet & SQLite Storage Architecture

To replace flat CSV files with an institutional-grade storage backend:

```
data_cache/
├── parquet/
│   ├── daily/
│   │   ├── AAPL.parquet
│   │   ├── NVDA.parquet
│   │   └── ATW_BVC.parquet
│   ├── intraday_5m/
│   │   ├── AAPL.parquet
│   │   └── ...
│   ├── intraday_15m/
│   ├── intraday_30m/
│   └── intraday_60m/
├── market_catalog.db          <-- SQLite metadata catalog
└── news_sentiment.db          <-- SQLite news & sentiment database
```

#### Parquet Table Schema
```
Column Name         Arrow Type              Description
timestamp           timestamp[ns, UTC]      Standardized UTC bar close/start
open                float64                 Bar open price
high                float64                 Bar high price
low                 float64                 Bar low price
close               float64                 Bar close price
volume              float64                 Total shares traded
vwap                float64                 Volume Weighted Average Price
trade_count         int32                   Estimated / actual trade count
market              string                  'US' or 'BVC'
currency            string                  'USD' or 'MAD'
```

#### SQLite Catalog Schema (`market_catalog.db`)
```sql
CREATE TABLE IF NOT EXISTS asset_registry (
    symbol TEXT PRIMARY KEY,
    name TEXT,
    market TEXT NOT NULL,
    currency TEXT NOT NULL,
    sector TEXT,
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS ingestion_manifest (
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,          -- 'daily', '5m', '15m', '30m', '60m'
    first_timestamp TEXT NOT NULL,
    last_timestamp TEXT NOT NULL,
    total_bars INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    file_size_bytes INTEGER NOT NULL,
    checksum_sha256 TEXT NOT NULL,
    last_updated_utc TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, timeframe)
);
```

### 3.5 Point-in-Time Multi-Timeframe Alignment & No-Lookahead Guarantees

To ensure that 30-year macro daily features and high-resolution intraday bars are merged without future information leakage:
1. **Strict Lagging Rule**:
   - For an intraday bar occurring at datetime $t = \text{2026-08-31 10:15:00 UTC}$, the daily macro features (e.g. 200 SMA, 52-Week High, Volatility regime) available to the model are computed using daily bars **strictly ending on or before $T - 1$** (i.e. $\le \text{2026-08-28 20:00:00 UTC}$).
   - This eliminates lookahead bias where today's closing price leaks into today's opening/intraday decision.
2. **Resampling Engine**:
   - Vectorized OHLCV resampling function `resample_ohlcv(df_5m, target_freq="15m"|"30m"|"60m")`:
     - $\text{Open} = \text{first}(Open)$
     - $\text{High} = \max(High)$
     - $\text{Low} = \min(Low)$
     - $\text{Close} = \text{last}(Close)$
     - $\text{Volume} = \sum(Volume)$
     - $\text{VWAP} = \frac{\sum (Close \times Volume)}{\sum Volume}$

---

## 4. Architecture of R2: RSS News Harvester & Sentiment Analyzer

```
+-----------------------------------------------------------------------------------------------+
|                         R2: RSS NEWS HARVESTER & NLP SENTIMENT PIPELINE                       |
+-----------------------------------------------------------------------------------------------+
|                                                                                               |
|   +--------------------------+  +--------------------------+  +---------------------------+   |
|   |    Google News RSS API   |  |   Yahoo Finance RSS & API|  | Moroccan Financial News   |   |
|   |  - Query by Ticker/Name  |  |   - yf.Ticker.news feed  |  | - Casablanca RSS feeds    |   |
|   +------------+-------------+  +------------+-------------+  +-------------+-------------+   |
|                |                             |                              |                 |
|                +--------------------+        |        +---------------------+                 |
|                                     v        v        v                                       |
|                         +---------------------------------------+                             |
|                         |       News Feed Deduplication Engine  |                             |
|                         |  - SHA-256 URL Hashing & Canonicalize |                             |
|                         |  - SimHash / Title Fuzzy Deduplication|                             |
|                         |  - SQLite Deduplicated Storage        |                             |
|                         +-------------------+-------------------+                             |
|                                             |                                                 |
|                                             v                                                 |
|                         +---------------------------------------+                             |
|                         |       Multi-Tier NLP Sentiment Engine |                             |
|                         |  Tier 1: Loughran-McDonald & VADER Lex|                             |
|                         |  Tier 2: FinBERT Transformer (PyTorch)|                             |
|                         |  Tier 3: Graceful CPU/Lexicon Fallback|                             |
|                         +-------------------+-------------------+                             |
|                                             |                                                 |
|                                             v                                                 |
|                         +---------------------------------------+                             |
|                         |    Time-Series Sentiment Aggregator   |                             |
|                         |  - Exponential Half-life Decay (4h/24h|                             |
|                         |  - Polarity, Confidence, Volume Surge |                             |
|                         |  - Dispersion, Momentum, Bull/Bear Rat|                             |
|                         |  - Point-in-Time Intraday Bar Fusion  |                             |
|                         +---------------------------------------+                             |
+-----------------------------------------------------------------------------------------------+
```

### 4.1 RSS Feeds & Harvesting Strategy

1. **Google News RSS Feed**:
   - URL Pattern: `https://news.google.com/rss/search?q={QUERY}&hl=en-US&gl=US&ceid=US:en`
   - Company Query: `{SYMBOL}+stock+OR+{COMPANY_NAME}`
   - Moroccan Query: `{COMPANY_NAME}+Bourse+Casablanca&hl=fr&gl=MA&ceid=MA:fr`
2. **Yahoo Finance RSS & API**:
   - URL Pattern: `https://finance.yahoo.com/rss/headline?s={SYMBOL}`
   - Programmatic: `yfinance.Ticker(symbol).news`
3. **Casablanca Specialized Feeds**:
   - French & English news tracking for MASI components (e.g. Attijariwafa, Maroc Telecom, Marsa Maroc, HPS).

### 4.2 Feed Deduplication & SQLite Schema

Every incoming news item is passed through a multi-stage deduplication pipeline:
1. **URL Canonicalization & SHA-256 Hashing**: Strips tracking parameters (`utm_source`, `ref`, etc.) and hashes the URL.
2. **Title SimHash / Normalized Token Jaccard**: Rejects duplicate articles published across syndication networks with different URLs but identical text within $\pm 12$ hours.
3. **Database Schema (`news_sentiment.db`)**:

```sql
CREATE TABLE IF NOT EXISTS raw_articles (
    article_id TEXT PRIMARY KEY,
    url_hash TEXT UNIQUE NOT NULL,
    symbol TEXT NOT NULL,
    market TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT,
    source TEXT,
    url TEXT NOT NULL,
    published_at_utc TIMESTAMP NOT NULL,
    harvested_at_utc TIMESTAMP NOT NULL,
    raw_content_hash TEXT
);

CREATE TABLE IF NOT EXISTS sentiment_scores (
    article_id TEXT PRIMARY KEY REFERENCES raw_articles(article_id),
    symbol TEXT NOT NULL,
    published_at_utc TIMESTAMP NOT NULL,
    sentiment_score REAL NOT NULL,       -- [-1.0 (Very Bearish) to +1.0 (Very Bullish)]
    sentiment_confidence REAL NOT NULL,  -- [0.0 to 1.0]
    sentiment_label TEXT NOT NULL,       -- 'POSITIVE', 'NEUTRAL', 'NEGATIVE'
    pos_prob REAL,
    neu_prob REAL,
    neg_prob REAL,
    model_name TEXT NOT NULL,
    inference_latency_ms REAL,
    processed_at_utc TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sent_sym_pub ON sentiment_scores(symbol, published_at_utc);
CREATE INDEX IF NOT EXISTS idx_art_pub ON raw_articles(published_at_utc);
```

### 4.3 Multi-Tier NLP Sentiment Architecture

To guarantee both high accuracy and 100% operational reliability in all runtime environments:

#### Tier 1: Specialized Financial Lexicon & Rule-Based Engine (VADER + Loughran-McDonald)
- **Design**: Implements financial domain lexicons containing weighted financial sentiment stems (e.g. *guidance raised*, *earnings beat*, *downgraded*, *bankruptcy*, *lawsuit*, *FDA approval*, *dividend cut*, *acquisition*, *margin expansion*).
- **Latency**: $< 0.1$ ms per headline on CPU. Zero network or GPU dependency.
- **Formulation**:
  $$S_{lex} = \frac{\sum_{w \in T} v(w)}{\sqrt{\sum_{w \in T} v(w)^2 + \alpha}} \in [-1.0, 1.0]$$

#### Tier 2: Deep Financial Transformer (FinBERT / DistilBERT via PyTorch & HuggingFace)
- **Model**: `ProsusAI/finbert` (or `distilbert-base-uncased-finetuned-sst-2-english`).
- **Inference**: Tokenizes headline + summary, computes soft probabilities $[P_{pos}, P_{neu}, P_{neg}]$:
  $$\text{Sentiment Score} = P_{pos} - P_{neg}$$
  $$\text{Confidence} = \max(P_{pos}, P_{neu}, P_{neg})$$
- **Batch Processing**: Batches up to 32 articles per forward pass.

#### Tier 3: Resilient Fallback Engine
- When initializing the sentiment pipeline, the system attempts to load the Tier 2 Transformer model. If model weights are not pre-cached or GPU/network is constrained, it seamlessly routes all inference through the Tier 1 Financial Lexicon Engine without raising exceptions or degrading pipeline availability.

### 4.4 Sentiment Time-Series Aggregation & Intraday Feature Engineering

Irregular, discrete news article arrival times $t_1, t_2, \dots, t_K$ must be transformed into continuous, regularly spaced feature vectors aligned with 5m, 15m, 30m, 60m, and daily bars.

#### Mathematical Formulation

1. **Exponential Time Decay**:
   For any evaluation bar timestamp $t$, each preceding article $i$ published at $t_i \le t$ receives weight:
   $$w_i(t) = \exp\left( -\frac{\ln 2}{H} (t - t_i) \right)$$
   where $H$ is the half-life:
   - $H = 4$ hours for intraday trading cycles (5m to 60m).
   - $H = 24$ hours for daily swing cycles.

2. **Extracted Continuous Sentiment Features (per asset per bar $t$)**:

| Feature Name | Mathematical Definition | Financial Intuition |
|---|---|---|
| `sentiment_score_ema` | $\bar{S}(t) = \frac{\sum w_i S_i}{\sum w_i} \in [-1.0, 1.0]$ | Net directional sentiment weighted towards fresh news |
| `sentiment_confidence_ema` | $\bar{C}(t) = \frac{\sum w_i C_i}{\sum w_i} \in [0.0, 1.0]$ | Confidence level of the prevailing news tone |
| `news_volume_1h` | $N_{1h}(t) = \sum_{t - 1\text{h} \le t_i \le t} 1$ | Immediate headline volume surge / breaking news catalyst |
| `news_volume_24h` | $N_{24h}(t) = \sum_{t - 24\text{h} \le t_i \le t} 1$ | 24-hour total media attention / retail buzz |
| `sentiment_dispersion` | $\sigma_S^2(t) = \frac{\sum w_i (S_i - \bar{S})^2}{\sum w_i}$ | Narrative disagreement; high dispersion signals volatility expansion |
| `sentiment_momentum` | $\Delta \bar{S}(t) = \bar{S}(t) - \bar{S}(t - 3\text{ bars})$ | Velocity of narrative change / accelerating optimism or panic |
| `bull_bear_ratio` | $\frac{N_{pos} + 1}{N_{neg} + 1}$ | Ratio of positive to negative stories |

3. **Strict Point-in-Time Causality**:
   - For an intraday bar spanning $[t_{start}, t_{end})$, only news items with $t_{pub} \le t_{start}$ are included in the feature set for trades executed at $t_{start}$, or $t_{pub} < t_{end}$ for orders submitted at bar close.
   - If no news has occurred in the past 7 days, $\bar{S}(t) \to 0.0$ (neutral), $N \to 0$, and confidence decays gracefully to baseline.

---

## 5. Interface Specifications & Contracts

### 5.1 Data Ingestion Interface (`src/data/ingestion/`)

```python
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
from datetime import datetime

class AssetMarket(str, Enum):
    US = "US"
    BVC = "BVC"

class BarResolution(str, Enum):
    DAILY = "daily"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    M60 = "60m"

@dataclass
class DualUniverseConfig:
    us_symbols: List[str] = field(default_factory=lambda: [
        "NVDA", "AAPL", "MSFT", "AMZN", "META", "GOOGL", "TSLA",
        "AMD", "AVGO", "PLTR", "SMCI", "NFLX", "CRM", "INTC",
        "PANW", "CRWD", "NOW", "UBER", "ABNB", "COIN"
    ])
    bvc_symbols: List[str] = field(default_factory=lambda: [
        "ATW.BVC", "BCP.BVC", "IAM.BVC", "BOA.BVC", "CSR.BVC",
        "LBV.BVC", "HPS.BVC", "TQM.BVC", "TGCC.BVC", "MSA.BVC",
        "CIM.BVC", "ADH.BVC"
    ])
    benchmark_symbol: str = "SPY"
    daily_start_date: str = "1995-01-01"
    daily_end_date: str = "2026-01-01"
    intraday_lookback_days: int = 60
    cache_root: str = "data_cache"
    mad_usd_rate: float = 0.10

class IDualResolutionIngestor:
    """Contract for Dual-Resolution Data Ingestion."""
    def fetch_daily_universe(self, use_cache: bool = True) -> Dict[str, pd.DataFrame]: ...
    def fetch_intraday_universe(self, resolution: BarResolution, use_cache: bool = True) -> Dict[str, pd.DataFrame]: ...
    def fetch_single_asset_dual_resolution(self, symbol: str, market: AssetMarket) -> Tuple[pd.DataFrame, pd.DataFrame]: ...
```

### 5.2 Storage Subsystem Interface (`src/data/storage/`)

```python
class IParquetStorage:
    """Contract for Columnar Parquet Storage Subsystem."""
    def save_bars(self, symbol: str, resolution: BarResolution, df: pd.DataFrame) -> str: ...
    def load_bars(self, symbol: str, resolution: BarResolution, columns: Optional[List[str]] = None, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> Optional[pd.DataFrame]: ...
    def manifest_sync(self) -> pd.DataFrame: ...
```

### 5.3 RSS Harvester & Sentiment Interface (`src/news/`)

```python
@dataclass
class NewsArticle:
    article_id: str
    symbol: str
    market: AssetMarket
    title: str
    summary: str
    source: str
    url: str
    published_at_utc: datetime
    harvested_at_utc: datetime

@dataclass
class SentimentResult:
    article_id: str
    symbol: str
    published_at_utc: datetime
    sentiment_score: float        # [-1.0, 1.0]
    sentiment_confidence: float   # [0.0, 1.0]
    sentiment_label: str          # 'POSITIVE' | 'NEUTRAL' | 'NEGATIVE'
    probabilities: Dict[str, float]
    model_name: str

class INewsHarvester:
    """Contract for Financial News Harvester."""
    def harvest_symbol_news(self, symbol: str, market: AssetMarket, max_items: int = 50) -> List[NewsArticle]: ...
    def harvest_all_universe(self, config: DualUniverseConfig) -> Dict[str, List[NewsArticle]]: ...

class ISentimentEngine:
    """Contract for NLP Sentiment Scoring."""
    def score_article(self, article: NewsArticle) -> SentimentResult: ...
    def score_batch(self, articles: List[NewsArticle]) -> List[SentimentResult]: ...

class ISentimentAggregator:
    """Contract for Time-Series Sentiment Fusion."""
    def build_sentiment_feature_series(self, symbol: str, target_timestamps: pd.DatetimeIndex, half_life_hours: float = 4.0) -> pd.DataFrame: ...
    def fuse_with_price_bars(self, ohlcv_df: pd.DataFrame, sentiment_features_df: pd.DataFrame) -> pd.DataFrame: ...
```

---

## 6. Proposed Directory Structure & Component Layout

```
c:\Users\Anwar\Documents\trader\
├── data_cache/
│   ├── parquet/
│   │   ├── daily/
│   │   ├── intraday_5m/
│   │   ├── intraday_15m/
│   │   ├── intraday_30m/
│   │   └── intraday_60m/
│   ├── market_catalog.db
│   └── news_sentiment.db
│
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── config.py                 # Universe definitions & parameters
│   │   ├── daily_loader.py           # 30-year daily historical ingestion
│   │   ├── intraday_loader.py        # 5m-60m high-res intraday ingestion
│   │   ├── storage.py                # Parquet & SQLite storage backend
│   │   ├── synthetic.py              # Realistic intraday microstructure generator
│   │   └── aligner.py                # Multi-timeframe no-lookahead aligner & resampler
│   │
│   ├── news/
│   │   ├── __init__.py
│   │   ├── harvester.py              # Google News, Yahoo Finance & BVC RSS crawler
│   │   ├── deduplicator.py           # SHA256 URL & SimHash title deduplication
│   │   ├── sentiment_engine.py       # FinBERT Transformer + Financial Lexicon Tier 1
│   │   ├── storage.py                # SQLite news & sentiment database manager
│   │   └── aggregator.py             # Exponential decay time-series aggregator
│   │
│   ├── signals/
│   │   ├── __init__.py
│   │   ├── whale_detector.py         # Microstructure + Macro alpha signals
│   │   └── insider_scraper.py        # SEC Form 4 insider intelligence
│   │
│   ├── models/                       # (R3 DNN Modules - downstream)
│   │   ├── __init__.py
│   │   ├── dataset.py                # Multi-horizon tensor batch generator
│   │   └── neural_net.py             # Multi-horizon LSTM/TCN model
│   │
│   ├── backtest/
│   │   ├── __init__.py
│   │   ├── engine.py                 # Intraday discrete-event backtester
│   │   └── metrics.py                # Quantitative performance metrics
│   │
│   ├── execution/
│   │   ├── __init__.py
│   │   ├── simulator.py              # Almgren-Chriss impact & frictions
│   │   └── alpaca_trader.py          # Alpaca paper live execution connector
│   │
│   ├── portfolio/
│   │   ├── __init__.py
│   │   └── allocator.py              # Volatility parity & position sizing
│   │
│   └── validation/
│       ├── __init__.py
│       ├── monte_carlo.py
│       ├── cpcv.py
│       └── deflated_sharpe.py
│
├── tests/
│   ├── test_system.py                # Existing regression suite
│   ├── test_data_ingestion.py        # R1 dual-resolution ingestion tests
│   ├── test_parquet_storage.py       # R1 parquet read/write & catalog tests
│   ├── test_news_harvester.py        # R2 RSS harvesting & deduplication tests
│   ├── test_sentiment_engine.py      # R2 FinBERT & Lexicon sentiment tests
│   └── test_sentiment_alignment.py   # R2 exponential decay & alignment tests
│
├── requirements.txt
└── main.py
```

---

## 7. Implementation & Verification Roadmap

### Milestone 1 (R1 & R2 Implementation)
1. **R1 Data Ingestion & Storage**:
   - Implement `src/data/config.py`, `src/data/storage.py`, `src/data/daily_loader.py`, `src/data/intraday_loader.py`, `src/data/synthetic.py`, and `src/data/aligner.py`.
   - Validate 30-year daily and 5m-60m intraday ingestion across both US and BVC universes.
   - Verify Parquet read/write speed and SQLite catalog integrity.
2. **R2 RSS News Harvester & Sentiment Pipeline**:
   - Implement `src/news/harvester.py`, `src/news/deduplicator.py`, `src/news/sentiment_engine.py`, `src/news/storage.py`, and `src/news/aggregator.py`.
   - Validate live RSS feed polling from Google News and Yahoo Finance.
   - Test NLP sentiment inference with FinBERT and Lexicon fallback on synthetic/live headlines.
   - Verify time-decay aggregation and point-in-time intraday price alignment.
3. **Comprehensive Pytest Suite**:
   - Add automated unit tests covering all components with 100% pass rate.

---

## 8. Summary of Findings & Next Steps

- **Feasibility Confirmed**: All required libraries (`torch`, `transformers`, `yfinance`, `alpaca-py`, `pyarrow`, `beautifulsoup4`, `pytest`) are installed, verified, and operational in Python 3.14.
- **Robust Fallback Strategy**: Both data ingestion and sentiment scoring feature zero-dependency deterministic/lexicon fallbacks, ensuring resilience across local development, CI/CD, and live production environments.
- **Zero Lookahead Architecture**: Strict point-in-time lagging rules mathematically prevent future data leakage across macro daily features and intraday sentiment indicators.
