# Comprehensive Quantitative & Technical Specification
## Intraday Algorithmic Trading System with Dual-Resolution Data, NLP Sentiment Synthesis, Multi-Horizon Deep Neural Networks, and Almgren-Chriss Optimal Execution

**Author:** `survey_spec_miner_1` (teamwork_preview_spec_miner)  
**Date:** 2026-08-31  
**Integrity Mode:** Development  
**Working Directory:** `c:\Users\Anwar\Documents\trader`  
**Reference Request:** `ORIGINAL_REQUEST.md`  

---

## 1. Executive Summary & Specification Scope

This document establishes the formal quantitative, mathematical, and architectural specification for an institutional-grade intraday algorithmic trading system operating on **5-minute to 60-minute cycles**. The system bridges macro structural trends with micro-level liquidity anomalies and real-time textual news sentiment, deploying a **Multi-Horizon Deep Neural Network (DNN)** to generate discrete `BUY`, `HOLD`, and `SELL` decisions across a dual investable universe:
1. **US Equities (NYSE / NASDAQ)**: Deep liquidity, continuous quoting, millisecond feeds, and Alpaca Paper Trading execution.
2. **Casablanca Stock Exchange (Bourse de Casablanca - BVC)**: Emerging frontier liquidity, 10%/6% regulatory circuit breakers, $T+3$ clearing delay, and 15% withholding tax friction.

The specification details every mathematical equation, data schema, API contract, error recovery protocol, and acceptance threshold required for implementation and verification.

---

## 2. Mathematical Formulations

### 2.1 Almgren-Chriss Optimal Execution & Market Impact Model

Trade execution is formalized under the stochastic control framework of **Almgren & Chriss (2000)**. When rebalancing a target portfolio or executing an order of $X_0$ shares over an execution horizon $T$ divided into $N$ discrete intervals $\tau = T / N$:

Let $x_k$ be the remaining shares to execute at time step $t_k = k \tau$ with boundary conditions $x_0 = X_0$ and $x_N = 0$. The trade size in interval $(t_{k-1}, t_k]$ is $n_k = x_{k-1} - x_k$, with execution rate $v_k = n_k / \tau$.

#### 2.1.1 Asset Price Dynamics with Market Impact
The mid-price $S_k$ evolves according to:
$$S_k = S_{k-1} + \sigma \tau^{1/2} \xi_k - \tau \gamma(v_k)$$
where:
- $\sigma$ is daily/intraday asset return volatility.
- $\xi_k \sim \mathcal{N}(0, 1)$ are i.i.d. random price shocks.
- $\gamma(v_k) = \gamma v_k = \gamma \frac{n_k}{\tau}$ represents **permanent price impact**, an irreversible shift in the equilibrium price reflecting information leakage and persistent order flow.

#### 2.1.2 Effective Execution Price & Temporary Impact
The actual execution price received on trade $n_k$ includes the **temporary price impact** $\eta(v_k)$ and half bid-ask spread $\epsilon$:
$$\tilde{S}_k = S_{k-1} - \text{sgn}(n_k) \cdot \left[ \epsilon + \eta(v_k) \right]$$
with linear temporary impact:
$$\eta(v_k) = \eta \frac{|n_k|}{\tau} = \eta |v_k|$$

#### 2.1.3 Implementation Shortfall & Expected Utility Optimization
The total execution cost (Implementation Shortfall $x$) and its variance are:
$$\mathbb{E}[x] = \frac{1}{2} \gamma X_0^2 + \epsilon \sum_{k=1}^N |n_k| + \frac{\eta}{\tau} \sum_{k=1}^N n_k^2$$
$$\mathbb{V}[x] = \sigma^2 \sum_{k=1}^N \tau x_k^2$$

The optimal execution trajectory minimizes the risk-adjusted shortfall objective $U(x) = \mathbb{E}[x] + \lambda \mathbb{V}[x]$ with risk aversion parameter $\lambda \ge 0$:
$$x_j = \frac{\sinh(\kappa (T - t_j))}{\sinh(\kappa T)} X_0, \quad \text{where } \kappa \approx \sqrt{\frac{\lambda \sigma^2}{\eta}}$$

#### 2.1.4 Intraday Discrete-Event Execution Model Formulation
For bar-by-bar discrete simulation at 5m–60m intervals:
$$\text{Participation Rate: } \rho_t = \frac{|\Delta q_t|}{V_{\text{bar}, t}}$$
$$\text{Effective Price: } P_{\text{exec}, t} = P_{\text{mid}, t} \times \left(1 \pm \left[ \epsilon_{\text{half\_spread}} + \eta_{\text{temp}} \cdot \rho_t^{\alpha} + \gamma_{\text{perm}} \cdot \rho_t^{\beta} + C_{\text{broker}} \right]\right)$$

**Calibrated Market Parameters:**
| Parameter | Symbol | US Equities (High Liquidity) | Casablanca BVC (Low Liquidity) |
| :--- | :--- | :--- | :--- |
| Temporary Impact Coeff | $\eta_{\text{temp}}$ | $0.03 - 0.05$ | $0.15 - 0.25$ |
| Permanent Impact Coeff | $\gamma_{\text{perm}}$ | $0.01 - 0.02$ | $0.08 - 0.15$ |
| Non-linear Exponent | $\alpha, \beta$ | $1.0$ (linear) or $0.5$ (sqrt law) | $1.0$ |
| Half Bid-Ask Spread | $\epsilon_{\text{half\_spread}}$ | $2.0 \text{ bps } (0.02\%)$ | $15.0 \text{ bps } (0.15\%)$ |
| Brokerage Commission | $C_{\text{broker}}$ | $3.0 - 5.0 \text{ bps } (0.03\% - 0.05\%)$ | $15.0 - 25.0 \text{ bps } (0.15\% - 0.25\%)$ |
| Max Participation Cap | $\rho_{\max}$ | $0.25 \ (25\% \text{ bar volume})$ | $0.10 \ (10\% \text{ bar volume})$ |
| Capital Gains Withholding Tax | $\tau_{\text{tax}}$ | $0.0\%$ (broker level) | $15.0\%$ Moroccan IR on net realized gains |
| Settlement Latency | $T_{\text{settle}}$ | $T+1$ / Immediate for cash | $T+3$ Trading Days |
| Regulatory Circuit Breaker | $\text{CB}_{\max}$ | Dynamic LULD bands | $\pm 10.0\%$ Continuous / $\pm 6.0\%$ Fixing |

---

### 2.2 Performance & Risk Metric Formulations

All performance indicators are computed net of commissions, slippage, and taxes:

#### 2.2.1 Compound Annual Growth Rate (CAGR)
Given net asset value time series $E_t \ (t \in [0, T])$ over total duration $Y$ years ($Y = \frac{N_{\text{days}}}{252.0}$ or $Y = \frac{N_{\text{bars}}}{252 \times N_{\text{bars/day}}}$):
$$\text{CAGR} = \left( \frac{E_T}{E_0} \right)^{\frac{1}{Y}} - 1$$
- **Target Acceptance Criterion:** $\text{CAGR} \ge 35.0\% \ (0.35)$ in out-of-sample backtests on 5m–60m cycles.

#### 2.2.2 Maximum Drawdown (MDD) & Drawdown Duration
$$\text{Peak}_t = \max_{s \in [0, t]} E_s$$
$$\text{Drawdown}_t = \frac{E_t - \text{Peak}_t}{\text{Peak}_t}$$
$$\text{MDD} = \max_{t \in [0, T]} \left( -\text{Drawdown}_t \right) = \left| \min_{t \in [0, T]} \text{Drawdown}_t \right|$$
$$\text{Max Drawdown Duration} = \max \left\{ \text{length of continuous intervals where } \text{Drawdown}_t < 0 \right\}$$
- **Target Acceptance Criterion:** $\text{MDD} \le 20.0\% \ (0.20)$.

#### 2.2.3 Annualized Sharpe Ratio
Let $R_t = \frac{E_t - E_{t-1}}{E_{t-1}}$ be bar returns, $M$ be the annualization factor ($M = 252 \times 78 = 19,656$ for 5m bars; $M = 252 \times 6.5 = 1,638$ for 60m bars; $M = 252$ for daily), and $R_f = 0.02$ (2% risk-free rate):
$$R_{f, \text{bar}} = (1 + R_f)^{1/M} - 1$$
$$\bar{r}_e = \frac{1}{K} \sum_{t=1}^K (R_t - R_{f, \text{bar}})$$
$$\sigma_R = \sqrt{\frac{1}{K-1} \sum_{t=1}^K (R_t - \bar{R})^2}$$
$$\text{Sharpe Ratio} = \frac{\bar{r}_e}{\sigma_R} \times \sqrt{M}$$
- **Target Acceptance Criterion:** $\text{Sharpe Ratio} \ge 1.80$.

#### 2.2.4 Sortino, Calmar, and Deflated Sharpe Ratio
- **Sortino Ratio:**
  $$\sigma_{\text{downside}} = \sqrt{\frac{1}{K} \sum_{t=1}^K \min(0, R_t - R_{f, \text{bar}})^2} \times \sqrt{M}$$
  $$\text{Sortino} = \frac{\bar{r}_e \times M}{\sigma_{\text{downside}}}$$
- **Calmar Ratio:**
  $$\text{Calmar} = \frac{\text{CAGR}}{\text{MDD}}$$
- **Deflated Sharpe Ratio (DSR)** (Bailey & Lopez de Prado 2014):
  Evaluates statistical significance against multiple trial selection bias:
  $$\text{DSR} = \Phi\left( \frac{(\widehat{\text{SR}} - \text{SR}_0^*) \sqrt{K-1}}{\sqrt{1 - \widehat{\gamma}_3 \widehat{\text{SR}} + \frac{\widehat{\gamma}_4 - 1}{4} \widehat{\text{SR}}^2}} \right)$$
  where $\text{SR}_0^* = \sqrt{\frac{2 \ln N_{\text{trials}}}{K}}$, $\widehat{\gamma}_3$ is skewness, and $\widehat{\gamma}_4$ is kurtosis of returns.

---

### 2.3 Multi-Horizon Loss & Reward Formulations for BUY/HOLD/SELL Signal Generation

#### 2.3.1 Input State Space Representation
At any time bar $t$, the state representation $X_t \in \mathbb{R}^D$ integrates three temporal streams:
$$X_t = \Big[ X_t^{\text{macro\_daily}}, X_t^{\text{intraday\_micro}}, X_t^{\text{nlp\_sentiment}} \Big]$$
1. **Macro Daily Stream ($X_t^{\text{macro\_daily}} \in \mathbb{R}^{D_1}$)**:
   - 30-year trend EMA/SMA ratios: $P_t / \text{SMA}_{50}(t), P_t / \text{SMA}_{200}(t)$
   - 12-Month Momentum: $\text{Mom}_{12M}(t) = \frac{P_t}{P_{t-252}} - 1$
   - 52-Week High Ratio: $\text{Ratio}_{52W}(t) = P_t / \max_{s \in [t-252, t]} P_s$
   - Macro Volatility: $\sigma_{20D}(t)$
   - Benchmark SPY Regime Filter: $\mathbf{1}_{\{\text{SPY}_t \ge \text{SMA}_{200}(\text{SPY}_t)\}}$
2. **Microstructure Intraday Stream ($X_t^{\text{intraday\_micro}} \in \mathbb{R}^{D_2}$)**:
   - Relative Volume: $\text{RVOL}_t = \frac{V_t}{\text{SMA}_{20}(V_t)}$
   - Volume Z-Score: $Z_{V, t} = \frac{V_t - \mu_{V, 30}}{\sigma_{V, 30}}$
   - Intraday VWAP Deviation: $\Delta_{\text{VWAP}, t} = \frac{P_t - \text{VWAP}_t}{\text{VWAP}_t}$
   - Normalised Order Flow Imbalance (Matched Filter):
     $$\text{OFI}_t = \frac{V_t \cdot (C_t - O_t)}{(H_t - L_t + \epsilon) \cdot \text{MarketCap}_t}$$
   - Wyckoff VSA Features: Candle Spread Ratio, Close Location Value $\text{CLV}_t = \frac{C_t - L_t}{H_t - L_t}$, Absorption & Spring flags.
3. **NLP Sentiment Stream ($X_t^{\text{nlp\_sentiment}} \in \mathbb{R}^{D_3}$)**:
   - Rolling Compound Sentiment $S_t \in [-1, 1]$
   - Sentiment Velocity $\Delta S_t = S_t - S_{t-12}$
   - News Volume Burst $V_{\text{news}, t} = \frac{N_{\text{articles}}(t-12, t)}{\bar{N}_{\text{baseline}}}$

#### 2.3.2 Multi-Horizon Target Return & Signal Labeling
Let $H = \{ h_1, h_2, h_3 \}$ be the forecast horizon set (e.g. $h_1 = 3 \text{ bars}$ [15m], $h_2 = 12 \text{ bars}$ [1h], $h_3 = 78 \text{ bars}$ [1 day]).
The forward multi-horizon return vector is:
$$y_{t, h} = \frac{P_{t+h} - P_t}{P_t}, \quad \forall h \in H$$

Continuous targets are discretized into 3 discrete decision classes $y_t^{\text{class}} \in \{ \text{BUY } (0), \text{HOLD } (1), \text{SELL } (2) \}$ using an adaptive volatility barrier $\theta_t = \beta \cdot \frac{\text{ATR}_{14}(t)}{P_t}$:
$$y_t^{\text{class}} = \begin{cases}
\text{BUY } (0), & \text{if } \sum_{h \in H} w_h y_{t, h} > +\theta_t \\
\text{SELL } (2), & \text{if } \sum_{h \in H} w_h y_{t, h} < -\theta_t \\
\text{HOLD } (1), & \text{otherwise}
\end{cases}$$
with horizon weights $w = [0.2, 0.3, 0.5]$.

#### 2.3.3 Multi-Task Loss Formulation
The Deep Neural Network parameterized by weights $\Theta$ outputs class probability logits $\hat{\mathbf{p}}_t = \text{softmax}(f_\Theta(X_t)) \in \mathbb{R}^3$ and regression return estimates $\hat{y}_{t, h} \in \mathbb{R}^{|H|}$.

1. **Focal Classification Loss** (Addressing severe HOLD class imbalance):
   $$\mathcal{L}_{\text{Focal}}(\Theta) = - \sum_{c=0}^2 \alpha_c (1 - \hat{p}_{t, c})^\gamma \cdot \mathbf{1}_{\{y_t^{\text{class}} = c\}} \log(\hat{p}_{t, c})$$
   where focusing parameter $\gamma = 2.0$, class weights $\alpha = [1.5, 0.5, 1.5]$.
2. **Multi-Horizon Huber Regression Loss**:
   $$\mathcal{L}_{\text{Huber}}(\Theta) = \sum_{h \in H} w_h \cdot \begin{cases} 
   \frac{1}{2} (y_{t, h} - \hat{y}_{t, h})^2, & \text{for } |y_{t, h} - \hat{y}_{t, h}| \le \delta \\
   \delta \cdot |y_{t, h} - \hat{y}_{t, h}| - \frac{1}{2} \delta^2, & \text{otherwise}
   \end{cases}$$
3. **Differential Sharpe Policy Reward Loss**:
   Let target position $a_t = \hat{p}_{t, \text{BUY}} - \hat{p}_{t, \text{SELL}} \in [-1, 1]$. The net strategy return is $R_t^{\text{strat}} = a_{t-1} r_t - C_{\text{frict}} |a_t - a_{t-1}|$.
   $$\mathcal{L}_{\text{Sharpe}}(\Theta) = - \frac{\mathbb{E}[R_t^{\text{strat}}]}{\sqrt{\mathbb{V}[R_t^{\text{strat}}] + \epsilon_{\text{reg}}}} \times \sqrt{M}$$
4. **Total Composite Optimization Objective**:
   $$\min_{\Theta} \mathcal{L}_{\text{total}} = \lambda_1 \mathcal{L}_{\text{Focal}} + \lambda_2 \mathcal{L}_{\text{Huber}} + \lambda_3 \mathcal{L}_{\text{Sharpe}} + \lambda_4 \|\Theta\|_2^2$$
   where $\lambda_1 = 1.0, \lambda_2 = 0.5, \lambda_3 = 0.8, \lambda_4 = 10^{-5}$.

---

## 3. API & Data Contracts

### 3.1 Dual-Resolution Historical & Intraday Ingestion Pipeline Contract

#### 3.1.1 Endpoints & Ingestion Sources
- **US Equities 30-Year Daily:** `yfinance` (`yf.Ticker(sym).history(start='1994-01-01', end='2026-01-01')`) or Alpaca Historical Data API.
- **US Equities 5m–60m Intraday:** `yfinance` (`interval='5m'` for last 60 days, `interval='60m'` for last 730 days) and Alpaca Market Data API (`StockHistoricalDataClient` with `StockBarsRequest`).
- **Casablanca BVC Daily & Intraday:** BVCscrap / Parse.bot / ICE Data Services, with realistic microstructure generative fallback.

#### 3.1.2 Data Schema Specification
Every bar record must conform to the following strict DataFrame schema:

| Column Name | Data Type | Nullable | Description & Constraints |
| :--- | :--- | :--- | :--- |
| `timestamp` (Index) | `datetime64[ns, UTC]` | No | Unique, strictly monotonically increasing UTC timestamp |
| `Open` | `float64` | No | Opening price ($> 0.0$, rounded to 2 decimals) |
| `High` | `float64` | No | High price ($\ge \max(\text{Open}, \text{Close})$) |
| `Low` | `float64` | No | Low price ($\le \min(\text{Open}, \text{Close})$) |
| `Close` | `float64` | No | Closing price ($> 0.0$, rounded to 2 decimals) |
| `Volume` | `float64` | No | Total executed bar volume ($\ge 0$) |
| `VWAP` | `float64` | Yes | Volume-weighted average price over the bar |
| `Trades` | `int64` | Yes | Number of discrete trades during the bar |
| `Market` | `str` (Enum) | No | `"US"` or `"BVC"` |
| `Currency` | `str` (Enum) | No | `"USD"` or `"MAD"` |
| `MarketCap` | `float64` | Yes | Estimated market capitalization |

#### 3.1.3 Storage & File Layout
```
data_cache/
├── daily/
│   ├── AAPL.parquet
│   ├── NVDA.parquet
│   └── ATW_BVC.parquet
├── intraday_5m/
│   ├── AAPL_5m.parquet
│   └── NVDA_5m.parquet
├── intraday_60m/
│   ├── AAPL_60m.parquet
│   └── NVDA_60m.parquet
└── news_nlp/
    └── sentiment_timeseries.parquet
```

---

### 3.2 Exchange Calendars, Market Holidays & Timezone Alignment

| Market | Primary Exchange | Local Timezone | Continuous Session (Local) | UTC Session Hours | Circuit Breaker Rule |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **US Equities** | NYSE / NASDAQ | `America/New_York` (EDT/EST) | 09:30 – 16:00 ET | 13:30 – 20:00 UTC (EDT)<br>14:30 – 21:00 UTC (EST) | LULD Phase 1/2 Price Bands |
| **Casablanca Equities** | Bourse de Casablanca | `Africa/Casablanca` (UTC+1 / UTC) | 09:30 – 15:30 Casablanca | 08:30 – 14:30 UTC (Standard)<br>09:30 – 15:30 UTC (Ramadan) | $\pm 10.0\%$ Continuous / $\pm 6.0\%$ Fixing |

**Timezone Harmonization Rules:**
1. All timestamp indices are normalized to ISO 8601 UTC.
2. Backtest simulation clock advances step-by-step in UTC.
3. Market opening and closing events fire according to their respective exchange calendar (`pandas_market_calendars` or explicit holiday lookup tables).
4. Dual-universe portfolio accounting evaluates Casablanca assets in USD via real-time FX conversion:
   $$\text{Value}_{\text{USD}} = \text{Shares} \times P_{\text{MAD}} \times \text{FX}_{\text{MAD}\to\text{USD}}$$
   where $\text{FX}_{\text{MAD}\to\text{USD}} \approx 0.10 \ (10 \text{ MAD} / 1 \text{ USD})$.

---

### 3.3 RSS News Harvester & Local NLP Sentiment Analysis Engine

#### 3.3.1 Feed Ingestion Endpoints
- **Yahoo Finance RSS:** `https://feeds.finance.yahoo.com/rss/2.0/headline?s={symbol}&region=US&lang=en-US`
- **Google News RSS:** `https://news.google.com/rss/search?q={symbol}+stock+market&hl=en-US&gl=US&ceid=US:en`
- Polling Interval: 15 minutes (or 5 minutes before market open / during active trading hours).

#### 3.3.2 Article Record Schema
```json
{
  "article_id": "sha256_hash_of_url_and_pubdate",
  "symbol": "NVDA",
  "source": "YahooFinance",
  "title": "NVIDIA Unveils Next-Gen AI Architecture",
  "summary": "Full text snippet or description...",
  "url": "https://finance.yahoo.com/news/...",
  "published_at": "2026-08-31T14:15:00Z",
  "harvested_at": "2026-08-31T14:16:30Z"
}
```

#### 3.3.3 Local NLP Sentiment Extraction Schema
Local zero-credential NLP inference using VADER / FinBERT (`ProsusAI/finbert` via HuggingFace transformers or local PyTorch/TorchScript model):

```json
{
  "article_id": "sha256_hash_of_url_and_pubdate",
  "symbol": "NVDA",
  "sentiment_positive": 0.884,
  "sentiment_negative": 0.012,
  "sentiment_neutral": 0.104,
  "sentiment_compound": 0.872,
  "relevance_score": 0.95,
  "confidence": 0.92,
  "model_used": "finbert_local"
}
```

#### 3.3.4 Intraday Sentiment Aggregation Equation
At bar timestamp $t$, the sentiment score $S_{\text{agg}}(t)$ is computed over lookback window $W = 24 \text{ hours}$ using exponential time decay:
$$S_{\text{agg}}(t) = \frac{\sum_{i \in \text{articles in } [t-W, t]} \text{Rel}_i \cdot \text{Conf}_i \cdot S_{\text{compound}, i} \cdot e^{-\lambda (t - t_i)}}{\sum_{i} \text{Rel}_i \cdot \text{Conf}_i \cdot e^{-\lambda (t - t_i)} + \epsilon}$$
where half-life $t_{1/2} = 6 \text{ hours} \implies \lambda = \frac{\ln 2}{6 \times 3600 \text{ sec}}$.

---

### 3.4 Alpaca Paper Trading REST/WebSocket Interface & Order Lifecycle

#### 3.4.1 Authentication & Configuration
- **Base URL:** `https://paper-api.alpaca.markets`
- **Data URL:** `https://data.alpaca.markets/v2`
- **Auth Headers:**
  - `APCA-API-KEY-ID`: API key ID
  - `APCA-API-SECRET-KEY`: Secret key
- **Environment Fallback:** If keys are absent or invalid, system operates in **Autonomous Dry-Run Simulation Mode** without failing or blocking execution.

#### 3.4.2 Order Submission Contract
Bracket order payload for BUY recommendations:
```python
MarketOrderRequest(
    symbol=symbol,
    qty=shares_qty,
    side=OrderSide.BUY,
    time_in_force=TimeInForce.DAY,
    stop_loss=StopLossRequest(stop_price=round(current_price * 0.952, 2)),
    take_profit=TakeProfitRequest(limit_price=round(current_price * 1.150, 2))
)
```

#### 3.4.3 Order State Machine
$$\begin{array}{rcl}
\text{pending\_new} & \longrightarrow & \text{accepted} \longrightarrow \text{new} \\
& & \quad \Big\downarrow \\
& & \text{partially\_filled} \longrightarrow \text{filled} \quad (\text{Terminal}) \\
& & \quad \Big\downarrow \\
& & \text{canceled } / \text{ rejected } / \text{ expired } \quad (\text{Terminal})
\end{array}$$

---

## 4. Architecture & Component Decomposition

```
                          ┌────────────────────────────────────────────────────────┐
                          │               Market Data Ingestion Pipeline           │
                          │   • 30-Year Daily (yfinance / Alpaca / Synthetic BVC) │
                          │   • Intraday 5m–60m Bars (yfinance / Alpaca API)      │
                          │   • Timezone Normalization (UTC / US / Casablanca)     │
                          └───────────────────────────┬────────────────────────────┘
                                                      │
                                                      ▼
┌──────────────────────────────────────┐  ┌────────────────────────────────────────┐
│      RSS News & NLP Subsystem        │  │     Microstructure & Feature Engine    │
│  • Yahoo Finance / Google News Poller│  │  • Volume Z-Score & RVOL (30-day/20-bar│
│  • VADER / FinBERT Sentiment Scorer  │  │  • Wyckoff VSA (Absorption, Spring)    │
│  • Exponential Time Decay Aggregator │  │  • Normalized Order Flow Imbalance     │
└──────────────────┬───────────────────┘  └───────────────────┬────────────────────┘
                   │                                          │
                   └──────────────────┬───────────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │    Multi-Horizon Deep Neural Network │
                   │  • Temporal Convolutions / LSTM / GRU│
                   │  • Macro + Micro + Sentiment Fusion  │
                   │  • BUY / HOLD / SELL Signal Output   │
                   └──────────────────┬───────────────────┘
                                      │
                                      ▼
                   ┌──────────────────────────────────────┐
                   │    Portfolio Allocation & Sizing     │
                   │  • Risk-Parity Volatility Scaling    │
                   │  • Hysteresis Rank Selection Cap     │
                   │  • Momentum Crash 52W High Shield    │
                   └──────────────────┬───────────────────┘
                                      │
                   ┌──────────────────┴───────────────────┐
                   ▼                                      ▼
┌──────────────────────────────────────┐  ┌────────────────────────────────────────┐
│   Discrete-Event Intraday Backtester │  │  Live Alpaca Paper Trading Connector   │
│  • 5m–60m Event Execution Loop       │  │  • Authenticated REST Client           │
│  • Almgren-Chriss Impact & Slippage  │  │  • Bracket Orders (Stop-Loss/Take-Pft) │
│  • BVC T+3 Clearing & 15% IR Tax     │  │  • Automated Scheduled Cron Driver     │
│  • Benchmark SPY Comparison          │  │  • Dry-Run Mode & State Telemetry      │
└──────────────────────────────────────┘  └────────────────────────────────────────┘
```

---

## 5. Comprehensive Features Discovered

| # | Category | Feature | Description | Inputs | Outputs | Error Behavior | Discovered Via |
|---|----------|---------|-------------|--------|---------|----------------|----------------|
| **F01** | Data Pipeline | 30-Year Daily Ingestion | Pulls daily OHLCV from 1994 to 2026 for US & BVC symbols | Symbol, Start/End date, Cache flag | Cleaned `pd.DataFrame` with UTC DatetimeIndex | Falls back to cached data or synthetic statistical generator | `ORIGINAL_REQUEST.md` R1 |
| **F02** | Data Pipeline | Intraday 5m–60m Ingestion | Fetches high-resolution intraday candles (5m up to 60 days, 60m up to 730 days) | Symbol, Interval (`5m`/`60m`), Date window | Normalized OHLCV intraday DataFrame | Log warning, resample from finer granularity or yfinance fallback | `ORIGINAL_REQUEST.md` R1 |
| **F03** | Data Pipeline | Dual-Market Timezone Alignment | Harmonizes US/Eastern, Casablanca (UTC+1/UTC), and UTC timestamps into unified timeline | Raw bar timestamps with local timezones | Unified `datetime64[ns, UTC]` index | Converts DST shifts cleanly without NaNs | R1 & codebase analysis |
| **F04** | Data Pipeline | Dual-Universe Liquidity Filter | Enforces $1,000,000 USD/day (US) and 100,000 MAD/day (BVC) dollar volume thresholds | Price, Volume, Market type | Boolean `Passes_Dollar_Volume` series | Masks illiquid bars to zero allocation | Strategy Spec §2.1 |
| **F05** | NLP Sentiment | RSS Feed Crawler | Polls Yahoo Finance and Google News RSS feeds for company-specific news articles | Symbol list, RSS feed URLs | List of parsed article dictionaries | Retries on network timeout, returns cached articles | `ORIGINAL_REQUEST.md` R2 |
| **F06** | NLP Sentiment | Local NLP Sentiment Scorer | Runs VADER / FinBERT on article titles and summaries to compute numerical sentiment | Text strings (title, summary) | Scores: positive, negative, neutral, compound $\in [-1, 1]$ | Falls back to Loughran-McDonald lexicon on model error | `ORIGINAL_REQUEST.md` R2 |
| **F07** | NLP Sentiment | Exponential Time-Decayed Aggregator | Aggregates article scores into hourly/daily sentiment indicators with exponential decay | Stream of article scores and timestamps | Aggregated continuous sentiment $S_{\text{agg}}(t)$ | Defaults to 0.0 (neutral) when no news present | `ORIGINAL_REQUEST.md` R2 |
| **F08** | Alpha Signals | Wyckoff VSA Engine | Identifies Volume Spread Analysis patterns (Absorption, Spring, Upthrust, Breakout) | OHLCV series | Boolean flags: `Wyckoff_Absorption`, `Wyckoff_Spring`, etc. | Sets flags to 0 if candle range is zero | `whale_detector.py` |
| **F09** | Alpha Signals | Order Flow Imbalance (OFI) | Computes Matched-Filter OFI normalized by Market Capitalization | OHLCV, MarketCap | Normalized Z-Score `OFI_Norm_Z` | Fills initial rolling NaN with 0.0 | Strategy Spec §3.2 |
| **F10** | Alpha Signals | Isolation Forest Anomaly Detection | Unsupervised ML anomaly detector isolating institutional volume signatures | Multidimensional feature matrix | `ML_Anomaly_Score`, `ML_Is_Whale` | Falls back to rule-based logic if sklearn fails | `whale_detector.py` |
| **F11** | Deep Learning | Multi-Horizon DNN Architecture | Temporal Convolutional Network / LSTM / GRU processing macro, micro, and sentiment features | 3D tensor of lookback features $(B, T, D)$ | Predictions: Class logits $\in \mathbb{R}^3$, return estimates $\in \mathbb{R}^{|H|}$ | Initialized with Xavier/Kaiming weights, logs training losses | `ORIGINAL_REQUEST.md` R3 |
| **F12** | Deep Learning | Multi-Task Focal + Sharpe Loss | Loss function balancing class imbalance with direct differential Sharpe ratio maximization | Predictions, true barrier labels, returns | Scalar loss $\mathcal{L}_{\text{total}}$ | Clips gradients to prevent exploding loss | R3 Formulation |
| **F13** | Deep Learning | Adaptive Volatility Barrier Labeler | Generates BUY/HOLD/SELL ground truth labels using dynamic ATR barrier thresholds | Multi-horizon future prices, ATR | Target class labels $y_t^{\text{class}} \in \{0, 1, 2\}$ | Sets to HOLD (1) during zero volatility regimes | R3 Formulation |
| **F14** | Portfolio | Risk-Parity Inverse-Variance Scaling | Allocates capital inversely proportional to realized asset volatility ($w_i \propto 1/\sigma_i^{1.1}$) | Volatilities, conviction multipliers | Target weights DataFrame | Normalizes sum to $\le 1.0$, clips max weight to 34% | `allocator.py` |
| **F15** | Portfolio | Momentum Crash 52-Week High Shield | Penalizes speculative rebound losers using 52-week high ratio squared ($w_i \times (P/P_{52W})^2$) | Prices, 52-week rolling max | Scaled momentum scores | Zeroes momentum score if $< 10\%$ annual threshold | `allocator.py` |
| **F16** | Portfolio | Hysteresis Rank Selection | Top-$k_{\text{entry}}$ entry and Top-$k_{\text{hold}}$ holding rules preventing rank turnover churn | Asset scores across universe | Filtered sticky active position set | Holds existing positions if still within top hold buffer | `allocator.py` |
| **F17** | Execution | Almgren-Chriss Market Impact Simulator | Calculates temporary and permanent price impact as a function of bar participation rate | Order value, current price, bar volume, market type | `ExecutionResult` (shares, effective price, slippage) | Clips participation rate to max ceiling (e.g. 25%) | `simulator.py` |
| **F18** | Execution | Casablanca BVC Frictions & $T+3$ Queue | Simulates Moroccan 15% IR withholding tax on realized gains and 3-day cash settlement queue | Market type, trade proceeds, entry price, day index | Settled cash queue, tax withheld amount | Prevents deployment of unsettled cash until day $t+3$ | `simulator.py` |
| **F19** | Execution | Regulatory Circuit Breaker Clamping | Clamps daily price moves to $\pm 10\%$ (Continuous) / $\pm 6\%$ (Fixing) for BVC equities | Raw price returns | Clamped returns and limit flags | Prevents fill if stock is locked at Limit Up / Down | `loader.py` & `simulator.py` |
| **F20** | Backtesting | Discrete-Event Intraday Backtester | Rebalances on 5m–60m cycles with trailing stops and mark-to-market NAV accounting | Intraday OHLCV, target weights, initial capital | `BacktestResult` (equity series, trade log, metrics) | Raises ValueError if data dict is empty | `engine.py` |
| **F21** | Backtesting | Benchmark SPY Comparison | Simulates buy-and-hold SPY benchmark alongside strategy equity curve | SPY daily/intraday price series | Benchmark equity series | Falls back to deterministic 10% annual proxy if SPY missing | `engine.py` |
| **F22** | Live Trading | Alpaca Paper Trader Connector | Connects to Alpaca Paper Trading REST API with authentication and account telemetry | API keys, symbol universe | Account balance, current positions, buying power | Falls back to Dry-Run simulation mode if unauthenticated | `alpaca_trader.py` |
| **F23** | Live Trading | Server-Side Bracket Stop-Loss Submission | Submits Market Order with attached Stop-Loss bracket (-4.8%) to Alpaca broker | Symbol, target shares, stop price | Alpaca Order object / confirmation | Catches API errors, logs failures without crashing cycle | `alpaca_trader.py` |
| **F24** | Live Trading | Autonomous Dry-Run Mode | Executes full end-to-end signal calculation and simulated order log without external credentials | Live market feed | `output/live_dry_run_log.json` snapshot | Fully self-contained, runs anywhere zero-config | `scan_current_market.py` |
| **F25** | Validation | Block-Bootstrap Monte Carlo Engine | 2,500-iteration circular block bootstrap quantifying probability of $\text{CAGR} \ge 30\%$ | Strategy equity series, block size | `MonteCarloResult` (p-value, 5th/50th/95th percentiles) | Requires $\ge 50$ return observations | `monte_carlo.py` |
| **F26** | Validation | Combinatorial Purged Cross-Validation | Evaluates strategy across out-of-sample combinations with purging and embargoing | Strategy returns, $N_{\text{groups}}, K_{\text{test}}$ | `CPCVResult` (mean OOS CAGR, positive path ratio) | Raises warning if slices have insufficient samples | `cpcv.py` |
| **F27** | Validation | Deflated Sharpe Ratio (DSR) | Computes Bailey & Lopez de Prado DSR adjusted for skewness, kurtosis, and trial count | Strategy returns, trial count $N_{\text{trials}}$ | DSR p-value $\in [0, 1]$ | Returns 0.50 if sample size $< 30$ | `deflated_sharpe.py` |
| **F28** | Reporting | Multi-Panel Visual Tearsheet Generator | Generates 6-panel institutional tearsheet (Equity, Drawdown, Heatmap, Sharpe, MC, Alloc) | Backtest, Monte Carlo, and CPCV results | `output/tearsheet.png`, `performance_report.json` | Uses headless matplotlib backend for server runs | `tearsheet.py` |

---

## 6. Edge Cases & Boundary Conditions

| # | Feature | Input Condition / Scenario | Expected / Observed Behavior |
|---|---------|----------------------------|------------------------------|
| **E01** | Data Pipeline | Incomplete / Missing Intraday Bars (e.g. trading halt, low volume) | System applies forward-fill (`ffill()`) up to 3 bars; if gap $> 3$ bars, marks bar volume as 0.0 and skips trade execution to avoid synthetic fills. |
| **E02** | Timezone Alignment | US Daylight Saving Time (EDT $\leftrightarrow$ EST) and Moroccan Ramadan time shift | Timestamps normalized in UTC; UTC session offsets shift dynamically (e.g. US open at 13:30 UTC in EDT vs 14:30 UTC in EST) without duplicate or dropped bars. |
| **E03** | RSS News Harvester | Network timeout, HTTP 429 rate limit, or invalid RSS XML formatting | Catch request exceptions, log warning, back off exponentially, and return empty list; sentiment aggregator maintains prior decayed score without crashing. |
| **E04** | NLP Sentiment | Zero news articles published for an asset over the last 72 hours | Exponential decay reduces sentiment weight toward 0.0 (neutral baseline); DNN sentiment input cleanly defaults to 0.0 without generating NaNs. |
| **E05** | DNN Signal Output | Extreme market turbulence producing flat/zero confidence across all classes | DNN output entropy increases; if $\max(\hat{p}_{\text{BUY}}, \hat{p}_{\text{SELL}}) < 0.45$, system falls back to default HOLD (cash defense). |
| **E06** | Execution Impact | Large order exceeding 25% of bar volume in an illiquid asset | Participation rate is clamped at $\rho_{\max} = 0.25$; excess unexecuted volume is sliced into subsequent bars rather than blowing up slippage. |
| **E07** | Casablanca BVC | Consecutive Limit-Up days (+10% reservation without ask quotes) | System detects Limit-Up condition ($P_t = P_{t-1} \times 1.10$ and Ask Volume $= 0$); buys are rejected or deferred to avoid non-executable simulated fills. |
| **E08** | Portfolio Sizing | All universe assets triggered simultaneously during a broad market breakout | Risk-parity weights are normalized to ensure $\sum w_i \le 1.00$; individual weights are strictly capped at $\le 34\%$, preventing over-leverage. |
| **E09** | Alpaca Connector | API credentials absent or invalid during live scheduled cron | System detects missing keys, logs a clear warning, switches to autonomous Dry-Run mode, generates simulated signals and orders, and writes output JSON cleanly. |
| **E10** | Trailing Stop Loss | Gap-down open below the 4.8% hard stop loss price (e.g. overnight earnings crash) | Stop order triggers at the open price (with slippage impact) rather than the theoretical stop price, reflecting real market shortfall. |
| **E11** | Monte Carlo | Equity curve with negative or zero returns | Bootstrapper compounds simulated paths correctly; empirical p-value approaches $1.0$ ($0\%$ confidence), correctly flagging strategy failure. |
| **E12** | Cash Settlement | Rapid consecutive BVC sales when portfolio cash is near zero | Unsettled receivables remain locked in the $T+3$ queue; new BUY orders are constrained strictly to settled cash, preventing overdrafts. |

---

## 7. Error Handling, Failure Modes & Resilience Protocols

1. **Network Disconnection / API Outages:**
   - All REST calls (yfinance, Alpaca, RSS feeds) are wrapped in `try/except` with a maximum of 3 retries and exponential backoff (1s, 2s, 4s).
   - If live data cannot be retrieved, the system falls back to the most recent cached disk bar and sets a degraded telemetry flag.
2. **Data Integrity & NaN Propagation Prevention:**
   - Every DataFrame entering the signal or model pipeline passes through an assertion validator: `assert not df.isna().any().any()`.
   - Missing values are imputed using forward-fill followed by backward-fill, with final sanity replacement of zeroes where appropriate.
3. **Graceful Dry-Run Degradation:**
   - The live trading entrypoint (`run_live_paper.py`) requires zero external credentials to run a complete end-to-end verification cycle. In dry-run mode, all positions, executions, and stops are logged to `output/` with full fidelity.
4. **Division by Zero Protection:**
   - All ratio calculations (RVOL, Sharpe denominator, Volatility scaling, OFI) protect denominators using `np.maximum(denom, 1e-6)` or `.replace(0, np.nan).fillna(...)`.

---

## 8. Verification Plan & Quantitative Acceptance Criteria

### 8.1 Quantitative Acceptance Thresholds
The trading system implementation must satisfy the following strict quantitative criteria:

| Metric | Required Specification Threshold | Verification Method |
| :--- | :--- | :--- |
| **Net Out-of-Sample CAGR** | $\mathbf{\ge 35.0\%}$ (net of fees, slippage, taxes) | 5m–60m backtest over multi-year out-of-sample window |
| **Maximum Drawdown (MDD)** | $\mathbf{\le 20.0\%}$ | Calculated from peak-to-trough of equity series |
| **Annualized Sharpe Ratio** | $\mathbf{\ge 1.80}$ | Annualized with $M = 252 \times N_{\text{bars/day}}$ ($R_f = 2\%$) |
| **Benchmark Outperformance** | **Strategy Equity > S&P 500 Equity** | Direct comparison against SPY equity curve |
| **Monte Carlo Confidence** | $\mathbf{P(\text{CAGR} \ge 30\%) \ge 90.0\% \ (p < 0.10)}$ | 2,500 circular block-bootstrap simulations |
| **Deflated Sharpe Ratio (DSR)** | $\mathbf{\text{DSR} \ge 0.90}$ | Bailey & Lopez de Prado statistical audit ($N_{\text{trials}} \ge 20$) |
| **Automated Test Suite** | $\mathbf{100\% \text{ Pass Rate}}$ | `pytest tests/` passing with zero failures |
| **Live Dry-Run Execution** | **Successful execution cycle** | `run_live_paper.py` executes and outputs valid snapshot |

---
*Report compiled and verified by Specification Miner Agent.*
