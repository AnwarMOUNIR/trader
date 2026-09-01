# Architectural Survey & Specification: Multi-Horizon Deep Neural Network, High-Frequency Intraday Backtester, & Quantitative Acceptance Engine

**Author**: `survey_explorer_2` (Teamwork Explorer)  
**Parent Orchestrator**: `orchestrator_1` (Conversation ID: `221d5d39-4a0b-46d8-8d77-134da45be38f`)  
**Target Milestone**: Architecture Specification & Verification Protocol  
**Scope**: Requirements R3 (Multi-Horizon DNN), R4 (HF Intraday Backtester & Paper Simulator), and Acceptance Verification Strategy  

---

## 1. Executive Summary & Problem Formulation

The objective is to design and specify an institutional-grade, multi-horizon algorithmic trading system operating at high-frequency intraday cycles ($5\text{m} \le \Delta t \le 60\text{m}$) across a dual universe: **US Equities** (liquid growth/mega-caps) and **Moroccan Equities** (Casablanca Stock Exchange / BVC).

```
+--------------------------------------------------------------------------------------------------+
|                                    INTRADAY QUANTITATIVE ENGINE                                  |
+--------------------------------------------------------------------------------------------------+
|                                                                                                  |
|   +--------------------------+   +--------------------------+   +----------------------------+   |
|   | 30-Year Macro History    |   | 5m-60m Intraday Stream   |   | News RSS & NLP Sentiment   |   |
|   | (SPY/QQQ, Trend, Vol)    |   | (VWAP, OFI, RVOL, Range) |   | (FinBERT/VADER Polarity)   |   |
|   +------------+-------------+   +------------+-------------+   +--------------+-------------+   |
|                |                              |                                |                 |
|                v                              v                                v                 |
|   +------------------------------------------------------------------------------------------+   |
|   |                 R3: Multi-Horizon Deep Neural Network (TCN / TFT / Hierarchical)          |   |
|   |                 - Temporal Fusion / Cross-Attention Mechanism                            |   |
|   |                 - Target Labeling: Triple-Barrier Method + Volatility Normalization      |   |
|   |                 - Output: Calibrated Directional Probabilities + Expected Return/Risk    |   |
|   +---------------------------------------------+--------------------------------------------+   |
|                                                 |                                                |
|                                                 v                                                |
|   +------------------------------------------------------------------------------------------+   |
|   |                 R4: Intraday Discrete-Event Backtester & Microstructure Simulator         |   |
|   |                 - Almgren-Chriss Market Impact (Linear Temporary + Square-Root Permanent)|   |
|   |                 - Bid-Ask Spread Friction & Tiered Commissions                           |   |
|   |                 - Casablanca BVC Mechanics (+/-10% Limit, T+3 Clearing, 15% IR Tax)      |   |
|   |                 - Alpaca Paper Trading Connector (REST/WS Stream, Bracket Orders)         |   |
|   +---------------------------------------------+--------------------------------------------+   |
|                                                 |                                                |
|                                                 v                                                |
|   +------------------------------------------------------------------------------------------+   |
|   |                 Quantitative Acceptance & Anti-Overfitting Protocol                      |   |
|   |                 - Combinatorial Purged Cross-Validation (CPCV) + Embargoing              |   |
|   |                 - Deflated Sharpe Ratio (DSR > 0.95), Monte Carlo Confidence >= 90%      |   |
|   |                 - Target: CAGR >= 35%, MDD <= 20%, Sharpe >= 1.80, 100% Pytest Pass      |   |
|   +------------------------------------------------------------------------------------------+   |
+--------------------------------------------------------------------------------------------------+
```

---

## 2. Requirement 3: Multi-Horizon Deep Neural Network (R3)

### 2.1 Architecture Design & Multi-Horizon Fusion

Financial markets exhibit strong multi-scale temporal dynamics:
1. **Long-Term Macro Dynamics ($T_{macro} \approx 30 \text{ years / } 60\text{--}252 \text{ days lookback}$)**: Macro regimes (bull/bear/stagnation), secular volatility clustering, 200-day moving averages, 52-week highs, and inter-market correlations (SPY, QQQ, VIX, 10-Year Treasury Yields).
2. **Short-Term Microstructure Dynamics ($T_{micro} \approx 20\text{--}78 \text{ intraday bars, } 5\text{m--}60\text{m}$)**: Intraday price-volume momentum, volume-weighted average price (VWAP) deviations, Order Flow Imbalance (OFI), Relative Volume (RVOL), and Bar Spread Compression.
3. **Event-Driven Sentiment Dynamics ($T_{sent} \approx \text{real-time to 72 hours}$)**: Real-time news sentiment polarity scores $[-1.0, +1.0]$, confidence scores, news burst frequencies, and exponential decay models.

#### Proposed Neural Architecture: **Dual-Branch Temporal Convolutional Transformer (TCT)**

```
Macro Feature Sequence [B x 60 x D_macro]        Intraday Bar Sequence [B x 39 x D_micro]    News Sentiment Vector [B x D_sent]
                 |                                                |                                         |
                 v                                                v                                         v
   +---------------------------+                    +---------------------------+             +---------------------------+
   | Dilated Causal TCN Branch |                    | Dilated Causal TCN Branch |             | Linear Projection +       |
   | (Receptive Field: 60 Days)|                    | (Receptive Field: 1-Day)  |             | LayerNorm & Swish Gate    |
   | Dilation: [1, 2, 4, 8]    |                    | Dilation: [1, 2, 4]       |             |                           |
   +-------------+-------------+                    +-------------+-------------+             +-------------+-------------+
                 |                                                |                                         |
                 v                                                v                                         |
   +--------------------------------------------------------------------------------+                       |
   |                   Cross-Temporal Multi-Head Attention Fusion                   |                       |
   |   (Query: Intraday Hidden State | Key/Value: Macro Regime Context Hidden State)|                       |
   +----------------------------------------+---------------------------------------+                       |
                                            |                                                               |
                                            +-----------------------+---------------------------------------+
                                                                    |
                                                                    v
                                                     +-------------------------------+
                                                     | Gated Residual Network (GRN)  |
                                                     | + LayerNorm & Dropout (0.20)  |
                                                     +---------------+---------------+
                                                                     |
                                           +-------------------------+-------------------------+
                                           |                         |                         |
                                           v                         v                         v
                              +-------------------------+  +-------------------+  +------------------------+
                              | Direction Classification|  | Expected Return   |  | Adaptive Volatility /  |
                              | (3-class: SELL/HOLD/BUY)|  | Continuous (r_hat)|  | Downside Risk (sigma)  |
                              +-------------------------+  +-------------------+  +------------------------+
```

#### Mathematical Specification:
1. **Dilated Causal Convolutions**:
   $$\mathbf{y}_t = \sum_{k=0}^{K-1} f_k \cdot \mathbf{x}_{t - d \cdot k}$$
   where $d \in \{1, 2, 4, 8\}$ is the dilation factor and $K=3$ is kernel size, guaranteeing zero lookahead bias ($t$ depends strictly on $t' \le t$).
2. **Cross-Attention Fusion**:
   $$\mathbf{Q} = \mathbf{H}_{micro} \mathbf{W}_Q, \quad \mathbf{K} = \mathbf{H}_{macro} \mathbf{W}_K, \quad \mathbf{V} = \mathbf{H}_{macro} \mathbf{W}_V$$
   $$\mathbf{Attn}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
3. **Sentiment Conditioning**:
   $$\mathbf{e}_{fusion} = \text{GRN}\left(\left[\mathbf{H}_{cross}; \mathbf{H}_{micro}; \mathbf{e}_{sent}\right]\right)$$

---

### 2.2 Feature Pipeline & Matrix Engineering

The input feature pipeline constructs three synchronized tensors:

| Feature Domain | Feature Name | Computation Formula / Description | Update Frequency | Lookahead Protection |
| :--- | :--- | :--- | :--- | :--- |
| **Macro (Daily)** | `SMA200_Ratio` | $\frac{\text{Close}_t}{\text{SMA}_{200}(\text{Close}_t)}$ | Daily | Lagged 1 bar ($t-1$) |
| **Macro (Daily)** | `Macro_Trend_Regime` | $\mathbb{I}(\text{SPY}_{\text{Close}} \ge \text{SMA}_{200}(\text{SPY}))$ | Daily | Lagged 1 bar ($t-1$) |
| **Macro (Daily)** | `Vol_20D` | $\text{Std}(\text{Returns}_{t-20:t}) \cdot \sqrt{252}$ | Daily | Past rolling window only |
| **Macro (Daily)** | `Mom_12M` | $\frac{\text{Close}_{t-1}}{\text{Close}_{t-252}} - 1.0$ | Daily | Past rolling window only |
| **Macro (Daily)** | `Ratio_52W_High` | $\frac{\text{Close}_{t-1}}{\max_{k \in [1, 252]} \text{High}_{t-k}}$ | Daily | Past rolling window only |
| **Micro (Intraday)** | `Log_Return_k` | $\ln(\text{Close}_\tau / \text{Close}_{\tau-1})$ | 5m/15m/60m | Current closed bar only |
| **Micro (Intraday)** | `VWAP_Deviation` | $\frac{\text{Close}_\tau - \text{VWAP}_\tau}{\text{VWAP}_\tau}$ | 5m/15m/60m | Intraday cumulative from open |
| **Micro (Intraday)** | `RVOL_Intraday` | $\frac{\text{Volume}_\tau}{\text{Mean}(\text{Volume}_{\tau, \text{historical days}})}$ | 5m/15m/60m | Past 20-day historical time slot |
| **Micro (Intraday)** | `Vol_ZScore` | $\frac{\text{Volume}_\tau - \mu_V(\tau)}{\sigma_V(\tau)}$ | 5m/15m/60m | 30-bar rolling window |
| **Micro (Intraday)** | `Bar_Spread_Ratio`| $\frac{\text{High}_\tau - \text{Low}_\tau}{\text{SMA}_{20}(\text{High} - \text{Low})}$ | 5m/15m/60m | Past 20 bars |
| **Micro (Intraday)** | `Close_Location` | $\frac{\text{Close}_\tau - \text{Low}_\tau}{\text{High}_\tau - \text{Low}_\tau + \epsilon}$ | 5m/15m/60m | Intra-bar geometry |
| **Micro (Intraday)** | `OFI_Norm` | $\frac{\text{Volume}_\tau \cdot (\text{Close}_\tau - \text{Open}_\tau)}{(\text{High}_\tau - \text{Low}_\tau + \epsilon) \cdot \text{MarketCap}}$ | 5m/15m/60m | Order flow proxy |
| **Sentiment** | `Sentiment_Score` | $\text{FinBERT Polarity} \in [-1.0, +1.0]$ | Discrete Event | Timestamped prior to bar close |
| **Sentiment** | `Sentiment_Decay` | $\sum_{i} S_i \cdot \exp(-\lambda (t - t_i))$ with $\lambda = \frac{\ln(2)}{6 \text{ hours}}$ | Continuous | Exponential half-life |
| **Sentiment** | `News_Volume_Burst`| Count of articles in last 60 minutes vs 7-day average | 5m/15m/60m | Past rolling window |

---

### 2.3 Target Labeling Protocol: De Prado's Triple-Barrier Method

To avoid the pitfalls of fixed-horizon binary labeling (which ignores path dependency, drawdowns, and real-world stop outs), we deploy **Marcos Lopez de Prado's Triple-Barrier Method** combined with **Meta-Labeling**:

```
        Price Path
           ^
Upper Barrier (+1.5 * sigma_t) --------------------------------+ [Touch = BUY SUCCESS (+1)]
                                   / \        /\
                                  /   \  /\  /  \
                                 /     \/  \/    \
Current Bar (t) --------------->+-------------------------------- Time Barrier (T_max = 12 bars) [Expire = HOLD (0)]
                                 \               /
                                  \             /
Lower Barrier (-1.0 * sigma_t) ----\-----------+---------------- [Touch = STOP HIT / SELL (-1)]
```

#### Triple-Barrier Rules:
1. **Upper Horizontal Barrier**: $P_{upper} = P_t \cdot (1 + k_u \cdot \sigma_t)$, where $\sigma_t$ is the 20-bar intraday rolling volatility and $k_u = 1.5$.
2. **Lower Horizontal Barrier**: $P_{lower} = P_t \cdot (1 - k_d \cdot \sigma_t)$, where $k_d = 1.0$ (asymmetric risk-reward $1.5:1$).
3. **Vertical Time Barrier**: $T_{max} = t + H$, where $H = 12$ bars (e.g. 1 hour for 5m bars, or 3 hours for 15m bars).
4. **Label Mapping**:
   $$y_t = \begin{cases} +1 & \text{if path touches } P_{upper} \text{ first (BUY / Outperform)} \\ -1 & \text{if path touches } P_{lower} \text{ first (SELL / Stop-out)} \\ 0 & \text{if path touches } T_{max} \text{ before either horizontal barrier (HOLD / Stagnation)} \end{cases}$$

---

### 2.4 Loss Function & Training Protocol

1. **Focal Multi-Class Loss**: To resolve the heavy class imbalance (HOLD $\approx 70\%$, BUY $\approx 15\%$, SELL $\approx 15\%$):
   $$\mathcal{L}_{cls} = -\sum_{c \in \{-1, 0, +1\}} \alpha_c (1 - p_c)^\gamma \log(p_c), \quad \gamma = 2.0$$
2. **Differential Sharpe Regularization**:
   $$\mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda_{reg} \|\Theta\|_2^2 - \lambda_{sharpe} \cdot \widehat{\text{Sharpe}}(\mathbf{w}_t \cdot \mathbf{r}_{t+1})$$
3. **Purged Combinatorial Cross-Validation (CPCV)**:
   - Partition historical dataset into $N=6$ groups.
   - For each combination of $k=2$ test groups:
     * **Purge**: Remove training samples whose label horizons overlap with the test set boundaries.
     * **Embargo**: Drop an additional 2% of samples immediately following test sets to prevent auto-regressive leakage.
4. **Optimization Routine**:
   - Optimizer: AdamW ($\text{lr} = 3 \times 10^{-4}$, weight decay $= 10^{-4}$).
   - Learning Rate Scheduler: Cosine Annealing with Warm Restarts.
   - Early Stopping: Monitored on Purged Validation Sharpe Ratio (patience = 8 epochs).

---

### 2.5 BUY / HOLD / SELL Policy Execution

The continuous model output probabilities $\mathbf{p} = [p_{\text{sell}}, p_{\text{hold}}, p_{\text{buy}}]$ are translated into deterministic portfolio decisions:

```python
# Policy Decision Logic
if p_buy > 0.60 and p_buy > (p_sell + 0.20) and macro_regime == 1:
    action = "BUY"
    conviction = (p_buy - 0.50) / 0.50  # [0.20, 1.00]
    target_weight = min(max_single_weight, base_weight * conviction)
elif p_sell > 0.55 or (holding_pnl < -0.045) or (current_price < trailing_stop_price):
    action = "SELL"
    target_weight = 0.0
else:
    action = "HOLD"
    target_weight = current_weight
```

---

## 3. Requirement 4: High-Frequency Intraday Backtester & Microstructure Paper Simulator (R4)

### 3.1 Discrete-Event Intraday Engine Architecture

The backtester operates as a high-speed, discrete-event state machine iterating through time ticks $\tau_k \in \{5\text{m}, 15\text{m}, 30\text{m}, 60\text{m}\}$:

```
[Event Generator] -> [Bar Update] -> [Signal Inference] -> [Allocation Sizer] -> [Order Routing] -> [Almgren-Chriss Fill] -> [Portfolio Accounting]
```

#### Core Components & State Variables:
- **Event Clock**: Synchronized monotonic timestamp iterator.
- **Portfolio State**:
  * Cash balance $C_\tau \in \mathbb{R}$
  * Share holdings $\mathbf{h}_\tau \in \mathbb{R}^K$
  * Position Entry Prices $\mathbf{p}_{\text{entry}} \in \mathbb{R}^K$
  * High-Water Mark Peak Prices $\mathbf{p}_{\text{peak}} \in \mathbb{R}^K$
  * Pending Receivables Queue (Casablanca BVC $T+3$ Settlement deque)
- **Order Types Supported**:
  * `MARKET`: Immediate fill with spread and Almgren-Chriss impact.
  * `LIMIT`: Fill conditional on $P_{\text{low}} \le P_{\text{limit}} \le P_{\text{high}}$.
  * `BRACKET`: Market entry coupled with atomic Take-Profit and Stop-Loss triggers.

---

### 3.2 Realistic Execution Frictions & Almgren-Chriss Impact

#### Almgren-Chriss Market Impact Formulation:
When an order of $q$ shares is executed over a bar of total volume $V_\tau$ and price $P_{\tau}$, the execution price is modeled as:

$$P_{\text{fill}}^{\text{BUY}} = P_{\tau}^{\text{close}} \cdot \left(1 + \frac{s}{2} + \eta \cdot \left(\frac{q}{V_\tau}\right)^\alpha + \gamma \cdot \left(\frac{q}{V_\tau}\right)\right)$$

$$P_{\text{fill}}^{\text{SELL}} = P_{\tau}^{\text{close}} \cdot \left(1 - \frac{s}{2} - \eta \cdot \left(\frac{q}{V_\tau}\right)^\alpha - \gamma \cdot \left(\frac{q}{V_\tau}\right)\right)$$

Where:
- $s$: Half bid-ask spread ($2\text{ bps}$ for US large-caps, $15\text{ bps}$ for Casablanca BVC).
- $\eta$: Temporary market impact coefficient ($0.03\text{ US}, 0.18\text{ BVC}$).
- $\gamma$: Permanent market impact coefficient ($0.01\text{ US}, 0.10\text{ BVC}$).
- $\alpha$: Power exponent ($\alpha = 1.0$ linear or $\alpha = 0.5$ square-root law of Kyle/Almgren-Chriss).
- Maximum Participation Rate Cap: $\frac{q}{V_\tau} \le 0.25$ (orders exceeding 25% of bar volume are split or rejected).

#### Comprehensive Friction Table:

| Market | Half-Spread ($s/2$) | Broker Commission | Exchange / SEC Fees | Market Impact ($\eta + \gamma$) | Tax Withholding | Settlement Window |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **US Equities (Alpaca)** | 2.0 bps (0.02%) | $0.00 / share (or $0.0035/sh) | 0.5 bps ($0.0000278 SEC) | $\sim 4.0 \text{ bps}$ | 0.0% (US Domestic / W8-BEN handled) | $T+1$ (instant for margin) |
| **Casablanca BVC** | 15.0 bps (0.15%) | 25.0 bps (0.25%) | 5.0 bps AMMC | $\sim 28.0 \text{ bps}$ | 15.0% IR on net capital gains | $T+3$ delayed clearing |

---

### 3.3 Alpaca Paper Trader API Integration

The live execution bridge integrates directly with Alpaca's Paper Trading environment (`https://paper-api.alpaca.markets`):

1. **Authentication & Session Management**:
   - Reads `APCA_API_KEY_ID` and `APCA_API_SECRET_KEY` from `.env`.
   - Supports both `alpaca-py` (v0.40+) and robust REST/HTTP fallbacks.
2. **Streaming & Intraday Bar Ingestion**:
   - Pulls real-time 5m/15m/60m IEX bars via `StockHistoricalDataClient`.
   - Local fallback to `yfinance` multi-threading when API rate limits occur.
3. **Bracket Order Execution Flow**:
   - Submits `MarketOrderRequest` with nested `StopLossRequest(stop_price=round(P * 0.952, 2))` and `TakeProfitRequest(limit_price=round(P * 1.15, 2))`.
   - Server-side execution guarantees stop execution even during local disconnects.
4. **Idempotency & Rebalance Deadband**:
   - Minimum trade deadband: $\Delta W_{\text{threshold}} = 4.0\%$ or $\$2,000$ to eliminate unprofitable churning.

---

### 3.4 Automated Scheduler & Dry-Run Cycle Execution

1. **Scheduling Cadence**:
   - Operates during market hours (09:30 - 16:00 EST for US, 09:30 - 15:30 GMT+1 for Casablanca).
   - Frequency: Every 5, 15, or 60 minutes via native cron or Windows Task Scheduler (`schtasks.exe`).
2. **CLI Entrypoint & Dry-Run Flags**:
   ```bash
   # Run a single dry-run cycle (no live broker execution)
   python run_live_paper.py --mode dry-run --interval 15m --universe us_growth
   
   # Run live paper trading cycle with Alpaca connection
   python run_live_paper.py --mode paper --interval 15m
   ```
3. **Execution State Persistence**:
   - Saves timestamped execution logs, order receipts, and allocation states to `output/paper_execution_state.json`.

---

## 4. Quantitative Verification & Acceptance Strategy

To guarantee that the strategy achieves the acceptance criteria (**CAGR $\ge 35\%$**, **MDD $\le 20\%$**, **Sharpe $\ge 1.80$**) without lookahead bias, data leakage, or overfitting:

### 4.1 Anti-Lookahead Bias & Data Integrity Protocol

```
Timeline:              t-2          t-1           t (Current Bar)        t+1
Data Available:  [== Past Bar ==] [== Past Bar ==] | <- Decision Point
Execution Price:                                   |                     [== Next Bar Open/VWAP ==]
```

1. **Strict Bar Lagging**: All feature calculations at bar index $t$ are strictly computed using information up to $t-1$. The decision made at $t$ is executed at the Open/VWAP of bar $t$.
2. **Zero In-Sample Normalization**: Z-scores, standard deviations, and scalers are fitted **only** on the training window and transformed on validation/out-of-sample slices without peeking forward.
3. **Point-in-Time Sentiment Stamping**: News articles are matched strictly by `published_timestamp < bar_start_timestamp`.

---

### 4.2 Statistical Significance & Anti-Overfitting Protocol

```
CPCV Split 1: [Train Grp 1,2,3,4] -- Purge -- [Test Grp 5,6] -> CAGR_1, Sharpe_1
CPCV Split 2: [Train Grp 1,2,5,6] -- Purge -- [Test Grp 3,4] -> CAGR_2, Sharpe_2
...
CPCV Split 15: Combinatorial out-of-sample paths evaluated
```

1. **Combinatorial Purged Cross-Validation (CPCV)**:
   - 15 combinatorial splits across $N=6, k=2$.
   - **Acceptance Gate**: $\ge 90\%$ of all OOS paths must achieve positive CAGR, with mean $\text{CAGR} \ge 35\%$.
2. **Deflated Sharpe Ratio (DSR)**:
   $$\text{DSR} = \Phi\left(\frac{(\widehat{\text{SR}} - \text{SR}_0) \sqrt{N - 1}}{\sqrt{1 - \hat{\gamma}_3 \widehat{\text{SR}} + \frac{\hat{\gamma}_4 - 1}{4} \widehat{\text{SR}}^2}}\right) \ge 0.95$$
   Corrects for non-normality (skewness $\gamma_3$, kurtosis $\gamma_4$) and number of strategy trials ($N_{trials} \ge 50$).
3. **Block-Bootstrap Monte Carlo (2,500 iterations)**:
   - 10-bar stationary circular block bootstrapping.
   - **Acceptance Gate**: Empirical $p$-value $P(\text{CAGR} < 35\%) < 0.10$ (Confidence $\ge 90\%$).

---

### 4.3 100% Pytest Test Suite Architecture

The automated test suite is structured into 5 opaque test tiers:

| Tier | Focus | Test File Target | Expected Pass Criteria |
| :--- | :--- | :--- | :--- |
| **Tier 1: Unit** | Data pipeline, feature shapes, no NaNs, zero lookahead | `tests/test_features.py` | 100% pass, runtime < 5s |
| **Tier 2: Model** | Torch DNN forward pass, loss gradient flow, barrier labels | `tests/test_neural_model.py` | 100% pass, zero NaN loss |
| **Tier 3: Simulator**| Almgren-Chriss impact, spread, BVC T+3 settlement, 15% tax | `tests/test_execution.py` | Exact matching vs analytical formula |
| **Tier 4: Backtest** | 5m-60m discrete engine, metric calculations, benchmark | `tests/test_backtest_engine.py` | CAGR >= 35%, MDD <= 20%, Sharpe >= 1.80 |
| **Tier 5: Paper/Live**| Alpaca API mock, bracket orders, dry-run CLI, cron cycle | `tests/test_alpaca_live.py` | 100% pass in simulated environment |

---

## 5. Interface Contracts & Component Specifications

### 5.1 R3 Multi-Horizon DNN Interface Contract (`src/models/multi_horizon_dnn.py`)

```python
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn

@dataclass
class ModelConfig:
    macro_input_dim: int = 12
    micro_input_dim: int = 16
    sent_input_dim: int = 4
    macro_seq_len: int = 60
    micro_seq_len: int = 39
    hidden_dim: int = 64
    num_heads: int = 4
    dropout: float = 0.20
    learning_rate: float = 3e-4

@dataclass
class PredictionOutput:
    direction_logits: torch.Tensor       # [B, 3] (SELL=0, HOLD=1, BUY=2)
    direction_probs: torch.Tensor        # [B, 3] (Softmax probabilities)
    expected_return: torch.Tensor        # [B, 1] (Predicted forward return)
    predicted_volatility: torch.Tensor   # [B, 1] (Predicted forward sigma)
    conviction_score: torch.Tensor       # [B, 1] (Scaled confidence [0.0, 1.0])

class MultiHorizonTCT(nn.Module):
    """Dual-Branch Temporal Convolutional Transformer with Sentiment Cross-Attention."""
    def __init__(self, config: ModelConfig): ...
    def forward(
        self,
        macro_seq: torch.Tensor,   # [B, 60, macro_dim]
        micro_seq: torch.Tensor,   # [B, 39, micro_dim]
        sent_vec: torch.Tensor     # [B, sent_dim]
    ) -> PredictionOutput: ...
```

---

### 5.2 R4 Intraday Backtester Interface Contract (`src/backtest/intraday_engine.py`)

```python
@dataclass
class IntradayBacktestConfig:
    bar_interval: str = "15m"             # '5m', '15m', '30m', '60m'
    initial_capital: float = 100_000.0
    target_annual_vol: float = 0.32
    max_single_weight: float = 0.34
    rebalance_deadband_usd: float = 2000.0
    stop_loss_pct: float = 0.048          # -4.8% stop loss
    profit_target_pct: float = 0.150       # +15.0% take profit

@dataclass
class IntradayBacktestResult:
    equity_series: pd.Series
    trade_log: pd.DataFrame
    metrics: PerformanceMetrics
    cpcv_result: CPCVResult
    monte_carlo_result: MonteCarloResult
    benchmark_equity: pd.Series

class IntradayBacktestEngine:
    def __init__(self, config: IntradayBacktestConfig): ...
    def run_simulation(
        self,
        intraday_data: Dict[str, pd.DataFrame],
        daily_macro_data: Dict[str, pd.DataFrame],
        sentiment_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> IntradayBacktestResult: ...
```

---

## 6. Worker Implementation Plan & Decomposition

To implement M3 (Multi-Horizon DNN) and M4 (Intraday Backtester & Paper Simulator), the downstream sub-orchestrator and workers should follow this execution sequence:

1. **Step 1: Intraday Data Ingestion & Triple-Barrier Labeling (`src/data/intraday_loader.py` & `src/data/labeling.py`)**:
   - Implement 5m/15m/60m bar data processing.
   - Implement de Prado's Triple Barrier Method with rolling ATR volatility bands.
2. **Step 2: PyTorch Multi-Horizon Model Implementation (`src/models/multi_horizon_dnn.py`)**:
   - Build Dilated Causal TCN branches + Cross-Attention fusion + Gated Residual Networks.
   - Implement Focal Loss + Sharpe regularized loss.
3. **Step 3: Discrete-Event Intraday Backtester Engine (`src/backtest/intraday_engine.py`)**:
   - Integrate Almgren-Chriss impact, dynamic bid-ask spread, Casablanca T+3 queue, and Moroccan 15% IR tax.
4. **Step 4: Alpaca Paper Trading Real-Time Bridge (`src/execution/alpaca_trader.py`)**:
   - Update Alpaca Paper Trader to support 15m cycles, bracket orders, and dry-run execution mode.
5. **Step 5: Automated Test Suite & Statistical Acceptance Gate (`tests/test_*.py`)**:
   - Complete 100% pytest test suite covering all modules.
   - Run full 6-year backtest and CPCV/Monte Carlo validation to confirm CAGR >= 35%, MDD <= 20%, Sharpe >= 1.80.

---
