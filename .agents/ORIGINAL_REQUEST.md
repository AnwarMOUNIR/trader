# Original User Request

## 2026-08-31T15:19:55Z

An intraday algorithmic trading system operating on 5-60 minute cycles, integrating real-time news RSS crawling and NLP synthesis, guided by a multi-horizon deep neural network model trained on 30 years of daily historical data combined with recent intraday data.

Working directory: c:\Users\Anwar\Documents\trader
Integrity mode: development

## Requirements

### R1. Dual-Resolution Data Ingestion Pipeline
- Pull 30 years of daily historical price and volume data for a broad universe of US Equities using free historical data APIs/yfinance.
- Fetch high-resolution intraday data (5-minute to 60-minute intervals) for the maximum available historical limit (e.g., last 60-730 days) to support intraday training and backtesting.

### R2. RSS News Harvester and Sentiment Analyzer
- Periodically poll free finance news RSS feeds (Yahoo Finance, Google News) and public APIs for company-specific news articles.
- Implement an NLP/Sentiment Analyzer (using local models like VADER, FinBERT, or lightweight HuggingFace transformers) to synthesize news content into hourly/daily numerical sentiment scores.

### R3. Multi-Horizon Deep Neural Network
- Design and train a Deep Neural Network (e.g., LSTM, GRU, or Temporal Convolutional Network) utilizing:
  - Long-term macro features learned from the 30-year daily dataset.
  - Short-term microstructure features (intraday price-volume momentum).
  - Real-time news sentiment and volume features.
- Output high-frequency trading decisions (BUY, HOLD, SELL) for a dual US/Moroccan universe.

### R4. High-Frequency Intraday Backtester & Paper Simulator
- Build a discrete-event backtester operating at intraday frequencies (5m to 60m).
- Implement trading frictions: bid-ask spread, broker commissions, and Almgren-Chriss market impact.
- Integrate the neural network model into the live Alpaca Paper Trader connector to run on scheduled automated crons.

## Verification Plan

### Automated Tests
- Run intraday backtests comparing the neural network strategy against a simple buy-and-hold benchmark.
- Validate sentiment output aligns with expected positive/negative news signals.

## Acceptance Criteria

### Performance & Risk Metrics
- Achieve a net annualized return (CAGR) >= 35% in out-of-sample backtests on 5-60 minute cycles.
- Maximum Drawdown (MDD) <= 20%.
- Sharpe Ratio >= 1.80.
- Automated pytest suite passes 100%.
- Successfully execute a live paper trading dry-run cycle using the new neural network model and live news feeds.
