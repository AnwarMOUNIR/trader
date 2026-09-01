"""
Evaluates Out-of-Sample Intraday Backtests across intervals and friction models.
"""
from __future__ import annotations
import sys
import os
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import numpy as np
import pandas as pd

from src.data.loader import MarketDataLoader, UniverseConfig
from src.signals.whale_detector import WhaleDetector
from src.portfolio.allocator import PortfolioAllocator
from src.backtest.intraday_engine import IntradayBacktestEngine, BacktestResult
from src.backtest.order_book import OrderBookSimulator, AlmgrenChrissConfig
from src.backtest.bvc_adapter import BVCAdapter
from src.backtest.metrics import calculate_metrics

def evaluate_all():
    cfg = UniverseConfig(
        us_symbols=["NVDA", "AAPL", "MSFT", "AMZN", "TSLA", "META", "GOOGL", "AVGO"],
        bvc_symbols=["ATW.BVC", "BCP.BVC", "IAM.BVC", "BOA.BVC", "CSR.BVC"],
        benchmark_symbol="SPY"
    )
    loader = MarketDataLoader(cfg)
    data_dict = loader.fetch_all(use_cache=True)

    oos_data = {}
    for sym, df in data_dict.items():
        df_copy = df.copy()
        col_map = {c: c.capitalize() for c in df_copy.columns if c.lower() in ["open", "high", "low", "close", "volume", "vwap"]}
        df_copy = df_copy.rename(columns=col_map)
        oos_df = df_copy[(df_copy.index >= "2024-01-01") & (df_copy.index <= "2025-12-31")]
        if not oos_df.empty:
            oos_data[sym] = oos_df

    print(f"Total OOS assets: {len(oos_data)}")

    # 1. Base Backtest
    engine_base = IntradayBacktestEngine(universe_config=cfg, initial_capital=100_000.0, rebalance_frequency=1)
    res_base = engine_base.run(data_dict=oos_data, initial_capital=100_000.0)
    m_base = res_base.summary()

    # 2. Zero Friction
    zero_frictions = OrderBookSimulator(AlmgrenChrissConfig(
        us_linear_impact_eta=0.0, us_permanent_impact_gamma=0.0, us_brokerage_bps=0.0, us_half_spread_bps=0.0,
        bvc_linear_impact_eta=0.0, bvc_permanent_impact_gamma=0.0, bvc_brokerage_bps=0.0, bvc_half_spread_bps=0.0,
        moroccan_ir_tax_rate=0.0, settlement_days_bvc=0
    ))
    engine_zero = IntradayBacktestEngine(universe_config=cfg, execution_model=zero_frictions, initial_capital=100_000.0)
    res_zero = engine_zero.run(data_dict=oos_data, initial_capital=100_000.0)
    m_zero = res_zero.summary()

    # 3. High Friction Stress Test
    high_frictions = OrderBookSimulator(AlmgrenChrissConfig(
        us_linear_impact_eta=0.10, us_permanent_impact_gamma=0.04, us_brokerage_bps=10.0, us_half_spread_bps=5.0,
        bvc_linear_impact_eta=0.50, bvc_permanent_impact_gamma=0.30, bvc_brokerage_bps=50.0, bvc_half_spread_bps=30.0,
        moroccan_ir_tax_rate=0.15, settlement_days_bvc=3
    ))
    engine_high = IntradayBacktestEngine(universe_config=cfg, execution_model=high_frictions, initial_capital=100_000.0)
    res_high = engine_high.run(data_dict=oos_data, initial_capital=100_000.0)
    m_high = res_high.summary()

    print("=" * 85)
    print(f"{'Metric':<28} | {'Zero Friction':<16} | {'Standard Base':<16} | {'High Friction':<16}")
    print("=" * 85)
    keys_to_print = [
        "CAGR (%)", "Total Return (%)", "Sharpe Ratio", "Sortino Ratio",
        "Calmar Ratio", "Max Drawdown (%)", "Max Drawdown (Days)",
        "Win Rate (%)", "Profit Factor", "Total Trades",
        "Total Commission ($)", "Total Slippage ($)", "Total Tax Paid ($)"
    ]
    for k in keys_to_print:
        v0 = m_zero.get(k, 0)
        v1 = m_base.get(k, 0)
        v2 = m_high.get(k, 0)
        print(f"{k:<28} | {str(v0):<16} | {str(v1):<16} | {str(v2):<16}")
    print("=" * 85)

    # Acceptance Criteria Check
    cagr = m_base["CAGR (%)"]
    mdd = m_base["Max Drawdown (%)"]
    sharpe = m_base["Sharpe Ratio"]

    print("\n" + "=" * 60)
    print("ACCEPTANCE CRITERIA VERIFICATION (OUT-OF-SAMPLE 2024-2025)")
    print("=" * 60)
    print(f"1. CAGR >= 35.0%:   {cagr:.2f}%  ->  {'PASS' if cagr >= 35.0 else 'FAIL'}")
    print(f"2. MDD <= 20.0%:    {mdd:.2f}%   ->  {'PASS' if mdd <= 20.0 else 'FAIL'}")
    print(f"3. Sharpe >= 1.80:  {sharpe:.2f}   ->  {'PASS' if sharpe >= 1.80 else 'FAIL'}")
    print("=" * 60)

if __name__ == "__main__":
    evaluate_all()
