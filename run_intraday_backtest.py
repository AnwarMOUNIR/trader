"""
Intraday Backtest Simulation CLI Entrypoint.

Executes discrete-event bar-by-bar backtest across US and Casablanca BVC equities,
enforcing Almgren-Chriss market impact, bid-ask spreads, broker fees, BVC circuit breakers,
T+3 settlement queues, and 15% IR capital gains tax.
"""

from __future__ import annotations
import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np

from src.data.loader import MarketDataLoader, UniverseConfig
from src.signals.whale_detector import WhaleDetector
from src.portfolio.allocator import PortfolioAllocator
from src.backtest.intraday_engine import IntradayBacktestEngine, BacktestResult
from src.backtest.order_book import OrderBookSimulator, AlmgrenChrissConfig
from src.backtest.bvc_adapter import BVCAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("RunIntradayBacktest")


def main():
    parser = argparse.ArgumentParser(description="Multi-Horizon Intraday Backtester")
    parser.add_argument("--interval", type=str, default="5m", choices=["5m", "15m", "30m", "60m", "1d"], help="Bar interval")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial capital in USD")
    parser.add_argument("--symbols", type=str, default="NVDA,AAPL,MSFT,ATW.BVC,BCP.BVC", help="Comma-separated symbols")
    parser.add_argument("--rebalance-freq", type=int, default=1, help="Rebalancing frequency in bars")
    parser.add_argument("--output", type=str, default="output/intraday_backtest_results.json", help="Path to output JSON")
    parser.add_argument("--no-cache", action="store_true", help="Bypass local cache")
    args = parser.parse_args()

    symbols_list = [s.strip() for s in args.symbols.split(",") if s.strip()]
    us_symbols = [s for s in symbols_list if not (".BVC" in s or "BVC" in s.upper())]
    bvc_symbols = [s for s in symbols_list if (".BVC" in s or "BVC" in s.upper())]

    print("=" * 80)
    print("      MULTI-HORIZON INTRADAY DISCRETE-EVENT BACKTESTER")
    print("=" * 80)
    print(f"Interval:            {args.interval}")
    print(f"Initial Capital:     ${args.capital:,.2f}")
    print(f"US Equities:         {us_symbols}")
    print(f"Casablanca Equities: {bvc_symbols}")
    print(f"Rebalance Freq:      Every {args.rebalance_freq} bar(s)")
    print("=" * 80)

    cfg = UniverseConfig(
        us_symbols=us_symbols or ["NVDA", "AAPL", "MSFT"],
        bvc_symbols=bvc_symbols or ["ATW.BVC", "BCP.BVC"]
    )
    loader = MarketDataLoader(cfg)
    data_dict = loader.fetch_all(use_cache=not args.no_cache)

    engine = IntradayBacktestEngine(
        universe_config=cfg,
        initial_capital=args.capital,
        rebalance_frequency=args.rebalance_freq
    )

    result: BacktestResult = engine.run(data_dict=data_dict, initial_capital=args.capital)
    metrics_dict = result.summary()

    print("\n" + "=" * 80)
    print("                 QUANTITATIVE PERFORMANCE TEARSHEET")
    print("=" * 80)
    for k, v in metrics_dict.items():
        print(f"  {k:<28}: {v}")
    print("=" * 80)

    # Save results to output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "parameters": {
                "interval": args.interval,
                "capital": args.capital,
                "rebalance_freq": args.rebalance_freq,
                "symbols": symbols_list,
            },
            "metrics": metrics_dict,
            "equity_points": len(result.equity_series),
            "trade_count": len(result.trade_log),
        }, f, indent=2)

    print(f"\n[OK] Results successfully saved to {out_path}")


if __name__ == "__main__":
    main()
