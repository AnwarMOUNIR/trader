"""
Live Paper Trading & Autonomous Dry-Run Entrypoint.

Executes the Multi-Horizon Trading strategy on Alpaca's Paper Trading API
or autonomous zero-credential dry-run engine, saving real-time telemetry snapshots.
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path
from src.execution.alpaca_client import AlpacaPaperTrader
from src.execution.dry_run_simulator import DryRunSimulator
from src.execution.scheduler import ExecutionScheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("RunLivePaper")


def main():
    parser = argparse.ArgumentParser(description="Multi-Horizon Live Paper & Dry-Run Trader")
    parser.add_argument("--api-key", type=str, default=os.getenv("APCA_API_KEY_ID"), help="Alpaca API Key ID")
    parser.add_argument("--secret-key", type=str, default=os.getenv("APCA_API_SECRET_KEY"), help="Alpaca Secret Key")
    parser.add_argument("--mode", type=str, default="dry-run", choices=["dry-run", "live", "scheduled"], help="Execution mode")
    parser.add_argument("--interval", type=int, default=5, help="Scheduler interval in minutes")
    parser.add_argument("--iterations", type=int, default=1, help="Number of scheduled iterations")
    parser.add_argument("--capital", type=float, default=100000.0, help="Initial simulation capital")
    args = parser.parse_args()

    print("=" * 80)
    print("      MULTI-HORIZON LIVE PAPER & DRY-RUN TRADING ENGINE")
    print("=" * 80)
    print(f"Mode:            {args.mode.upper()}")
    print(f"Interval:        {args.interval}m")
    print(f"Iterations:      {args.iterations}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print("=" * 80)

    if args.mode == "scheduled":
        sim = DryRunSimulator(initial_capital=args.capital)
        trader = AlpacaPaperTrader(api_key=args.api_key, secret_key=args.secret_key)
        scheduler = ExecutionScheduler(
            dry_run_simulator=sim,
            alpaca_trader=trader,
            interval_minutes=args.interval,
            force_market_open=True
        )
        reports = scheduler.run_iterations(max_iterations=args.iterations, sleep_seconds=1)
        print(f"\n[OK] Completed {len(reports)} scheduled execution iteration(s).")
    elif args.mode == "live" and args.api_key and args.secret_key:
        trader = AlpacaPaperTrader(api_key=args.api_key, secret_key=args.secret_key)
        result = trader.run_daily_cycle()
        print("\n" + "=" * 80)
        print("                      LIVE PAPER EXECUTION SUMMARY")
        print("=" * 80)
        print(f"Timestamp:       {result.get('timestamp')}")
        print(f"Account Equity:  ${result.get('equity', 0):,.2f}")
        print(f"Settled Cash:    ${result.get('cash', 0):,.2f}")
        print(f"Active Targets:  {result.get('target_allocation')}")
        print("Executed Orders:")
        for order in result.get("executed_orders", []):
            print(f"  • {order.get('side')} {order.get('qty')} {order.get('symbol')} @ ${order.get('filled_price', 0):,.2f} [Status: {order.get('status')}]")
        print("=" * 80)
    else:
        # Default Autonomous Dry-Run mode
        sim = DryRunSimulator(initial_capital=args.capital)
        report = sim.run_cycle()
        print("\n" + "=" * 80)
        print("                   AUTONOMOUS DRY-RUN EXECUTION REPORT")
        print("=" * 80)
        print(f"Cycle ID:        {report.cycle_id}")
        print(f"Timestamp:       {report.timestamp}")
        print(f"Universe Size:   {report.universe_size} assets")
        print(f"Account Equity:  ${report.account_equity:,.2f}")
        print(f"Settled Cash:    ${report.settled_cash:,.2f}")
        print(f"Active Targets:  {report.target_weights}")
        print("Executed Orders:")
        for order in report.executed_orders:
            print(f"  • {order.get('side')} {order.get('qty')} {order.get('symbol')} @ ${order.get('filled_price', 0):,.2f} (SL: ${order.get('stop_price')})")
        if not report.executed_orders:
            print("  • No new rebalance orders required this cycle.")
        print(f"Status:          {report.status}")
        print("=" * 80)
        print("[OK] Telemetry saved to output/current_signals_snapshot.json and output/live_execution_log.json")


if __name__ == "__main__":
    main()
