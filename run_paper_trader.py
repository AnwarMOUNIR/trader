"""Command-line runner and entry point for the Alpaca Paper Trading Daemon."""
import argparse
import sys
from pathlib import Path

from market_research.paper_trader import AlpacaPaperTrader


def main() -> None:
    parser = argparse.ArgumentParser(description="Alpaca Paper Trading Runner")
    parser.add_argument("--status", action="store_true", help="Print live account status, market clock, and signal preview without trading")
    parser.add_argument("--once", action="store_true", help="Execute one evaluation/rebalance cycle and exit")
    parser.add_argument("--daemon", action="store_true", help="Run continuously in the background whenever the system is running")
    parser.add_argument("--interval", type=int, default=15, help="Daemon polling interval in minutes during market hours (default: 15)")
    args = parser.parse_args()

    trader = AlpacaPaperTrader()

    if args.status or (not args.once and not args.daemon):
        print("=== ALPACA PAPER TRADING STATUS ===")
        res = trader.evaluate_and_rebalance(dry_run=True)
        print("\nSummary:")
        print(f"  Account Equity:    ${res['account_equity']:,.2f}")
        print(f"  High-Water Mark:   ${res['high_water_mark']:,.2f}")
        print(f"  Current Drawdown:  ${res['current_drawdown']:,.2f}")
        print(f"  Market Open:       {res['market_open']}")
        print(f"  Macro Regime:      {'BULL' if res['macro_bull'] else 'BEAR (100% Cash)'}")
        print(f"  Strategy State:    {res['strategy_state']}")
        print(f"  Target Picks:      {', '.join(res['target_symbols']) if res['target_symbols'] else 'None (Cash)'}")
        return

    if args.once:
        print("=== RUNNING SINGLE REBALANCE CYCLE ===")
        res = trader.evaluate_and_rebalance(dry_run=False)
        print(f"Cycle completed. State: {res['strategy_state']}. Orders placed: {res['orders_placed_count']}")
        return

    if args.daemon:
        trader.run_daemon(poll_interval_minutes=args.interval)


if __name__ == "__main__":
    main()
