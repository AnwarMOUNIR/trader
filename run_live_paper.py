"""
Live Paper Trading Entrypoint.
Executes the Whale-Tracking Momentum strategy on Alpaca's Paper Trading API.
Can be run on-demand, scheduled via Windows Task Scheduler, or via GitHub Actions.
"""

import sys
import os
import argparse
from src.execution.alpaca_trader import AlpacaPaperTrader


def main():
    parser = argparse.ArgumentParser(description="Whale-Tracking Live Paper Trader")
    parser.add_argument("--api-key", type=str, default=os.getenv("APCA_API_KEY_ID"), help="Alpaca API Key ID")
    parser.add_argument("--secret-key", type=str, default=os.getenv("APCA_API_SECRET_KEY"), help="Alpaca Secret Key")
    args = parser.parse_args()

    print("=" * 80)
    print("      WHALE-TRACKING MOMENTUM LIVE PAPER TRADING ENGINE")
    print("=" * 80)

    trader = AlpacaPaperTrader(
        api_key=args.api_key,
        secret_key=args.secret_key
    )

    result = trader.run_daily_cycle()

    print("\n" + "=" * 80)
    print("                      DAILY EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Timestamp:       {result.get('timestamp')}")
    print(f"Account Equity:  ${result.get('equity', 0):,.2f}")
    print(f"Settled Cash:    ${result.get('cash', 0):,.2f}")
    print(f"Active Targets:  {result.get('target_allocation')}")
    print("Actions:")
    for action in result.get("actions", []):
        print(f"  • {action}")
    if not result.get("actions"):
        print("  • Portfolio optimal. No rebalancing required today.")
    print("=" * 80)


if __name__ == "__main__":
    main()
