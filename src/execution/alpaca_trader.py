"""
Alpaca Paper Trading Live Execution Connector.
Handles live authentication, real-time market data scanning, position sizing,
and automated order execution on Alpaca's Paper Trading API.
"""

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import (
        MarketOrderRequest,
        GetOrdersRequest,
        StopLossRequest,
        TakeProfitRequest
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from src.signals.whale_detector import WhaleDetector
from src.portfolio.allocator import PortfolioAllocator

logger = logging.getLogger("WhaleTracking.AlpacaTrader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


class AlpacaPaperTrader:
    """
    Automated Paper Trading Executor connecting the Whale Momentum Strategy
    directly to Alpaca's $100k virtual broker simulation.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        symbols: Optional[List[str]] = None
    ):
        self.api_key = api_key or os.getenv("APCA_API_KEY_ID", "")
        self.secret_key = secret_key or os.getenv("APCA_API_SECRET_KEY", "")
        self.paper = True

        # Default US Momentum Universe
        self.symbols = symbols or [
            "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
            "AVGO", "AMD", "PLTR", "SMCI", "NFLX", "CRM", "ORCL",
            "QCOM", "PANW", "NOW", "SNOW", "CRWD", "UBER"
        ]

        self.trading_client: Optional[TradingClient] = None
        self.data_client: Optional[StockHistoricalDataClient] = None

        if self.api_key and self.secret_key and ALPACA_AVAILABLE:
            try:
                self.trading_client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper=True
                )
                self.data_client = StockHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key
                )
                logger.info("Successfully connected to Alpaca Paper Trading API.")
            except Exception as e:
                logger.error(f"Failed to authenticate with Alpaca API: {e}")
        else:
            logger.warning("Alpaca API keys not found or alpaca-py not available. Running in Dry-Run mode.")

    def get_account_status(self) -> Dict:
        """Fetch live account balance, cash, and equity from Alpaca."""
        if not self.trading_client:
            return {
                "status": "SIMULATED / DRY-RUN",
                "equity": 100_000.0,
                "cash": 100_000.0,
                "buying_power": 200_000.0,
                "portfolio_value": 100_000.0
            }

        try:
            account = self.trading_client.get_account()
            return {
                "status": account.status.value,
                "equity": float(account.equity),
                "cash": float(account.cash),
                "buying_power": float(account.buying_power),
                "portfolio_value": float(account.portfolio_value)
            }
        except Exception as e:
            logger.error(f"Error fetching account status: {e}")
            return {"error": str(e)}

    def get_current_positions(self) -> Dict[str, float]:
        """Fetch current open positions."""
        if not self.trading_client:
            return {}

        try:
            positions = self.trading_client.get_all_positions()
            return {p.symbol: float(p.qty) for p in positions}
        except Exception as e:
            logger.error(f"Error fetching positions: {e}")
            return {}

    def fetch_live_bars(self, days_lookback: int = 300) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical daily bars for the active universe using Alpaca Data API
        or yfinance fallback.
        """
        data_dict = {}
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=days_lookback)

        for sym in self.symbols:
            df = None
            if self.data_client:
                try:
                    req = StockBarsRequest(
                        symbol_or_symbols=sym,
                        timeframe=TimeFrame.Day,
                        start=start_dt,
                        end=end_dt,
                        feed=DataFeed.IEX
                    )
                    bars = self.data_client.get_stock_bars(req)
                    if bars and sym in bars:
                        raw_data = []
                        for b in bars[sym]:
                            raw_data.append({
                                "Date": b.timestamp.date(),
                                "Open": float(b.open),
                                "High": float(b.high),
                                "Low": float(b.low),
                                "Close": float(b.close),
                                "Volume": float(b.volume)
                            })
                        df = pd.DataFrame(raw_data).set_index("Date")
                except Exception as e:
                    logger.warning(f"Alpaca data fetch failed for {sym}: {e}. Trying fallback.")

            if df is None or df.empty:
                import yfinance as yf
                try:
                    ticker = yf.Ticker(sym)
                    hist = ticker.history(period="1y")
                    if not hist.empty:
                        df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
                        df.index = pd.to_datetime(df.index).date
                except Exception as e:
                    logger.error(f"Could not load data for {sym}: {e}")

            if df is not None and len(df) > 30:
                data_dict[sym] = df

        return data_dict

    def run_daily_cycle(self) -> Dict:
        """
        Main execution routine:
        1. Ingests latest live prices and volume.
        2. Computes Whale signals and identifies top 3 leaders.
        3. Rebalances live portfolio on Alpaca Paper with server-side Stop-Loss bracket orders.
        """
        logger.info("Starting Daily Whale-Tracking Alpaca Paper Trading Cycle...")
        account = self.get_account_status()
        current_equity = account.get("equity", 100_000.0)
        current_cash = account.get("cash", 100_000.0)
        current_positions = self.get_current_positions()

        logger.info(f"Account Equity: ${current_equity:,.2f} | Settled Cash: ${current_cash:,.2f}")
        logger.info(f"Current Open Positions: {current_positions}")

        # Ingest Market Data
        data_dict = self.fetch_live_bars()
        if not data_dict:
            logger.error("No market data available for live cycle.")
            return {"status": "FAILED", "reason": "No data"}

        # Benchmark SPY macro check
        import yfinance as yf
        try:
            spy_df = yf.Ticker("SPY").history(period="1y")[["Open", "High", "Low", "Close", "Volume"]]
            spy_df.index = pd.to_datetime(spy_df.index).date
        except Exception:
            spy_df = pd.DataFrame()

        detector = WhaleDetector()
        signals_dict = {}
        for sym, df in data_dict.items():
            signals_dict[sym] = detector.process_asset(df, market="US")

        allocator = PortfolioAllocator(
            target_annual_vol=0.32,
            max_single_weight=0.34,
            max_total_exposure=1.00
        )
        allocation_res = allocator.compute_weights(signals_dict, benchmark_df=spy_df)
        latest_date = allocation_res.target_weights.index[-1]
        target_weights = allocation_res.target_weights.loc[latest_date]

        active_targets = target_weights[target_weights > 0.05].to_dict()
        logger.info(f"Target Portfolio Allocation for {latest_date}: {active_targets}")

        executed_actions = []

        # 1. Close positions no longer in target
        for sym, qty in current_positions.items():
            if sym not in active_targets:
                logger.info(f"Closing position {sym} (Qty: {qty}) - Dropped from whale leaders.")
                if self.trading_client:
                    try:
                        self.trading_client.close_position(sym)
                        executed_actions.append(f"SOLD/CLOSED {sym} ({qty} shares)")
                    except Exception as e:
                        logger.error(f"Error closing position {sym}: {e}")
                else:
                    executed_actions.append(f"[SIMULATED] SOLD {sym} ({qty} shares)")

        # 2. Open or rebalance target positions
        for sym, target_w in active_targets.items():
            target_val = current_equity * target_w
            latest_price = float(data_dict[sym]["Close"].iloc[-1])
            desired_shares = int(target_val / latest_price)

            current_qty = current_positions.get(sym, 0.0)
            diff_shares = desired_shares - int(current_qty)

            if diff_shares > 0:
                stop_price = round(latest_price * 0.952, 2)  # Strict -4.8% Stop-Loss
                logger.info(
                    f"BUY Signal: {sym} | Shares: {diff_shares} @ ~${latest_price:.2f} "
                    f"(Target Value: ${target_val:,.2f}, Stop Loss: ${stop_price:.2f})"
                )

                if self.trading_client:
                    try:
                        # Submit Bracket Market Order with Server-Side Stop Loss
                        order_req = MarketOrderRequest(
                            symbol=sym,
                            qty=diff_shares,
                            side=OrderSide.BUY,
                            time_in_force=TimeInForce.DAY,
                            stop_loss=StopLossRequest(stop_price=stop_price)
                        )
                        order = self.trading_client.submit_order(order_req)
                        executed_actions.append(
                            f"BOUGHT {diff_shares} {sym} @ ${latest_price:.2f} (Stop: ${stop_price:.2f})"
                        )
                    except Exception as e:
                        logger.error(f"Failed to submit order for {sym}: {e}")
                else:
                    executed_actions.append(
                        f"[SIMULATED] BOUGHT {diff_shares} {sym} @ ${latest_price:.2f} (Stop: ${stop_price:.2f})"
                    )

        summary = {
            "timestamp": datetime.now().isoformat(),
            "equity": current_equity,
            "cash": current_cash,
            "target_allocation": active_targets,
            "actions": executed_actions
        }
        logger.info(f"Daily Cycle Completed: {summary}")
        return summary
