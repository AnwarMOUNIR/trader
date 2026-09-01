"""
Alpaca Paper Trading REST API Integration Client.

Implements:
1. Alpaca Paper Trading REST API connection via alpaca-py SDK.
2. Server-side Bracket Orders:
   - Market BUY order with Take-Profit at +1.5 * ATR and Stop-Loss at -1.0 * ATR (strictly capped at -4.8% max).
3. Idempotent position & order state management.
4. Autonomous graceful fallback to simulated Dry-Run mode when unauthenticated or credentials missing.
"""

from __future__ import annotations
import os
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

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
        TakeProfitRequest,
    )
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderStatus, QueryOrderStatus
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import DataFeed
    ALPACA_AVAILABLE = True
except ImportError:
    ALPACA_AVAILABLE = False

from ..signals.whale_detector import WhaleDetector
from ..portfolio.allocator import PortfolioAllocator

logger = logging.getLogger("Execution.AlpacaClient")


@dataclass
class OrderResult:
    """Telemetry record of submitted paper or simulated dry-run order."""
    order_id: str
    symbol: str
    qty: float
    side: str
    status: str
    filled_price: float
    stop_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    is_dry_run: bool = False
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "order_id": self.order_id,
            "symbol": self.symbol,
            "qty": self.qty,
            "side": self.side,
            "status": self.status,
            "filled_price": round(self.filled_price, 2),
            "stop_price": round(self.stop_price, 2) if self.stop_price else None,
            "take_profit_price": round(self.take_profit_price, 2) if self.take_profit_price else None,
            "timestamp": self.timestamp,
            "is_dry_run": self.is_dry_run,
            "details": self.details,
        }


class AlpacaPaperTrader:
    """
    Automated Paper Trading Executor connecting the trading strategy
    to Alpaca's virtual broker or autonomous dry-run engine.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        paper: bool = True,
        max_stop_loss_pct: float = 0.048
    ):
        self.api_key = api_key or os.getenv("APCA_API_KEY_ID", "")
        self.secret_key = secret_key or os.getenv("APCA_API_SECRET_KEY", "")
        self.paper = paper
        self.max_stop_loss_pct = max_stop_loss_pct

        # Default US Momentum Universe
        self.symbols = symbols or [
            "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
            "AVGO", "AMD", "PLTR", "SMCI", "NFLX", "CRM", "ORCL",
            "QCOM", "PANW", "NOW", "SNOW", "CRWD", "UBER"
        ]

        self.trading_client: Optional[TradingClient] = None
        self.data_client: Optional[StockHistoricalDataClient] = None
        self.is_authenticated = False

        # Simulated state for dry-run
        self.simulated_equity = 100_000.0
        self.simulated_cash = 100_000.0
        self.simulated_positions: Dict[str, float] = {}

        if self.api_key and self.secret_key and ALPACA_AVAILABLE:
            try:
                self.trading_client = TradingClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key,
                    paper=self.paper
                )
                self.data_client = StockHistoricalDataClient(
                    api_key=self.api_key,
                    secret_key=self.secret_key
                )
                # Verify credentials with account call
                acct = self.trading_client.get_account()
                self.is_authenticated = True
                logger.info(f"Connected to Alpaca Paper Trading API. Account Equity: ${float(acct.equity):,.2f}")
            except Exception as e:
                logger.warning(f"Could not authenticate with Alpaca API ({e}). Running in Autonomous Dry-Run mode.")
                self.is_authenticated = False
        else:
            logger.info("Alpaca API credentials not provided or alpaca-py missing. Operating in Autonomous Dry-Run mode.")

    def get_account_status(self) -> Dict:
        """Fetch account balance, cash, buying power, and portfolio value."""
        if self.is_authenticated and self.trading_client:
            try:
                account = self.trading_client.get_account()
                return {
                    "status": account.status.value,
                    "equity": float(account.equity),
                    "cash": float(account.cash),
                    "buying_power": float(account.buying_power),
                    "portfolio_value": float(account.portfolio_value),
                    "is_dry_run": False,
                }
            except Exception as e:
                logger.error(f"Error fetching Alpaca account status: {e}")

        return {
            "status": "SIMULATED / DRY-RUN",
            "equity": self.simulated_equity,
            "cash": self.simulated_cash,
            "buying_power": self.simulated_cash * 2.0,
            "portfolio_value": self.simulated_equity,
            "is_dry_run": True,
        }

    def get_current_positions(self) -> Dict[str, float]:
        """Fetch current open position quantities by symbol."""
        if self.is_authenticated and self.trading_client:
            try:
                positions = self.trading_client.get_all_positions()
                return {p.symbol: float(p.qty) for p in positions}
            except Exception as e:
                logger.error(f"Error fetching Alpaca positions: {e}")

        return self.simulated_positions.copy()

    def submit_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        current_price: float,
        atr: float = 0.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None
    ) -> OrderResult:
        """
        Submit a Bracket Order with take-profit and stop-loss:
        - Stop-loss: -1.0 * ATR or stop_loss_pct, strictly capped at -4.8% max.
        - Take-profit: +1.5 * ATR or take_profit_pct (default +15%).
        """
        if qty <= 0 or current_price <= 0:
            return OrderResult(
                order_id=f"invalid_{symbol}",
                symbol=symbol,
                qty=0.0,
                side=side.upper(),
                status="REJECTED",
                filled_price=current_price,
                is_dry_run=True,
                details={"reason": "Invalid qty or price"}
            )

        side_upper = side.upper()
        
        # Calculate dynamic ATR-based or percentage bracket stops
        if atr > 0:
            sl_distance = min(atr * 1.0, current_price * self.max_stop_loss_pct)
            tp_distance = atr * 1.5
        else:
            sl_pct = stop_loss_pct if stop_loss_pct is not None else self.max_stop_loss_pct
            sl_pct = min(sl_pct, self.max_stop_loss_pct)
            sl_distance = current_price * sl_pct
            tp_pct = take_profit_pct if take_profit_pct is not None else 0.15
            tp_distance = current_price * tp_pct

        stop_price = round(max(current_price - sl_distance, current_price * (1.0 - self.max_stop_loss_pct)), 2)
        take_profit_price = round(current_price + tp_distance, 2)

        order_id = f"bracket_{symbol}_{int(datetime.now().timestamp())}"

        if self.is_authenticated and self.trading_client and side_upper == "BUY":
            try:
                order_req = MarketOrderRequest(
                    symbol=symbol,
                    qty=int(qty),
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                    stop_loss=StopLossRequest(stop_price=stop_price),
                    take_profit=TakeProfitRequest(limit_price=take_profit_price)
                )
                order = self.trading_client.submit_order(order_req)
                return OrderResult(
                    order_id=str(order.id),
                    symbol=symbol,
                    qty=float(order.qty),
                    side="BUY",
                    status=str(order.status),
                    filled_price=current_price,
                    stop_price=stop_price,
                    take_profit_price=take_profit_price,
                    is_dry_run=False,
                    details={"alpaca_client_order_id": order.client_order_id}
                )
            except Exception as e:
                logger.error(f"Alpaca bracket order submission error for {symbol}: {e}")

        # Simulated Dry-Run Execution
        if side_upper == "BUY":
            self.simulated_positions[symbol] = self.simulated_positions.get(symbol, 0.0) + qty
            cost = qty * current_price
            self.simulated_cash = max(0.0, self.simulated_cash - cost)
        else:
            self.simulated_positions[symbol] = max(0.0, self.simulated_positions.get(symbol, 0.0) - qty)
            proceeds = qty * current_price
            self.simulated_cash += proceeds

        return OrderResult(
            order_id=order_id,
            symbol=symbol,
            qty=qty,
            side=side_upper,
            status="FILLED_SIMULATED",
            filled_price=current_price,
            stop_price=stop_price if side_upper == "BUY" else None,
            take_profit_price=take_profit_price if side_upper == "BUY" else None,
            is_dry_run=True,
            details={"note": "Autonomous Dry-Run Simulation"}
        )

    def execute_decision(
        self,
        symbol: str,
        action: str,
        target_weight: float,
        current_price: float,
        atr: float = 0.0
    ) -> OrderResult:
        """
        Translates model decision (BUY, SELL, HOLD) into bracket execution order.
        """
        acct = self.get_account_status()
        equity = acct.get("equity", 100_000.0)
        action_upper = action.upper()

        if action_upper == "BUY" and target_weight > 0 and current_price > 0:
            target_val = equity * min(target_weight, 0.34)
            desired_shares = max(int(target_val / current_price), 1)
            current_qty = self.get_current_positions().get(symbol, 0.0)
            diff_shares = desired_shares - int(current_qty)
            if diff_shares > 0:
                return self.submit_bracket_order(
                    symbol=symbol,
                    qty=float(diff_shares),
                    side="BUY",
                    current_price=current_price,
                    atr=atr
                )

        elif action_upper == "SELL":
            current_qty = self.get_current_positions().get(symbol, 0.0)
            if current_qty > 0:
                return self.close_position(symbol, qty=current_qty)

        return OrderResult(
            order_id=f"hold_{symbol}",
            symbol=symbol,
            qty=0.0,
            side="HOLD",
            status="NO_ACTION",
            filled_price=current_price,
            is_dry_run=True
        )

    def close_position(self, symbol: str, qty: Optional[float] = None) -> OrderResult:
        """Close open position for a given symbol."""
        current_qty = self.get_current_positions().get(symbol, 0.0)
        close_qty = qty if qty is not None else current_qty

        if self.is_authenticated and self.trading_client:
            try:
                self.trading_client.close_position(symbol)
                return OrderResult(
                    order_id=f"close_{symbol}",
                    symbol=symbol,
                    qty=close_qty,
                    side="SELL",
                    status="CLOSED",
                    filled_price=0.0,
                    is_dry_run=False
                )
            except Exception as e:
                logger.error(f"Error closing position {symbol}: {e}")

        # Simulated Dry-Run Close
        if symbol in self.simulated_positions:
            del self.simulated_positions[symbol]

        return OrderResult(
            order_id=f"close_sim_{symbol}",
            symbol=symbol,
            qty=close_qty,
            side="SELL",
            status="CLOSED_SIMULATED",
            filled_price=0.0,
            is_dry_run=True
        )

    def fetch_live_bars(self, days_lookback: int = 300) -> Dict[str, pd.DataFrame]:
        """Fetch historical daily bars for active universe using Alpaca, yfinance, or cached store."""
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
                    logger.warning(f"Alpaca data fetch failed for {sym}: {e}")

            if df is None or df.empty:
                import yfinance as yf
                try:
                    ticker = yf.Ticker(sym)
                    hist = ticker.history(period="1y")
                    if not hist.empty:
                        df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
                        df.index = pd.to_datetime(df.index).date
                except Exception as e:
                    logger.debug(f"yfinance fallback fetch error for {sym}: {e}")

            if df is not None and len(df) > 30:
                data_dict[sym] = df

        if not data_dict:
            try:
                from ..data.loader import MarketDataLoader, UniverseConfig
                loader = MarketDataLoader(UniverseConfig(us_symbols=self.symbols, bvc_symbols=[]))
                loaded = loader.fetch_all(use_cache=True)
                for sym, ldf in loaded.items():
                    if sym in self.symbols and not ldf.empty:
                        data_dict[sym] = ldf
            except Exception as e:
                logger.warning(f"Fallback to MarketDataLoader failed: {e}")

        return data_dict

    def run_daily_cycle(self) -> Dict:
        """Run standard daily paper cycle."""
        logger.info("Executing Alpaca Paper Daily Cycle...")
        account = self.get_account_status()
        current_equity = account.get("equity", 100_000.0)
        current_cash = account.get("cash", 100_000.0)
        current_positions = self.get_current_positions()

        data_dict = self.fetch_live_bars()
        if not data_dict:
            return {
                "timestamp": datetime.now().isoformat(),
                "status": "NO_DATA",
                "equity": current_equity,
                "cash": current_cash,
                "target_allocation": {},
                "executed_orders": [],
                "actions": [],
                "is_dry_run": not self.is_authenticated,
            }

        # Use WhaleDetector and PortfolioAllocator
        detector = WhaleDetector()
        signals_dict = {sym: detector.process_asset(df, market="US", symbol=sym) for sym, df in data_dict.items()}

        allocator = PortfolioAllocator(target_annual_vol=0.32, max_single_weight=0.34, max_total_exposure=1.00)
        allocation_res = allocator.compute_weights(signals_dict)
        latest_date = allocation_res.target_weights.index[-1]
        target_weights = allocation_res.target_weights.loc[latest_date]
        active_targets = target_weights[target_weights > 0.05].to_dict()

        executed_orders = []

        # 1. Close dropped positions
        for sym, qty in current_positions.items():
            if sym not in active_targets:
                res = self.close_position(sym, qty=qty)
                executed_orders.append(res.to_dict())

        # 2. Open / rebalance active targets with bracket stop-loss
        for sym, w in active_targets.items():
            price = float(data_dict[sym]["Close"].iloc[-1])
            res = self.execute_decision(symbol=sym, action="BUY", target_weight=w, current_price=price)
            if res.status != "NO_ACTION":
                executed_orders.append(res.to_dict())

        return {
            "timestamp": datetime.now().isoformat(),
            "status": "SUCCESS",
            "equity": current_equity,
            "cash": current_cash,
            "target_allocation": active_targets,
            "executed_orders": executed_orders,
            "actions": executed_orders,
            "is_dry_run": not self.is_authenticated,
        }


# Alias for backward compatibility
AlpacaClient = AlpacaPaperTrader
