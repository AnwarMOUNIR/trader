"""
Vectorized & Event-Driven Backtest Simulation Engine.
Coordinates data ingestion, signal evaluation, portfolio allocation, trade execution, and analytics.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.loader import MarketDataLoader, UniverseConfig, AssetMarket
from ..signals.whale_detector import WhaleDetector
from ..portfolio.allocator import PortfolioAllocator, PortfolioAllocationResult
from ..execution.simulator import ExecutionModel, AlmgrenChrissConfig
from .metrics import PerformanceMetrics, calculate_metrics

logger = logging.getLogger("WhaleTracking.BacktestEngine")


@dataclass
class BacktestResult:
    equity_series: pd.Series
    cash_series: pd.Series
    positions_df: pd.DataFrame
    weights_df: pd.DataFrame
    trade_log: pd.DataFrame
    metrics: PerformanceMetrics
    benchmark_equity: pd.Series
    signals_dict: Dict[str, pd.DataFrame]


class BacktestEngine:
    """Institutional-grade backtesting engine for the Whale-Tracking Momentum strategy."""

    def __init__(
        self,
        universe_config: Optional[UniverseConfig] = None,
        whale_detector: Optional[WhaleDetector] = None,
        portfolio_allocator: Optional[PortfolioAllocator] = None,
        execution_model: Optional[ExecutionModel] = None,
        initial_capital: float = 100_000.0,
        rebalance_frequency: int = 1  # Daily rebalancing check
    ):
        self.universe_config = universe_config or UniverseConfig()
        self.data_loader = MarketDataLoader(self.universe_config)
        self.whale_detector = whale_detector or WhaleDetector()
        self.portfolio_allocator = portfolio_allocator or PortfolioAllocator()
        self.execution_model = execution_model or ExecutionModel()
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency

    def run(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        use_cache: bool = True
    ) -> BacktestResult:
        """Run the full backtest simulation."""
        logger.info("Starting Whale-Tracking Momentum Backtest...")

        # 1. Load market data if not provided
        if data_dict is None:
            data_dict = self.data_loader.fetch_all(use_cache=use_cache)

        benchmark_sym = self.universe_config.benchmark_symbol
        benchmark_df = data_dict.get(benchmark_sym)
        active_assets = {k: v for k, v in data_dict.items() if k != benchmark_sym}

        if not active_assets:
            raise ValueError("No active assets available for backtesting.")

        # 2. Extract Signals for all assets
        signals_dict: Dict[str, pd.DataFrame] = {}
        for sym, df in active_assets.items():
            market_type = "BVC" if ".BVC" in sym else "US"
            signals_df = self.whale_detector.process_asset(df, market=market_type, symbol=sym)
            signals_dict[sym] = signals_df

        # 3. Compute Risk-Parity & Volatility-Managed Target Weights
        allocation_res = self.portfolio_allocator.compute_weights(signals_dict, benchmark_df=benchmark_df)
        target_weights = allocation_res.target_weights

        # 4. Step-by-Step Microstructure Simulation Loop
        dates = target_weights.index
        symbols = list(active_assets.keys())

        # Align all active asset dataframes to simulation dates
        for sym in symbols:
            active_assets[sym] = active_assets[sym].reindex(dates).ffill().bfill()

        # Tracking state
        cash = self.initial_capital
        holdings: Dict[str, float] = {sym: 0.0 for sym in symbols}  # share counts
        entry_prices: Dict[str, float] = {sym: 0.0 for sym in symbols}

        equity_history = []
        cash_history = []
        positions_history = []
        trade_records = []

        for day_idx, current_date in enumerate(dates):
            # (a) Process T+3 settlement cash from Casablanca sales
            released_cash = self.execution_model.process_daily_settlement(day_idx)
            cash += released_cash

            # (b) Compute Current Portfolio Equity (Mark to Market)
            current_prices = {sym: active_assets[sym].loc[current_date, "Close"] for sym in symbols}
            daily_volumes = {sym: active_assets[sym].loc[current_date, "Volume"] for sym in symbols}

            # Convert BVC prices to USD for portfolio accounting
            fx_rate = self.universe_config.mad_usd_rate
            asset_values_usd = {}
            for sym in symbols:
                price = current_prices[sym]
                shares = holdings[sym]
                if ".BVC" in sym:
                    val = shares * price * fx_rate
                else:
                    val = shares * price
                asset_values_usd[sym] = val

            # Pending settlement receivables from T+3 clearing queue are part of NAV
            pending_receivables = sum(amount for _, amount in self.execution_model.settlement_queue)
            total_equity = cash + sum(asset_values_usd.values()) + pending_receivables

            # (c) Trailing Stop-Loss & Momentum Crash Protection
            for sym in symbols:
                if holdings[sym] > 0 and entry_prices[sym] > 0:
                    market_type = "BVC" if ".BVC" in sym else "US"
                    conv = fx_rate if market_type == "BVC" else 1.0
                    price_usd = current_prices[sym] * conv
                    ret_from_entry = (price_usd - entry_prices[sym]) / entry_prices[sym]
                    ema_20 = signals_dict[sym].loc[current_date, "EMA_20"] * conv

                    # Stop out if position down > 5% from entry or breaks below 20-day trend EMA
                    is_stop_triggered = (ret_from_entry < -0.05) or (price_usd < ema_20 * 0.98)
                    if is_stop_triggered:
                        sell_val_usd = asset_values_usd[sym]
                        res = self.execution_model.execute_order(
                            symbol=sym,
                            market=market_type,
                            side="SELL",
                            order_value_usd=sell_val_usd,
                            current_price=price_usd,
                            daily_volume=daily_volumes[sym],
                            avg_entry_price=entry_prices[sym],
                            day_index=day_idx
                        )
                        holdings[sym] = 0.0
                        entry_prices[sym] = 0.0
                        if market_type == "US":
                            cash += sell_val_usd - res.total_commission - res.total_slippage

                        trade_records.append({
                            "date": current_date,
                            "symbol": sym,
                            "market": market_type,
                            "side": "STOP_SELL",
                            "shares": res.executed_shares,
                            "price": res.executed_price,
                            "value": sell_val_usd,
                            "commission": res.total_commission,
                            "slippage": res.total_slippage,
                            "tax": res.tax_withheld,
                            "realized_pnl": res.realized_pnl
                        })

            # (d) Rebalance on schedule with Deadband Buffer
            if day_idx % self.rebalance_frequency == 0:
                desired_weights = target_weights.loc[current_date]
                rebalance_deadband_usd = max(2000.0, total_equity * 0.04)  # 4% deadband to avoid churning

                # First pass: Execute SELLS for rebalancing
                for sym in symbols:
                    market_type = "BVC" if ".BVC" in sym else "US"
                    conv = fx_rate if market_type == "BVC" else 1.0
                    price_usd = current_prices[sym] * conv
                    vol = daily_volumes[sym]

                    current_pos_usd = holdings[sym] * price_usd
                    target_pos_usd = total_equity * desired_weights[sym]
                    diff_usd = target_pos_usd - current_pos_usd

                    if diff_usd < -rebalance_deadband_usd and holdings[sym] > 0:  # SELL order
                        sell_val_usd = min(abs(diff_usd), current_pos_usd)
                        res = self.execution_model.execute_order(
                            symbol=sym,
                            market=market_type,
                            side="SELL",
                            order_value_usd=sell_val_usd,
                            current_price=price_usd,
                            daily_volume=vol,
                            avg_entry_price=entry_prices[sym],
                            day_index=day_idx
                        )

                        shares_sold = res.executed_shares
                        holdings[sym] = max(0.0, holdings[sym] - shares_sold)
                        if holdings[sym] <= 1e-4:
                            holdings[sym] = 0.0
                            entry_prices[sym] = 0.0

                        if market_type == "US":
                            cash += sell_val_usd - res.total_commission - res.total_slippage

                        trade_records.append({
                            "date": current_date,
                            "symbol": sym,
                            "market": market_type,
                            "side": "SELL",
                            "shares": shares_sold,
                            "price": res.executed_price,
                            "value": sell_val_usd,
                            "commission": res.total_commission,
                            "slippage": res.total_slippage,
                            "tax": res.tax_withheld,
                            "realized_pnl": res.realized_pnl
                        })

                # Second pass: Execute BUYS with available settled cash
                for sym in symbols:
                    market_type = "BVC" if ".BVC" in sym else "US"
                    conv = fx_rate if market_type == "BVC" else 1.0
                    price_usd = current_prices[sym] * conv
                    vol = daily_volumes[sym]

                    current_pos_usd = holdings[sym] * price_usd
                    target_pos_usd = total_equity * desired_weights[sym]
                    diff_usd = target_pos_usd - current_pos_usd

                    if diff_usd > rebalance_deadband_usd and cash > 1000.0:  # BUY order
                        buy_val_usd = min(diff_usd, cash * 0.95)
                        res = self.execution_model.execute_order(
                            symbol=sym,
                            market=market_type,
                            side="BUY",
                            order_value_usd=buy_val_usd,
                            current_price=price_usd,
                            daily_volume=vol,
                            day_index=day_idx
                        )

                        shares_bought = res.executed_shares
                        if holdings[sym] > 0:
                            prev_cost = holdings[sym] * entry_prices[sym]
                            new_cost = shares_bought * res.executed_price
                            entry_prices[sym] = (prev_cost + new_cost) / (holdings[sym] + shares_bought)
                        else:
                            entry_prices[sym] = res.executed_price

                        holdings[sym] += shares_bought
                        cash -= res.total_cost

                        trade_records.append({
                            "date": current_date,
                            "symbol": sym,
                            "market": market_type,
                            "side": "BUY",
                            "shares": shares_bought,
                            "price": res.executed_price,
                            "value": buy_val_usd,
                            "commission": res.total_commission,
                            "slippage": res.total_slippage,
                            "tax": 0.0,
                            "realized_pnl": 0.0
                        })

            # Record daily state (NAV = Cash + Holdings + Settlement Receivables)
            eod_asset_val = sum(holdings[s] * current_prices[s] * (fx_rate if ".BVC" in s else 1.0) for s in symbols)
            eod_receivables = sum(amount for _, amount in self.execution_model.settlement_queue)
            eod_equity = cash + eod_asset_val + eod_receivables
            equity_history.append(eod_equity)
            cash_history.append(cash + eod_receivables)
            positions_history.append(holdings.copy())

        # Construct result series
        equity_series = pd.Series(equity_history, index=dates, name="Strategy_Equity")
        cash_series = pd.Series(cash_history, index=dates, name="Cash")
        positions_df = pd.DataFrame(positions_history, index=dates)
        trade_log_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
            columns=["date", "symbol", "market", "side", "shares", "price", "value", "commission", "slippage", "tax", "realized_pnl"]
        )

        # Benchmark Equity
        if benchmark_df is not None and not benchmark_df.empty:
            bm_close = benchmark_df.loc[dates.intersection(benchmark_df.index), "Close"]
            bm_close = bm_close.reindex(dates).ffill().bfill()
            bm_returns = bm_close.pct_change().fillna(0.0)
            benchmark_equity = (1.0 + bm_returns).cumprod() * self.initial_capital
        else:
            # Default to 10% annual S&P proxy
            dt_days = (dates - dates[0]).days / 365.25
            benchmark_equity = pd.Series(self.initial_capital * (1.10 ** dt_days), index=dates)

        # Calculate final quantitative metrics
        metrics = calculate_metrics(equity_series, trade_log_df)

        logger.info(f"Backtest completed: CAGR = {metrics.cagr*100:.2f}%, Sharpe = {metrics.sharpe_ratio:.2f}, MDD = {metrics.max_drawdown*100:.2f}%")

        return BacktestResult(
            equity_series=equity_series,
            cash_series=cash_series,
            positions_df=positions_df,
            weights_df=target_weights,
            trade_log=trade_log_df,
            metrics=metrics,
            benchmark_equity=benchmark_equity,
            signals_dict=signals_dict
        )
