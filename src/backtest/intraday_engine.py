"""
Discrete-Event Intraday Backtesting Engine (5m to 60m & Daily Resolutions).

Implements:
1. High-frequency bar-by-bar discrete simulation loop across multi-asset universes.
2. Almgren-Chriss (2000) market impact, bid-ask spread crossing, and broker commissions.
3. Casablanca Stock Exchange (BVC) +/-10% circuit breakers, T+3 settlement queues, and 15% IR tax.
4. Portfolio risk rules: max single-asset allocation <= 34%, gross exposure <= 100%, bracket stop loss.
5. Realistic fills on gap opens and stops with slippage.
6. Detailed BacktestResult with time series, trade logs, and comprehensive quantitative metrics.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..data.loader import MarketDataLoader, UniverseConfig
from ..signals.whale_detector import WhaleDetector
from ..portfolio.allocator import PortfolioAllocator, PortfolioAllocationResult
from .order_book import OrderBookSimulator, AlmgrenChrissConfig, ExecutionResult
from .bvc_adapter import BVCAdapter
from .metrics import PerformanceMetrics, calculate_metrics, detect_annualization_factor

logger = logging.getLogger("Backtest.IntradayEngine")


@dataclass
class BacktestResult:
    """Complete results and analytics of a backtest simulation."""
    equity_series: pd.Series
    cash_series: pd.Series
    positions_df: pd.DataFrame
    weights_df: pd.DataFrame
    trade_log: pd.DataFrame
    metrics: PerformanceMetrics
    benchmark_equity: pd.Series
    signals_dict: Dict[str, pd.DataFrame]

    @property
    def equity_curve(self) -> pd.Series:
        """Alias for equity_series matching Interface Contract #3."""
        return self.equity_series

    @property
    def trades(self) -> pd.DataFrame:
        """Alias for trade_log matching Interface Contract #3."""
        return self.trade_log

    def summary(self) -> Dict[str, Union[float, int, str]]:
        """Return formatted dictionary of performance metrics."""
        return self.metrics.to_dict()


class IntradayBacktestEngine:
    """
    Institutional-grade discrete-event intraday backtesting engine.
    Operates at 5m, 15m, 30m, 60m, and daily intervals with full microstructure friction simulation.
    """

    def __init__(
        self,
        universe_config: Optional[UniverseConfig] = None,
        whale_detector: Optional[WhaleDetector] = None,
        portfolio_allocator: Optional[PortfolioAllocator] = None,
        execution_model: Optional[OrderBookSimulator] = None,
        bvc_adapter: Optional[BVCAdapter] = None,
        initial_capital: float = 100_000.0,
        rebalance_frequency: int = 1,
        stop_loss_pct: float = 0.048,
        deadband_pct: float = 0.04
    ):
        self.universe_config = universe_config or UniverseConfig()
        self.data_loader = MarketDataLoader(self.universe_config)
        self.whale_detector = whale_detector or WhaleDetector()
        self.portfolio_allocator = portfolio_allocator or PortfolioAllocator()
        self.execution_model = execution_model or OrderBookSimulator()
        self.bvc_adapter = bvc_adapter or BVCAdapter()
        self.initial_capital = initial_capital
        self.rebalance_frequency = rebalance_frequency
        self.stop_loss_pct = stop_loss_pct
        self.deadband_pct = deadband_pct

    def run_simulation(
        self,
        intraday_data: Optional[Dict[str, pd.DataFrame]] = None,
        daily_macro_data: Optional[Dict[str, pd.DataFrame]] = None,
        sentiment_data: Optional[Dict[str, pd.DataFrame]] = None,
        signals: Optional[Dict[str, pd.DataFrame]] = None,
        target_weights: Optional[pd.DataFrame] = None,
        initial_capital: Optional[float] = None,
        **kwargs
    ) -> BacktestResult:
        """Alias for run method supporting multi-source data and signals."""
        data = intraday_data or daily_macro_data or kwargs.get("data") or kwargs.get("data_dict")
        sig = signals or kwargs.get("signals") or kwargs.get("signals_dict")
        return self.run(
            data_dict=data,
            signals_dict=sig,
            target_weights=target_weights,
            initial_capital=initial_capital,
            **kwargs
        )

    def run(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        signals_dict: Optional[Dict[str, pd.DataFrame]] = None,
        target_weights: Optional[pd.DataFrame] = None,
        initial_capital: Optional[float] = None,
        use_cache: bool = True,
        **kwargs
    ) -> BacktestResult:
        """
        Run the discrete-event intraday simulation loop.
        """
        if data_dict is None and "data" in kwargs:
            data_dict = kwargs["data"]
        if signals_dict is None and "signals" in kwargs:
            signals_dict = kwargs["signals"]

        capital = initial_capital if initial_capital is not None else self.initial_capital
        logger.info(f"Starting Intraday Backtest Simulation with initial capital ${capital:,.2f}...")

        # 1. Ingest market data if not provided
        if data_dict is None:
            data_dict = self.data_loader.fetch_all(use_cache=use_cache)

        if not data_dict:
            raise ValueError("No market data available for backtesting.")

        benchmark_sym = self.universe_config.benchmark_symbol
        benchmark_df = data_dict.get(benchmark_sym)
        active_assets = {k: v.copy() for k, v in data_dict.items() if k != benchmark_sym and not v.empty}

        if not active_assets:
            raise ValueError("No active assets available for backtesting.")

        # Standardize column naming (capitalized Open, High, Low, Close, Volume)
        for sym, df in active_assets.items():
            col_map = {c: c.capitalize() for c in df.columns if c.lower() in ["open", "high", "low", "close", "volume", "vwap"]}
            active_assets[sym] = df.rename(columns=col_map)

        # 2. Extract or use provided signals
        if signals_dict is None:
            signals_dict = {}
            for sym, df in active_assets.items():
                market_type = "BVC" if (".BVC" in sym or "BVC" in sym.upper()) else "US"
                signals_df = self.whale_detector.process_asset(df, market=market_type, symbol=sym)
                signals_dict[sym] = signals_df

        # 3. Compute Target Portfolio Weights if not explicitly provided
        if target_weights is None:
            # Check if signals_dict has direct 'target_weight' column (e.g. from DNNPredictor / MultiHorizonDNN)
            has_target_weights = bool(signals_dict) and all(
                "target_weight" in df.columns for df in signals_dict.values() if not df.empty
            )
            if has_target_weights:
                all_ts = sorted(list(set.union(*[set(df.index) for df in signals_dict.values() if not df.empty])))
                target_weights = pd.DataFrame(0.0, index=all_ts, columns=list(active_assets.keys()))
                for sym in active_assets.keys():
                    if sym in signals_dict and not signals_dict[sym].empty and "target_weight" in signals_dict[sym].columns:
                        target_weights[sym] = signals_dict[sym]["target_weight"].reindex(all_ts).fillna(0.0)
                # Enforce single asset max allocation <= 34% and total gross exposure <= 100%
                target_weights = target_weights.clip(upper=0.34)
                total_exp = target_weights.sum(axis=1)
                over_exp = total_exp > 1.00
                if over_exp.any():
                    target_weights[over_exp] = target_weights[over_exp].div(total_exp[over_exp], axis=0)
            else:
                allocation_res = self.portfolio_allocator.compute_weights(signals_dict, benchmark_df=benchmark_df)
                target_weights = allocation_res.target_weights

        # 4. Align Simulation Timestamps
        timestamps = target_weights.index
        symbols = list(active_assets.keys())

        for sym in symbols:
            active_assets[sym] = active_assets[sym].reindex(timestamps).ffill().bfill()

        # State tracking
        cash = float(capital)
        holdings: Dict[str, float] = {sym: 0.0 for sym in symbols}
        entry_prices: Dict[str, float] = {sym: 0.0 for sym in symbols}
        fx_rate = self.universe_config.mad_usd_rate

        equity_history: List[float] = []
        cash_history: List[float] = []
        positions_history: List[Dict[str, float]] = []
        trade_records: List[Dict] = []

        # Reset adapters
        self.bvc_adapter.reset()

        for bar_idx, current_ts in enumerate(timestamps):
            # (a) Process T+3 settlement cash from prior BVC sales
            released_cash, _ = self.bvc_adapter.process_settlement_queue(bar_idx)
            # Also release from execution_model if separate
            exec_released = self.execution_model.process_daily_settlement(bar_idx)
            cash += (released_cash + exec_released)

            # (b) Compute Current Asset Mark-to-Market Values
            current_prices = {sym: float(active_assets[sym].loc[current_ts, "Close"]) for sym in symbols}
            current_opens = {sym: float(active_assets[sym].loc[current_ts, "Open"]) for sym in symbols}
            current_lows = {sym: float(active_assets[sym].loc[current_ts, "Low"]) for sym in symbols}
            bar_volumes = {sym: float(active_assets[sym].loc[current_ts, "Volume"]) for sym in symbols}

            asset_values_usd = {}
            for sym in symbols:
                price = current_prices[sym]
                shares = holdings[sym]
                conv = fx_rate if (".BVC" in sym or "BVC" in sym.upper()) else 1.0
                asset_values_usd[sym] = shares * price * conv

            # Pending receivables from T+3 settlement queue are included in total NAV
            pending_receivables = self.bvc_adapter.get_pending_settlement_total() + sum(
                amt for _, amt in self.execution_model.settlement_queue
            )
            total_equity = cash + sum(asset_values_usd.values()) + pending_receivables

            # (c) Stop-Loss & Risk Management Trigger Check
            for sym in symbols:
                if holdings[sym] > 0 and entry_prices[sym] > 0:
                    market_type = "BVC" if (".BVC" in sym or "BVC" in sym.upper()) else "US"
                    conv = fx_rate if market_type == "BVC" else 1.0
                    
                    entry_p = entry_prices[sym]
                    curr_p = current_prices[sym] * conv
                    low_p = current_lows[sym] * conv
                    open_p = current_opens[sym] * conv

                    # Check intraday lowest price for stop loss penetration
                    ret_from_entry = (low_p - entry_p) / entry_p
                    
                    # Stop triggered if dropped beyond stop_loss_pct (-4.8% max)
                    if ret_from_entry <= -self.stop_loss_pct:
                        # Realistic gap execution: execute at min(open_p, stop_price)
                        theoretical_stop_p = entry_p * (1.0 - self.stop_loss_pct)
                        exec_fill_ref = min(open_p, theoretical_stop_p) / conv

                        # Check BVC circuit breakers if applicable
                        if market_type == "BVC":
                            cb_res = self.bvc_adapter.check_circuit_breaker(exec_fill_ref, entry_p / conv)
                            exec_fill_ref = cb_res.clamped_price

                        sell_val_usd = holdings[sym] * exec_fill_ref * conv
                        res = self.execution_model.execute_order(
                            symbol=sym,
                            market=market_type,
                            side="SELL",
                            order_value_usd=sell_val_usd,
                            current_price=exec_fill_ref * conv,
                            daily_volume=bar_volumes[sym],
                            avg_entry_price=entry_p,
                            day_index=bar_idx,
                            timestamp=current_ts
                        )

                        # BVC T+3 handling
                        if market_type == "BVC":
                            self.bvc_adapter.queue_settlement_proceeds(
                                order_id=f"stop_{bar_idx}_{sym}",
                                symbol=sym,
                                net_proceeds_usd=res.total_cost,
                                current_day_index=bar_idx,
                                timestamp=current_ts
                            )
                        else:
                            cash += res.total_cost

                        trade_records.append({
                            "date": current_ts,
                            "symbol": sym,
                            "market": market_type,
                            "side": "STOP_LOSS",
                            "shares": holdings[sym],
                            "price": res.executed_price,
                            "value": sell_val_usd,
                            "commission": res.total_commission,
                            "slippage": res.total_slippage,
                            "tax": res.tax_withheld,
                            "realized_pnl": res.realized_pnl
                        })

                        holdings[sym] = 0.0
                        entry_prices[sym] = 0.0

            # (d) Rebalance on Scheduled Frequency with Deadband Buffer
            if bar_idx % self.rebalance_frequency == 0:
                desired_weights = target_weights.loc[current_ts]
                rebalance_deadband_usd = max(2000.0, total_equity * self.deadband_pct)

                # Pass 1: Execute SELL orders to free cash
                for sym in symbols:
                    market_type = "BVC" if (".BVC" in sym or "BVC" in sym.upper()) else "US"
                    conv = fx_rate if market_type == "BVC" else 1.0
                    price_usd = current_prices[sym] * conv
                    vol = bar_volumes[sym]

                    current_pos_usd = holdings[sym] * price_usd
                    target_pos_usd = total_equity * float(desired_weights.get(sym, 0.0))
                    diff_usd = target_pos_usd - current_pos_usd

                    if diff_usd < -rebalance_deadband_usd and holdings[sym] > 0:
                        sell_val_usd = min(abs(diff_usd), current_pos_usd)
                        
                        # Check circuit breaker for BVC
                        ref_p = entry_prices[sym] if entry_prices[sym] > 0 else price_usd
                        if market_type == "BVC":
                            cb_res = self.bvc_adapter.check_circuit_breaker(price_usd, ref_p)
                            exec_p = cb_res.clamped_price
                        else:
                            exec_p = price_usd

                        res = self.execution_model.execute_order(
                            symbol=sym,
                            market=market_type,
                            side="SELL",
                            order_value_usd=sell_val_usd,
                            current_price=exec_p,
                            daily_volume=vol,
                            avg_entry_price=entry_prices[sym],
                            day_index=bar_idx,
                            timestamp=current_ts
                        )

                        shares_sold = res.executed_shares
                        holdings[sym] = max(0.0, holdings[sym] - shares_sold)
                        if holdings[sym] <= 1e-4:
                            holdings[sym] = 0.0
                            entry_prices[sym] = 0.0

                        if market_type == "BVC":
                            self.bvc_adapter.queue_settlement_proceeds(
                                order_id=f"rebal_sell_{bar_idx}_{sym}",
                                symbol=sym,
                                net_proceeds_usd=res.total_cost,
                                current_day_index=bar_idx,
                                timestamp=current_ts
                            )
                        else:
                            cash += res.total_cost

                        trade_records.append({
                            "date": current_ts,
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

                # Pass 2: Execute BUY orders using settled cash
                for sym in symbols:
                    market_type = "BVC" if (".BVC" in sym or "BVC" in sym.upper()) else "US"
                    conv = fx_rate if market_type == "BVC" else 1.0
                    price_usd = current_prices[sym] * conv
                    vol = bar_volumes[sym]

                    current_pos_usd = holdings[sym] * price_usd
                    target_w = float(desired_weights.get(sym, 0.0))
                    # Enforce strict single-asset max weight ceiling (34%)
                    target_w = min(target_w, 0.34)
                    target_pos_usd = total_equity * target_w
                    diff_usd = target_pos_usd - current_pos_usd

                    if diff_usd > rebalance_deadband_usd and cash > 1000.0:
                        buy_val_usd = min(diff_usd, cash * 0.95)
                        
                        # Check circuit breaker for BVC
                        ref_p = entry_prices[sym] if entry_prices[sym] > 0 else price_usd
                        if market_type == "BVC":
                            cb_res = self.bvc_adapter.check_circuit_breaker(price_usd, ref_p)
                            exec_p = cb_res.clamped_price
                        else:
                            exec_p = price_usd

                        res = self.execution_model.execute_order(
                            symbol=sym,
                            market=market_type,
                            side="BUY",
                            order_value_usd=buy_val_usd,
                            current_price=exec_p,
                            daily_volume=vol,
                            day_index=bar_idx,
                            timestamp=current_ts
                        )

                        shares_bought = res.executed_shares
                        if shares_bought > 0:
                            if holdings[sym] > 0:
                                prev_cost = holdings[sym] * entry_prices[sym]
                                new_cost = shares_bought * res.executed_price
                                entry_prices[sym] = (prev_cost + new_cost) / (holdings[sym] + shares_bought)
                            else:
                                entry_prices[sym] = res.executed_price

                            holdings[sym] += shares_bought
                            cash -= res.total_cost

                            trade_records.append({
                                "date": current_ts,
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

            # End of Bar: Mark-to-Market Portfolio Valuation
            eod_asset_val = sum(
                holdings[s] * current_prices[s] * (fx_rate if (".BVC" in s or "BVC" in s.upper()) else 1.0)
                for s in symbols
            )
            eod_receivables = self.bvc_adapter.get_pending_settlement_total() + sum(
                amt for _, amt in self.execution_model.settlement_queue
            )
            eod_equity = cash + eod_asset_val + eod_receivables

            equity_history.append(eod_equity)
            cash_history.append(cash + eod_receivables)
            positions_history.append(holdings.copy())

        # Construct result Series and DataFrames
        equity_series = pd.Series(equity_history, index=timestamps, name="Strategy_Equity")
        cash_series = pd.Series(cash_history, index=timestamps, name="Cash")
        positions_df = pd.DataFrame(positions_history, index=timestamps)
        trade_log_df = pd.DataFrame(trade_records) if trade_records else pd.DataFrame(
            columns=["date", "symbol", "market", "side", "shares", "price", "value", "commission", "slippage", "tax", "realized_pnl"]
        )

        # Benchmark Equity series
        if benchmark_df is not None and not benchmark_df.empty:
            bm_close = benchmark_df.loc[timestamps.intersection(benchmark_df.index), "Close"]
            bm_close = bm_close.reindex(timestamps).ffill().bfill()
            bm_returns = bm_close.pct_change().fillna(0.0)
            benchmark_equity = (1.0 + bm_returns).cumprod() * capital
        else:
            # Deterministic 10% annual S&P benchmark proxy
            if isinstance(timestamps, pd.DatetimeIndex):
                dt_days = (timestamps - timestamps[0]).total_seconds() / (86400.0 * 365.25)
            else:
                dt_days = np.arange(len(timestamps)) / 252.0
            benchmark_equity = pd.Series(capital * (1.10 ** dt_days), index=timestamps, name="Benchmark_SPY")

        # Annualization factor inference and metrics
        ann_factor = detect_annualization_factor(equity_series)
        metrics = calculate_metrics(equity_series, trade_log_df, annualization_factor=ann_factor)

        logger.info(
            f"Backtest Finished: CAGR = {metrics.cagr*100:.2f}%, Sharpe = {metrics.sharpe_ratio:.2f}, "
            f"MDD = {metrics.max_drawdown*100:.2f}%, Win Rate = {metrics.win_rate*100:.2f}%"
        )

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


# Alias for backward compatibility
BacktestEngine = IntradayBacktestEngine
