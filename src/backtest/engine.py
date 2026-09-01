"""
Backtest Engine facade and backward compatibility module.
Re-exports IntradayBacktestEngine, BacktestEngine, and BacktestResult from intraday_engine.
"""

from .intraday_engine import (
    IntradayBacktestEngine,
    BacktestEngine,
    BacktestResult,
)
from .order_book import OrderBookSimulator, ExecutionModel, AlmgrenChrissConfig, ExecutionResult
from .bvc_adapter import BVCAdapter, BVCMicrostructureHandler
from .metrics import PerformanceMetrics, calculate_metrics, deflated_sharpe_ratio

__all__ = [
    "IntradayBacktestEngine",
    "BacktestEngine",
    "BacktestResult",
    "OrderBookSimulator",
    "ExecutionModel",
    "AlmgrenChrissConfig",
    "ExecutionResult",
    "BVCAdapter",
    "BVCMicrostructureHandler",
    "PerformanceMetrics",
    "calculate_metrics",
    "deflated_sharpe_ratio",
]
