"""
Backtesting and Performance Analytics Package.

Exposes:
- IntradayBacktestEngine & BacktestEngine
- BacktestResult
- OrderBookSimulator & ExecutionModel
- AlmgrenChrissConfig & ExecutionResult
- BVCAdapter & BVCMicrostructureHandler
- PerformanceMetrics, calculate_metrics, and deflated_sharpe_ratio
"""

from .intraday_engine import (
    IntradayBacktestEngine,
    BacktestEngine,
    BacktestResult,
)
from .order_book import (
    OrderBookSimulator,
    ExecutionModel,
    AlmgrenChrissConfig,
    OrderBookConfig,
    ExecutionResult,
)
from .bvc_adapter import (
    BVCAdapter,
    BVCMicrostructureHandler,
    BVCPriceLimitResult,
    SettlementItem,
)
from .metrics import (
    PerformanceMetrics,
    calculate_metrics,
    deflated_sharpe_ratio,
    detect_annualization_factor,
)

__all__ = [
    "IntradayBacktestEngine",
    "BacktestEngine",
    "BacktestResult",
    "OrderBookSimulator",
    "ExecutionModel",
    "AlmgrenChrissConfig",
    "OrderBookConfig",
    "ExecutionResult",
    "BVCAdapter",
    "BVCMicrostructureHandler",
    "BVCPriceLimitResult",
    "SettlementItem",
    "PerformanceMetrics",
    "calculate_metrics",
    "deflated_sharpe_ratio",
    "detect_annualization_factor",
]
