"""
Execution Simulator facade and backward compatibility module.
Re-exports ExecutionModel, AlmgrenChrissConfig, and ExecutionResult from src.backtest.order_book.
"""

from ..backtest.order_book import (
    ExecutionModel,
    OrderBookSimulator,
    AlmgrenChrissConfig,
    OrderBookConfig,
    ExecutionResult,
)

__all__ = [
    "ExecutionModel",
    "OrderBookSimulator",
    "AlmgrenChrissConfig",
    "OrderBookConfig",
    "ExecutionResult",
]
