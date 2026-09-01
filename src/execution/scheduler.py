"""
Automated Trading Scheduler & Market Session Controller.

Implements:
1. Scheduled cycle triggers for 5m, 15m, 30m, 60m, and daily trading cycles.
2. Market calendar and session checks for US (NYSE/NASDAQ) and Casablanca (BVC) markets.
3. Execution cycle loop runner with exception resilience and telemetry updates.
4. Single-iteration step execution and multi-iteration daemon scheduling.
"""

from __future__ import annotations
import logging
import time
from datetime import datetime, time as dtime, timezone
from typing import Callable, Dict, List, Optional, Union

import pandas as pd

from .dry_run_simulator import DryRunSimulator, ExecutionCycleReport
from .alpaca_client import AlpacaPaperTrader

logger = logging.getLogger("Execution.Scheduler")


class ExecutionScheduler:
    """
    Automated Execution Scheduler managing trading loop triggers and session checks.
    """

    def __init__(
        self,
        dry_run_simulator: Optional[DryRunSimulator] = None,
        alpaca_trader: Optional[AlpacaPaperTrader] = None,
        interval_minutes: int = 5,
        force_market_open: bool = False
    ):
        self.simulator = dry_run_simulator or DryRunSimulator()
        self.trader = alpaca_trader or AlpacaPaperTrader()
        self.interval_minutes = interval_minutes
        self.force_market_open = force_market_open
        self.is_running = False

    def is_market_open(self, market: str = "US", current_dt: Optional[datetime] = None) -> bool:
        """
        Determine whether the specified market is currently open for trading.
        """
        if self.force_market_open:
            return True

        dt = current_dt or datetime.now(timezone.utc)
        weekday = dt.weekday()  # Monday = 0, Sunday = 6

        # Weekend check
        if weekday >= 5:
            return False

        hour = dt.hour
        minute = dt.minute
        current_time_utc = dtime(hour, minute)

        market_upper = market.upper()
        if "BVC" in market_upper:
            # Casablanca standard session: 08:30 to 14:30 UTC
            open_utc = dtime(8, 30)
            close_utc = dtime(14, 30)
            return open_utc <= current_time_utc <= close_utc
        else:
            # US regular trading hours: 13:30 to 20:00 UTC (EDT) or 14:30 to 21:00 UTC (EST)
            open_utc = dtime(13, 30)
            close_utc = dtime(21, 0)
            return open_utc <= current_time_utc <= close_utc

    def step(self) -> ExecutionCycleReport:
        """
        Run a single execution iteration and persist telemetry.
        """
        logger.info(f"Executing scheduled cycle step (interval: {self.interval_minutes}m)...")
        report = self.simulator.run_cycle()
        logger.info(f"Cycle completed with status: {report.status}. Orders placed: {len(report.executed_orders)}")
        return report

    def run_iterations(self, max_iterations: int = 1, sleep_seconds: Optional[int] = None) -> List[ExecutionCycleReport]:
        """
        Run a specified number of scheduled cycles.
        """
        reports: List[ExecutionCycleReport] = []
        delay = sleep_seconds if sleep_seconds is not None else (self.interval_minutes * 60)

        logger.info(f"Starting scheduler loop for {max_iterations} iteration(s) with {delay}s delay...")
        self.is_running = True

        for i in range(max_iterations):
            try:
                report = self.step()
                reports.append(report)
            except Exception as e:
                logger.error(f"Error during iteration {i+1}: {e}", exc_info=True)

            if i < max_iterations - 1:
                time.sleep(delay)

        self.is_running = False
        return reports
