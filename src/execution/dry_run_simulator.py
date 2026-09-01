"""
Autonomous Dry-Run Simulator & Execution Telemetry Engine.

Implements:
1. Zero-credential autonomous market scanning and signal generation.
2. Target position sizing with risk-parity and portfolio caps (<= 34% single, <= 100% gross).
3. Bracket order generation with server-side / simulated stop-loss (-4.8% max) and take-profit (+1.5 ATR / +15%).
4. Persistent execution logging and snapshot exports to:
   - output/current_signals_snapshot.json
   - output/live_execution_log.json
   - output/live_dry_run_log.json
"""

from __future__ import annotations
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..data.loader import MarketDataLoader, UniverseConfig
from ..signals.whale_detector import WhaleDetector
from ..portfolio.allocator import PortfolioAllocator
from .alpaca_client import AlpacaPaperTrader, OrderResult

logger = logging.getLogger("Execution.DryRunSimulator")


@dataclass
class ExecutionCycleReport:
    """Detailed telemetry record for an autonomous execution cycle."""
    timestamp: str
    cycle_id: str
    universe_size: int
    account_equity: float
    settled_cash: float
    active_signals: Dict[str, Dict]
    target_weights: Dict[str, float]
    executed_orders: List[Dict]
    status: str
    is_dry_run: bool = True

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "cycle_id": self.cycle_id,
            "universe_size": self.universe_size,
            "account_equity": round(self.account_equity, 2),
            "settled_cash": round(self.settled_cash, 2),
            "active_signals": self.active_signals,
            "target_weights": {k: round(v, 4) for k, v in self.target_weights.items()},
            "executed_orders": self.executed_orders,
            "status": self.status,
            "is_dry_run": self.is_dry_run,
        }


class DryRunSimulator:
    """
    Autonomous Execution Engine running scanner, model inference,
    portfolio weighting, and bracket order logging with zero dependencies.
    """

    def __init__(
        self,
        universe_config: Optional[UniverseConfig] = None,
        output_dir: str = "output",
        initial_capital: float = 100_000.0,
        stop_loss_pct: float = 0.048
    ):
        self.universe_config = universe_config or UniverseConfig()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.initial_capital = initial_capital
        self.stop_loss_pct = stop_loss_pct

        self.trader = AlpacaPaperTrader(max_stop_loss_pct=self.stop_loss_pct)
        self.whale_detector = WhaleDetector()
        self.allocator = PortfolioAllocator(
            target_annual_vol=0.32,
            max_single_weight=0.34,
            max_total_exposure=1.00
        )
        self.data_loader = MarketDataLoader(self.universe_config)

    def run_cycle(
        self,
        data_dict: Optional[Dict[str, pd.DataFrame]] = None,
        custom_signals: Optional[Dict[str, pd.DataFrame]] = None
    ) -> ExecutionCycleReport:
        """
        Execute one full market scan and paper/dry-run trading cycle.
        """
        cycle_ts = datetime.now().isoformat()
        cycle_id = f"cycle_{int(datetime.now().timestamp())}"
        logger.info(f"Starting Autonomous Dry-Run Cycle [{cycle_id}] at {cycle_ts}...")

        # 1. Ingest market data
        if data_dict is None:
            data_dict = self.data_loader.fetch_all(use_cache=True)

        benchmark_sym = self.universe_config.benchmark_symbol
        benchmark_df = data_dict.get(benchmark_sym)
        active_assets = {k: v for k, v in data_dict.items() if k != benchmark_sym and not v.empty}

        if not active_assets:
            logger.warning("No active market data found for dry-run cycle.")
            report = ExecutionCycleReport(
                timestamp=cycle_ts,
                cycle_id=cycle_id,
                universe_size=0,
                account_equity=self.initial_capital,
                settled_cash=self.initial_capital,
                active_signals={},
                target_weights={},
                executed_orders=[],
                status="NO_DATA",
                is_dry_run=True
            )
            self._save_telemetry(report)
            return report

        # 2. Extract Signals
        signals_dict = custom_signals or {}
        signals_summary = {}
        for sym, df in active_assets.items():
            if sym not in signals_dict:
                m_type = "BVC" if (".BVC" in sym or "BVC" in sym.upper()) else "US"
                sig_df = self.whale_detector.process_asset(df, market=m_type, symbol=sym)
                signals_dict[sym] = sig_df
            
            last_sig = signals_dict[sym].iloc[-1]
            signals_summary[sym] = {
                "signal": int(last_sig.get("Whale_Signal", 0)),
                "in_position": int(last_sig.get("In_Whale_Position", 0)),
                "mom_score": round(float(last_sig.get("Mom_Score", 0.0)), 4),
                "conviction": round(float(last_sig.get("Conviction_Multiplier", 1.0)), 2),
                "close": round(float(df["Close"].iloc[-1]), 2),
            }

        # 3. Portfolio Allocation
        allocation_res = self.allocator.compute_weights(signals_dict, benchmark_df=benchmark_df)
        latest_date = allocation_res.target_weights.index[-1]
        target_weights_series = allocation_res.target_weights.loc[latest_date]
        target_weights = target_weights_series[target_weights_series > 0.0].to_dict()

        # 4. Execute Orders & Update Simulator State
        account = self.trader.get_account_status()
        equity = account.get("equity", self.initial_capital)
        cash = account.get("cash", self.initial_capital)
        current_positions = self.trader.get_current_positions()

        executed_orders: List[Dict] = []

        # (a) Close positions no longer targeted
        for sym, qty in list(current_positions.items()):
            if sym not in target_weights or target_weights[sym] <= 0.0:
                res = self.trader.close_position(sym, qty=qty)
                executed_orders.append(res.to_dict())

        # (b) Open or adjust active target positions
        for sym, weight in target_weights.items():
            if sym in active_assets and not active_assets[sym].empty:
                last_price = float(active_assets[sym]["Close"].iloc[-1])
                # Calculate ATR if available for dynamic bracket stop
                atr = 0.0
                if "ATR" in signals_dict[sym].columns:
                    atr = float(signals_dict[sym]["ATR"].iloc[-1])
                elif len(active_assets[sym]) > 14:
                    highs = active_assets[sym]["High"]
                    lows = active_assets[sym]["Low"]
                    atr = float((highs - lows).rolling(14).mean().iloc[-1])

                res = self.trader.execute_decision(
                    symbol=sym,
                    action="BUY",
                    target_weight=weight,
                    current_price=last_price,
                    atr=atr
                )
                if res.status != "NO_ACTION":
                    executed_orders.append(res.to_dict())

        # 5. Build Report
        report = ExecutionCycleReport(
            timestamp=cycle_ts,
            cycle_id=cycle_id,
            universe_size=len(active_assets),
            account_equity=equity,
            settled_cash=cash,
            active_signals=signals_summary,
            target_weights=target_weights,
            executed_orders=executed_orders,
            status="SUCCESS",
            is_dry_run=not self.trader.is_authenticated
        )

        self._save_telemetry(report)
        return report

    def _save_telemetry(self, report: ExecutionCycleReport) -> None:
        """Persist execution snapshot and append to audit trail logs."""
        report_dict = report.to_dict()

        # 1. Current Signals Snapshot
        snap_path = self.output_dir / "current_signals_snapshot.json"
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)

        # 2. Live Execution Log (append mode)
        exec_log_path = self.output_dir / "live_execution_log.json"
        self._append_json_log(exec_log_path, report_dict)

        # 3. Live Dry Run Log
        dry_log_path = self.output_dir / "live_dry_run_log.json"
        self._append_json_log(dry_log_path, report_dict)

        logger.info(f"Telemetry saved to {snap_path} and {exec_log_path}.")

    def _append_json_log(self, filepath: Path, record: Dict) -> None:
        """Append record to a JSON list file safely."""
        history = []
        if filepath.exists():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    history = json.load(f)
                if not isinstance(history, list):
                    history = [history]
            except Exception:
                history = []

        history.append(record)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
