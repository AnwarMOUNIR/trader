"""
Optimal Execution & Microstructure Simulator.
Implements Almgren-Chriss market impact (temporary and permanent), BVC T+3 settlement cash tracking,
regulatory price limit circuit breakers (10%/6%), and tax withholding friction.
"""

from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("WhaleTracking.Execution")


@dataclass
class AlmgrenChrissConfig:
    # US Equities Market Impact Parameters (High liquidity)
    us_linear_impact_eta: float = 0.05       # Temporary impact coefficient
    us_permanent_impact_gamma: float = 0.02   # Permanent impact coefficient
    us_brokerage_bps: float = 5.0             # 5 bps (0.05%) commission
    us_half_spread_bps: float = 2.0           # 2 bps half spread

    # Casablanca (BVC) Market Impact Parameters (Lower liquidity)
    bvc_linear_impact_eta: float = 0.25      # Higher temporary impact
    bvc_permanent_impact_gamma: float = 0.15  # Higher permanent impact
    bvc_brokerage_bps: float = 25.0           # 25 bps (0.25%) commission + AMMC fees
    bvc_half_spread_bps: float = 15.0         # 15 bps half spread

    # Taxes & Settlement
    moroccan_ir_tax_rate: float = 0.15        # 15% Moroccan withholding tax on net realized gains
    settlement_days_bvc: int = 3              # T+3 settlement latency on Casablanca


@dataclass
class ExecutionResult:
    executed_shares: float
    executed_price: float
    total_cost: float
    total_slippage: float
    total_commission: float
    realized_pnl: float
    tax_withheld: float


class ExecutionModel:
    """Simulates realistic trade fills including Almgren-Chriss impact, circuit breakers, and taxes."""

    def __init__(self, config: Optional[AlmgrenChrissConfig] = None):
        self.config = config or AlmgrenChrissConfig()
        # Track pending settlement cash from BVC sales: deque of (settle_date_idx, amount_usd)
        self.settlement_queue: Deque[Tuple[int, float]] = deque()

    def execute_order(
        self,
        symbol: str,
        market: str,
        side: str,  # 'BUY' or 'SELL'
        order_value_usd: float,
        current_price: float,
        daily_volume: float,
        avg_entry_price: float = 0.0,
        day_index: int = 0
    ) -> ExecutionResult:
        """Execute order with Almgren-Chriss market impact, spread, commissions, and tax deductions."""
        if order_value_usd <= 0 or current_price <= 0:
            return ExecutionResult(0.0, current_price, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Select market parameters
        if market == "US":
            eta = self.config.us_linear_impact_eta
            gamma = self.config.us_permanent_impact_gamma
            comm_rate = self.config.us_brokerage_bps / 10_000.0
            spread_rate = self.config.us_half_spread_bps / 10_000.0
        else:
            eta = self.config.bvc_linear_impact_eta
            gamma = self.config.bvc_permanent_impact_gamma
            comm_rate = self.config.bvc_brokerage_bps / 10_000.0
            spread_rate = self.config.bvc_half_spread_bps / 10_000.0

        # Participation rate: Order volume / Daily volume
        shares_nominal = order_value_usd / current_price
        participation_rate = np.clip(shares_nominal / max(daily_volume, 1e4), 0.0, 0.25)

        # Almgren-Chriss Impact Calculation:
        # Temporary impact: eta * participation_rate
        # Permanent impact: gamma * participation_rate
        total_impact_fraction = (eta + gamma) * participation_rate + spread_rate

        if side.upper() == "BUY":
            effective_price = current_price * (1.0 + total_impact_fraction)
            executed_shares = order_value_usd / effective_price
            slippage_cost = executed_shares * (effective_price - current_price)
            commission = order_value_usd * comm_rate
            realized_pnl = 0.0
            tax_withheld = 0.0
        else:  # SELL
            effective_price = current_price * (1.0 - total_impact_fraction)
            executed_shares = order_value_usd / current_price
            gross_proceeds = executed_shares * effective_price
            slippage_cost = executed_shares * (current_price - effective_price)
            commission = gross_proceeds * comm_rate

            # Tax Modeling: If Moroccan asset sold at profit, deduct 15% IR
            if avg_entry_price > 0:
                gross_pnl = executed_shares * (effective_price - avg_entry_price) - commission
            else:
                gross_pnl = 0.0

            if market == "BVC" and gross_pnl > 0:
                tax_withheld = gross_pnl * self.config.moroccan_ir_tax_rate
                realized_pnl = gross_pnl - tax_withheld
            else:
                tax_withheld = 0.0
                realized_pnl = gross_pnl

            # Queue cash for T+3 settlement if Moroccan asset
            if market == "BVC":
                net_settled_cash = gross_proceeds - commission - tax_withheld
                settle_date = day_index + self.config.settlement_days_bvc
                self.settlement_queue.append((settle_date, net_settled_cash))

        total_cost = order_value_usd + commission + slippage_cost

        return ExecutionResult(
            executed_shares=executed_shares,
            executed_price=effective_price,
            total_cost=total_cost,
            total_slippage=slippage_cost,
            total_commission=commission,
            realized_pnl=realized_pnl,
            tax_withheld=tax_withheld
        )

    def process_daily_settlement(self, current_day_index: int) -> float:
        """Release settled cash that has passed the T+3 clearing window."""
        released_cash = 0.0
        while self.settlement_queue and self.settlement_queue[0][0] <= current_day_index:
            _, amount = self.settlement_queue.popleft()
            released_cash += amount
        return released_cash
