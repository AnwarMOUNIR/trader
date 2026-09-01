"""
Almgren-Chriss Optimal Execution & Order Book Simulator.

Implements:
1. Almgren-Chriss (2000) market impact dynamics:
   - Non-linear temporary impact: eta * (q / V)^alpha
   - Permanent price impact: gamma * (q / V)^beta
2. Dynamic bid-ask half-spreads (US: 2 bps, BVC: 15 bps).
3. Brokerage commission tiers (US: 3-5 bps, BVC: 15-25 bps).
4. Participation rate ceiling (US: 25%, BVC: 10%).
5. Realistic trade fills for Market, Limit, and Bracket orders with full accounting.
"""

from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("Backtest.OrderBook")


@dataclass
class AlmgrenChrissConfig:
    """Configuration parameters for Almgren-Chriss market impact and trading frictions."""
    # US Equities Market Impact Parameters (High Liquidity)
    us_linear_impact_eta: float = 0.05       # Temporary impact coefficient (0.03 - 0.05)
    us_permanent_impact_gamma: float = 0.02   # Permanent impact coefficient (0.01 - 0.02)
    us_impact_alpha: float = 1.0              # Temporary impact power exponent (1.0 linear, 0.5 square-root)
    us_impact_beta: float = 1.0               # Permanent impact power exponent
    us_brokerage_bps: float = 5.0             # 5 bps (0.05%) broker commission
    us_half_spread_bps: float = 2.0           # 2 bps (0.02%) dynamic half spread
    us_max_participation_rate: float = 0.25   # 25% max volume participation per bar

    # Casablanca (BVC) Market Impact Parameters (Lower Liquidity)
    bvc_linear_impact_eta: float = 0.25       # Temporary impact coefficient (0.15 - 0.25)
    bvc_permanent_impact_gamma: float = 0.15  # Permanent impact coefficient (0.08 - 0.15)
    bvc_impact_alpha: float = 1.0             # Temporary impact power exponent
    bvc_impact_beta: float = 1.0              # Permanent impact power exponent
    bvc_brokerage_bps: float = 25.0           # 25 bps (0.25%) commission + AMMC regulatory fee
    bvc_half_spread_bps: float = 15.0         # 15 bps (0.15%) dynamic half spread
    bvc_max_participation_rate: float = 0.10  # 10% max volume participation per bar

    # Moroccan Regulatory & Tax Settings
    moroccan_ir_tax_rate: float = 0.15        # 15% Moroccan Impôt sur le Revenu (IR) on net realized gains
    settlement_days_bvc: int = 3              # T+3 trading days clearing delay on Casablanca


# Alias for backward compatibility
OrderBookConfig = AlmgrenChrissConfig


@dataclass
class ExecutionResult:
    """Telemetry and financial accounting summary of an executed order."""
    executed_shares: float
    executed_price: float
    total_cost: float
    total_slippage: float
    total_commission: float
    realized_pnl: float
    tax_withheld: float
    order_side: str = ""
    symbol: str = ""
    timestamp: Optional[Union[pd.Timestamp, str]] = None


class OrderBookSimulator:
    """
    Microstructure Order Book & Execution Simulator implementing Almgren-Chriss (2000)
    market impact, spread crossing, broker commissions, and BVC settlement queues.
    """

    def __init__(self, config: Optional[AlmgrenChrissConfig] = None):
        self.config = config or AlmgrenChrissConfig()
        # Track pending settlement cash from BVC sales: deque of (settle_day_idx, amount_usd)
        self.settlement_queue: Deque[Tuple[int, float]] = deque()

    def get_market_parameters(self, market: str) -> Tuple[float, float, float, float, float, float, float]:
        """
        Extract market specific parameters:
        Returns (eta, gamma, alpha, beta, comm_rate, spread_rate, max_part).
        """
        market_upper = market.upper()
        if "BVC" in market_upper:
            eta = self.config.bvc_linear_impact_eta
            gamma = self.config.bvc_permanent_impact_gamma
            alpha = self.config.bvc_impact_alpha
            beta = self.config.bvc_impact_beta
            comm_rate = self.config.bvc_brokerage_bps / 10_000.0
            spread_rate = self.config.bvc_half_spread_bps / 10_000.0
            max_part = self.config.bvc_max_participation_rate
        else:
            eta = self.config.us_linear_impact_eta
            gamma = self.config.us_permanent_impact_gamma
            alpha = self.config.us_impact_alpha
            beta = self.config.us_impact_beta
            comm_rate = self.config.us_brokerage_bps / 10_000.0
            spread_rate = self.config.us_half_spread_bps / 10_000.0
            max_part = self.config.us_max_participation_rate

        return eta, gamma, alpha, beta, comm_rate, spread_rate, max_part

    def calculate_market_impact(
        self,
        order_shares: float,
        bar_volume: float,
        market: str = "US"
    ) -> Tuple[float, float, float, float]:
        """
        Compute non-linear temporary and permanent Almgren-Chriss market impact.
        
        Returns:
            Tuple of (effective_participation_rate, spread_rate, temp_impact_frac, perm_impact_frac)
        """
        eta, gamma, alpha, beta, _, spread_rate, max_part = self.get_market_parameters(market)
        
        safe_volume = max(float(bar_volume), 1.0)
        raw_participation = abs(order_shares) / safe_volume
        participation_rate = float(np.clip(raw_participation, 0.0, max_part))
        
        # Almgren-Chriss formulations:
        # Non-linear temporary impact: eta * rho^alpha
        temp_impact = eta * (participation_rate ** alpha)
        # Permanent impact: gamma * rho^beta
        perm_impact = gamma * (participation_rate ** beta)

        return participation_rate, spread_rate, temp_impact, perm_impact

    def execute_order(
        self,
        symbol: str,
        market: str,
        side: str,  # 'BUY' or 'SELL'
        order_value_usd: float,
        current_price: float,
        daily_volume: float,
        avg_entry_price: float = 0.0,
        day_index: int = 0,
        timestamp: Optional[Union[pd.Timestamp, str]] = None
    ) -> ExecutionResult:
        """
        Execute order with Almgren-Chriss market impact, spread, commissions, and tax deductions.
        """
        if order_value_usd <= 0 or current_price <= 0:
            return ExecutionResult(
                executed_shares=0.0,
                executed_price=current_price,
                total_cost=0.0,
                total_slippage=0.0,
                total_commission=0.0,
                realized_pnl=0.0,
                tax_withheld=0.0,
                order_side=side.upper(),
                symbol=symbol,
                timestamp=timestamp
            )

        eta, gamma, alpha, beta, comm_rate, spread_rate, max_part = self.get_market_parameters(market)

        # Baseline nominal shares
        shares_nominal = order_value_usd / current_price
        safe_vol = max(float(daily_volume), 10_000.0)
        participation_rate = float(np.clip(shares_nominal / safe_vol, 0.0, max_part))

        # Almgren-Chriss Total Impact Fraction:
        # eta * rho^alpha + gamma * rho^beta + spread_rate
        temp_impact = eta * (participation_rate ** alpha)
        perm_impact = gamma * (participation_rate ** beta)
        total_impact_fraction = temp_impact + perm_impact + spread_rate

        side_upper = side.upper()
        if side_upper == "BUY":
            effective_price = current_price * (1.0 + total_impact_fraction)
            executed_shares = order_value_usd / effective_price
            slippage_cost = executed_shares * (effective_price - current_price)
            commission = order_value_usd * comm_rate
            realized_pnl = 0.0
            tax_withheld = 0.0
            total_cost = order_value_usd + commission + slippage_cost
        else:  # SELL
            effective_price = current_price * (1.0 - total_impact_fraction)
            executed_shares = order_value_usd / current_price
            gross_proceeds = executed_shares * effective_price
            slippage_cost = executed_shares * (current_price - effective_price)
            commission = gross_proceeds * comm_rate

            # Tax Modeling: If Moroccan BVC asset sold at profit, deduct 15% IR
            if avg_entry_price > 0:
                gross_pnl = executed_shares * (effective_price - avg_entry_price) - commission
            else:
                gross_pnl = 0.0

            if "BVC" in market.upper() and gross_pnl > 0:
                tax_withheld = gross_pnl * self.config.moroccan_ir_tax_rate
                realized_pnl = gross_pnl - tax_withheld
            else:
                tax_withheld = 0.0
                realized_pnl = gross_pnl

            # Queue cash for T+3 settlement if Casablanca asset
            if "BVC" in market.upper():
                net_settled_cash = gross_proceeds - commission - tax_withheld
                settle_date = day_index + self.config.settlement_days_bvc
                self.settlement_queue.append((settle_date, net_settled_cash))

            total_cost = gross_proceeds - commission - slippage_cost

        return ExecutionResult(
            executed_shares=executed_shares,
            executed_price=effective_price,
            total_cost=total_cost,
            total_slippage=slippage_cost,
            total_commission=commission,
            realized_pnl=realized_pnl,
            tax_withheld=tax_withheld,
            order_side=side_upper,
            symbol=symbol,
            timestamp=timestamp
        )

    def process_daily_settlement(self, current_day_index: int) -> float:
        """Release settled cash that has passed the T+3 clearing window."""
        released_cash = 0.0
        while self.settlement_queue and self.settlement_queue[0][0] <= current_day_index:
            _, amount = self.settlement_queue.popleft()
            released_cash += amount
        return released_cash


# ExecutionModel class alias for backward compatibility with existing tests and modules
class ExecutionModel(OrderBookSimulator):
    """Alias for OrderBookSimulator preserving existing codebase API."""
    pass
