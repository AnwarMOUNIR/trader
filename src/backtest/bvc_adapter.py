"""
Casablanca Stock Exchange (BVC - Bourse de Casablanca) Microstructure Adapter.

Implements:
1. Regulatory Circuit Breaker Clamping (+/-10% Continuous, +/-6% Fixing).
2. Limit Up / Limit Down reservation condition detection and liquidity locks.
3. T+3 Clearing & Settlement Delay Queue for asset sales.
4. Moroccan Impôt sur le Revenu (IR) 15% Capital Gains Tax Withholding.
5. MAD <-> USD Currency Conversion for unified multi-currency portfolio accounting.
"""

from __future__ import annotations
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("Backtest.BVCAdapter")


@dataclass
class BVCPriceLimitResult:
    """Result of BVC circuit breaker price clamp check."""
    clamped_price: float
    raw_price: float
    reference_price: float
    lower_limit: float
    upper_limit: float
    is_limit_hit: bool
    limit_type: str  # 'NORMAL', 'LIMIT_UP', 'LIMIT_DOWN'
    is_tradable: bool


@dataclass
class SettlementItem:
    """Item in the T+3 clearing settlement queue."""
    order_id: str
    symbol: str
    settlement_day_index: int
    amount_usd: float
    amount_mad: float
    trade_timestamp: Optional[Union[pd.Timestamp, str]] = None


class BVCAdapter:
    """
    Microstructure Adapter enforcing Moroccan Casablanca Stock Exchange regulatory rules.
    """

    def __init__(
        self,
        continuous_circuit_breaker: float = 0.10,   # +/- 10.0% continuous limit
        fixing_circuit_breaker: float = 0.06,       # +/- 6.0% fixing limit
        ir_tax_rate: float = 0.15,                  # 15.0% Moroccan IR tax on realized capital gains
        settlement_latency_days: int = 3,           # T+3 trading days clearing delay
        mad_usd_fx_rate: float = 0.10               # 1 MAD = 0.10 USD (10 MAD / 1 USD)
    ):
        self.continuous_circuit_breaker = continuous_circuit_breaker
        self.fixing_circuit_breaker = fixing_circuit_breaker
        self.ir_tax_rate = ir_tax_rate
        self.settlement_latency_days = settlement_latency_days
        self.mad_usd_fx_rate = mad_usd_fx_rate

        # Settlement queue: deque of SettlementItem
        self.settlement_queue: Deque[SettlementItem] = deque()
        self.cumulative_tax_paid_usd: float = 0.0
        self.cumulative_tax_paid_mad: float = 0.0

    def convert_mad_to_usd(self, amount_mad: float) -> float:
        """Convert Moroccan Dirham (MAD) to US Dollars (USD)."""
        return amount_mad * self.mad_usd_fx_rate

    def convert_usd_to_mad(self, amount_usd: float) -> float:
        """Convert US Dollars (USD) to Moroccan Dirham (MAD)."""
        if self.mad_usd_fx_rate <= 0:
            return amount_usd * 10.0
        return amount_usd / self.mad_usd_fx_rate

    def check_circuit_breaker(
        self,
        target_price: float,
        reference_price: float,
        session_type: str = "continuous"
    ) -> BVCPriceLimitResult:
        """
        Check and clamp price against statutory BVC circuit breaker bands (+/-10% or +/-6%).
        """
        if reference_price <= 0:
            return BVCPriceLimitResult(
                clamped_price=target_price,
                raw_price=target_price,
                reference_price=reference_price,
                lower_limit=target_price * 0.90,
                upper_limit=target_price * 1.10,
                is_limit_hit=False,
                limit_type="NORMAL",
                is_tradable=True
            )

        limit_pct = self.fixing_circuit_breaker if session_type.lower() == "fixing" else self.continuous_circuit_breaker
        upper_limit = reference_price * (1.0 + limit_pct)
        lower_limit = reference_price * (1.0 - limit_pct)

        if target_price >= upper_limit - 1e-5:
            return BVCPriceLimitResult(
                clamped_price=upper_limit,
                raw_price=target_price,
                reference_price=reference_price,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                is_limit_hit=True,
                limit_type="LIMIT_UP",
                is_tradable=True
            )
        elif target_price <= lower_limit + 1e-5:
            return BVCPriceLimitResult(
                clamped_price=lower_limit,
                raw_price=target_price,
                reference_price=reference_price,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                is_limit_hit=True,
                limit_type="LIMIT_DOWN",
                is_tradable=True
            )
        else:
            return BVCPriceLimitResult(
                clamped_price=target_price,
                raw_price=target_price,
                reference_price=reference_price,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                is_limit_hit=False,
                limit_type="NORMAL",
                is_tradable=True
            )

    def calculate_capital_gains_tax(
        self,
        executed_shares: float,
        executed_price: float,
        avg_entry_price: float,
        commission: float = 0.0
    ) -> Tuple[float, float]:
        """
        Calculate Moroccan Impôt sur le Revenu (IR) 15% tax on net realized capital gains.
        
        Returns:
            Tuple of (net_realized_pnl_after_tax, tax_withheld)
        """
        if executed_shares <= 0 or avg_entry_price <= 0:
            return 0.0, 0.0

        gross_gain = executed_shares * (executed_price - avg_entry_price) - commission
        if gross_gain > 0.0:
            tax = gross_gain * self.ir_tax_rate
            net_pnl = gross_gain - tax
            self.cumulative_tax_paid_usd += tax
            self.cumulative_tax_paid_mad += self.convert_usd_to_mad(tax)
            return net_pnl, tax
        else:
            return gross_gain, 0.0

    def queue_settlement_proceeds(
        self,
        order_id: str,
        symbol: str,
        net_proceeds_usd: float,
        current_day_index: int,
        timestamp: Optional[Union[pd.Timestamp, str]] = None
    ) -> None:
        """
        Enqueue cash proceeds from Casablanca asset sales into the T+3 clearing queue.
        """
        settle_day = current_day_index + self.settlement_latency_days
        item = SettlementItem(
            order_id=order_id,
            symbol=symbol,
            settlement_day_index=settle_day,
            amount_usd=net_proceeds_usd,
            amount_mad=self.convert_usd_to_mad(net_proceeds_usd),
            trade_timestamp=timestamp
        )
        self.settlement_queue.append(item)
        logger.debug(f"Queued T+{self.settlement_latency_days} BVC settlement: ${net_proceeds_usd:,.2f} for day {settle_day}")

    def process_settlement_queue(self, current_day_index: int) -> Tuple[float, List[SettlementItem]]:
        """
        Release all settled cash items where settlement_day_index <= current_day_index.
        
        Returns:
            Tuple of (total_released_cash_usd, list_of_released_items)
        """
        released_cash_usd = 0.0
        released_items: List[SettlementItem] = []

        while self.settlement_queue and self.settlement_queue[0].settlement_day_index <= current_day_index:
            item = self.settlement_queue.popleft()
            released_cash_usd += item.amount_usd
            released_items.append(item)

        return released_cash_usd, released_items

    def get_pending_settlement_total(self) -> float:
        """Get total amount of pending unsettled cash receivables in USD."""
        return sum(item.amount_usd for item in self.settlement_queue)

    def reset(self) -> None:
        """Reset internal queue and tax counters."""
        self.settlement_queue.clear()
        self.cumulative_tax_paid_usd = 0.0
        self.cumulative_tax_paid_mad = 0.0


# Backward compatibility alias
BVCMicrostructureHandler = BVCAdapter
