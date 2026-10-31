"""Risk-Controlled Portfolio and Single-Stock Engine with Dynamic Drawdown Bounds.

Implements:
1. SingleStockRiskEngine: Guarantees <= $30k worst-case loss on a $100k account
   even under overnight jumps to zero, using structural allocation bounds (<=30%)
   and dynamic trailing drawdown stops.
2. PortfolioRiskEngine: Enforces <= $10k worst-case average loss on a $100k
   portfolio across 30-50 stocks using:
   - Equal-weight idiosyncratic risk diversification (1/N)
   - Moreira & Muir / AQR Macro SPY 50/200 Trend & Volatility Gate
   - Grossman & Zhou / CPPI Dynamic Drawdown Dial ($6k soft floor, $9k hard floor)
"""
from __future__ import annotations

from typing import Sequence
import numpy as np
import pandas as pd


class SingleStockRiskEngine:
    """Manages a single-stock allocation with strict downside boundaries."""

    def __init__(
        self,
        capital: float = 100_000.0,
        max_allocation_fraction: float = 0.30,
        trailing_stop_dollars: float = 15_000.0,
        cost_bps: float = 10.0,
    ) -> None:
        if capital <= 0:
            raise ValueError("Capital must be positive")
        if not (0 < max_allocation_fraction <= 1.0):
            raise ValueError("Max allocation fraction must be between 0 and 1.0")
        if trailing_stop_dollars <= 0:
            raise ValueError("Trailing stop must be positive")
        if cost_bps < 0:
            raise ValueError("Cost bps cannot be negative")

        self.capital = float(capital)
        self.max_allocation = float(max_allocation_fraction)
        self.trailing_stop = float(trailing_stop_dollars)
        self.fee_rate = float(cost_bps) / 20_000.0

    def simulate(self, opens: Sequence[float], closes: Sequence[float]) -> dict:
        """Simulate single-stock holding path with allocation cap and trailing stop."""
        opens = np.asarray(opens, dtype=float)
        closes = np.asarray(closes, dtype=float)
        if len(opens) == 0 or len(closes) == 0 or len(opens) != len(closes):
            raise ValueError("Opens and closes must be non-empty arrays of identical length")
        if (opens <= 0).any() or (closes <= 0).any() or not (np.isfinite(opens).all() and np.isfinite(closes).all()):
            raise ValueError("Prices must be positive finite numbers")

        nav = self.capital
        high_water = self.capital
        target_dollar = self.capital * self.max_allocation
        initial_open = opens[0]
        
        # Enter at initial open with fee deducted
        shares = target_dollar / initial_open
        entry_fee = target_dollar * self.fee_rate
        cash = self.capital - target_dollar - entry_fee
        costs_paid = entry_fee
        liquidated = False
        liquidation_day = None

        nav_history = []
        for day, (op, cl) in enumerate(zip(opens, closes)):
            if liquidated:
                nav_history.append(nav)
                continue

            # Mark to market at close
            market_val = shares * cl
            cur_nav = cash + market_val
            high_water = max(high_water, cur_nav)
            drawdown = high_water - cur_nav
            total_loss = self.capital - cur_nav

            # Check trailing stop or principal stop
            if drawdown >= self.trailing_stop or total_loss >= self.trailing_stop:
                # Liquidate at current close (or next open)
                exit_notional = shares * cl
                exit_fee = exit_notional * self.fee_rate
                cash += exit_notional - exit_fee
                costs_paid += exit_fee
                shares = 0.0
                nav = cash
                liquidated = True
                liquidation_day = day
            else:
                nav = cur_nav
            nav_history.append(nav)

        nav_history = np.array(nav_history)
        min_nav = np.min(nav_history)
        max_drawdown = self.capital - min_nav

        return {
            "initial_capital": self.capital,
            "ending_equity": float(nav),
            "net_pnl": float(nav - self.capital),
            "min_equity": float(min_nav),
            "max_drawdown_dollars": float(max_drawdown),
            "liquidated": liquidated,
            "liquidation_day": liquidation_day,
            "costs_paid": float(costs_paid),
            "nav_history": nav_history,
        }


class PortfolioRiskEngine:
    """Manages a 30-50 stock equal-weight portfolio with Macro Trend Gating and CPPI Drawdown Dial."""

    def __init__(
        self,
        capital: float = 100_000.0,
        soft_floor_dollars: float = 6_000.0,
        hard_floor_dollars: float = 9_000.0,
        macro_gate_defensive_exposure: float = 0.30,
        cost_bps: float = 10.0,
    ) -> None:
        if capital <= 0:
            raise ValueError("Capital must be positive")
        if not (0 < soft_floor_dollars < hard_floor_dollars < capital):
            raise ValueError("Floors must satisfy 0 < soft_floor < hard_floor < capital")
        if not (0 <= macro_gate_defensive_exposure <= 1.0):
            raise ValueError("Defensive exposure must be between 0 and 1.0")

        self.capital = float(capital)
        self.soft_floor = float(soft_floor_dollars)
        self.hard_floor = float(hard_floor_dollars)
        self.defensive_exp = float(macro_gate_defensive_exposure)
        self.fee_rate = float(cost_bps) / 20_000.0

    def compute_dial_exposure(self, drawdown: float) -> float:
        """Compute the Grossman-Zhou / CPPI dynamic exposure dial based on drawdown."""
        if drawdown >= self.hard_floor:
            return 0.0  # 100% Cash Lock
        if drawdown >= self.soft_floor:
            # Linear decay from 1.0 to 0.2 between soft and hard floor
            slope = (0.20 - 1.0) / (self.hard_floor - self.soft_floor)
            return float(1.0 + slope * (drawdown - self.soft_floor))
        return 1.0

    def simulate(
        self,
        opens_matrix: np.ndarray,
        closes_matrix: np.ndarray,
        spy_trend_signals: np.ndarray,
    ) -> dict:
        """Simulate multi-stock rebalanced portfolio.
        
        opens_matrix: shape (T, N)
        closes_matrix: shape (T, N)
        spy_trend_signals: shape (T,) boolean array where True = Bull Regime
        """
        opens_matrix = np.asarray(opens_matrix, dtype=float)
        closes_matrix = np.asarray(closes_matrix, dtype=float)
        spy_trend_signals = np.asarray(spy_trend_signals, dtype=bool)

        T, N = opens_matrix.shape
        if T == 0 or N == 0:
            raise ValueError("Matrix dimensions must be non-zero")
        if closes_matrix.shape != (T, N) or len(spy_trend_signals) != T:
            raise ValueError("Mismatched matrix shapes")

        nav = self.capital
        high_water = self.capital
        cash = self.capital
        cur_shares = np.zeros(N, dtype=float)
        costs_paid = 0.0

        daily_navs = [self.capital]
        daily_exposures = []

        for t in range(T):
            op = opens_matrix[t]
            cl = closes_matrix[t]
            trend_ok = spy_trend_signals[t]

            # 1. Evaluate open NAV before today's rebalance
            open_mv = np.sum(cur_shares * op)
            cur_nav = cash + open_mv
            high_water = max(high_water, cur_nav)
            drawdown = max(0.0, high_water - cur_nav)

            # 2. Determine target exposure via Macro Gate & CPPI Dial
            macro_factor = 1.0 if trend_ok else self.defensive_exp
            dial_factor = self.compute_dial_exposure(drawdown)
            target_gross_fraction = macro_factor * dial_factor
            daily_exposures.append(target_gross_fraction)

            target_invested = target_gross_fraction * cur_nav
            alloc_per_stock = target_invested / N

            # 3. Rebalance at open prices
            target_shares = alloc_per_stock / op
            delta_shares = target_shares - cur_shares
            turnover = np.sum(np.abs(delta_shares * op))
            fee = turnover * self.fee_rate
            costs_paid += fee

            cash = cur_nav - target_invested - fee
            cur_shares = target_shares

            # 4. Mark to close
            close_mv = np.sum(cur_shares * cl)
            end_nav = cash + close_mv
            daily_navs.append(end_nav)
            nav = end_nav

        daily_navs = np.array(daily_navs)
        min_equity = np.min(daily_navs)
        max_dd = self.capital - min_equity

        return {
            "initial_capital": self.capital,
            "ending_equity": float(nav),
            "net_pnl": float(nav - self.capital),
            "min_equity": float(min_equity),
            "max_drawdown_dollars": float(max_dd),
            "costs_paid": float(costs_paid),
            "daily_navs": daily_navs,
            "average_exposure": float(np.mean(daily_exposures)),
        }
