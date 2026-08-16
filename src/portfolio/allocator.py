"""
Risk-Parity & Volatility-Managed Portfolio Weighting Engine.
Implements inverse-variance risk scaling, conviction boosts, position caps, and momentum crash protection.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger("WhaleTracking.Portfolio")


@dataclass
class PortfolioAllocationResult:
    target_weights: pd.DataFrame
    portfolio_volatility: pd.Series
    cash_weight: pd.Series


class PortfolioAllocator:
    """Calculates risk-managed target portfolio weights across assets over time."""

    def __init__(
        self,
        target_annual_vol: float = 0.32,      # 32% annualized portfolio target vol
        max_single_weight: float = 0.34,      # Max 34% in top 3 alpha leaders
        min_weight_threshold: float = 0.04,   # Drop positions smaller than 4%
        holding_period_days: int = 25,        # Smart money swing holding period
        max_total_exposure: float = 1.00      # 100% max equity exposure (no margin)
    ):
        self.target_annual_vol = target_annual_vol
        self.max_single_weight = max_single_weight
        self.min_weight_threshold = min_weight_threshold
        self.holding_period_days = holding_period_days
        self.max_total_exposure = max_total_exposure

    def compute_weights(
        self,
        signals_dict: Dict[str, pd.DataFrame],
        benchmark_df: Optional[pd.DataFrame] = None
    ) -> PortfolioAllocationResult:
        """
        Compute daily target portfolio weights using inverse-variance risk parity
        weighted by whale momentum conviction scores and macro regime filter.
        """
        symbols = list(signals_dict.keys())
        if not symbols:
            raise ValueError("No assets provided to portfolio allocator.")

        # Find common date index across all assets
        common_dates = signals_dict[symbols[0]].index
        for s in symbols[1:]:
            common_dates = common_dates.intersection(signals_dict[s].index)
        
        if len(common_dates) == 0:
            common_dates = sorted(list(set.union(*[set(df.index) for df in signals_dict.values()])))

        dates = common_dates
        raw_weights_df = pd.DataFrame(0.0, index=dates, columns=symbols)
        vol_df = pd.DataFrame(0.20, index=dates, columns=symbols)

        # Macro Regime: Bullish when benchmark is above 200-day SMA
        if benchmark_df is not None and not benchmark_df.empty:
            bm_close = benchmark_df.loc[dates.intersection(benchmark_df.index), "Close"].reindex(dates).ffill().bfill()
            bm_sma200 = bm_close.rolling(200, min_periods=40).mean()
            macro_bullish = (bm_close >= bm_sma200 * 0.98).astype(int)
        else:
            macro_bullish = pd.Series(1, index=dates)

        # 1. Calculate unnormalized weights per asset: w_i = (Score * Conviction) / variance_i
        for sym, df_orig in signals_dict.items():
            df = df_orig.reindex(dates).ffill().bfill()
            
            # Active trend and accumulation signal
            in_pos = df.get("In_Whale_Position", df["Whale_Signal"]).fillna(0)
            conviction = df["Conviction_Multiplier"]
            mom_score = df["Mom_Score"]
            vol = df["Vol_20D"].clip(lower=0.10)
            vol_df[sym] = vol

            # Base score with trend requirement
            active = in_pos > 0
            base_score = np.maximum(mom_score, 0.20)
            
            unnorm_weight = np.where(
                active,
                (base_score * conviction) / (vol ** 1.1),
                0.0
            )
            raw_weights_df[sym] = unnorm_weight

        # 2. Hysteresis Rank Selection & Sticky Position Holding (Zero Rank-Churn)
        k_entry = 3
        k_hold = 7
        max_active_positions = 3

        normalized_weights = pd.DataFrame(0.0, index=dates, columns=symbols)
        current_active_portfolio = set()

        for t in dates:
            is_bull = macro_bullish.loc[t] == 1
            max_exp = self.max_total_exposure if is_bull else 0.20  # De-leverage to cash in bear regimes

            row_weights = raw_weights_df.loc[t]
            ranked_eligible = row_weights[row_weights > 0].sort_values(ascending=False)
            top_entry_candidates = set(ranked_eligible.head(k_entry).index)
            top_hold_candidates = set(ranked_eligible.head(k_hold).index)

            # Keep existing positions that are still in hold candidates and active
            surviving_positions = {s for s in current_active_portfolio if s in top_hold_candidates and row_weights[s] > 0}

            # In bull markets, add new candidates up to max_active_positions
            if is_bull:
                for candidate in ranked_eligible.index:
                    if len(surviving_positions) >= max_active_positions:
                        break
                    if candidate in top_entry_candidates:
                        surviving_positions.add(candidate)

            current_active_portfolio = surviving_positions

            if current_active_portfolio and max_exp > 0:
                active_list = list(current_active_portfolio)
                # Volatility-adjusted inverse risk weights for active assets
                active_vols = vol_df.loc[t, active_list]
                inv_vol_w = (1.0 / (active_vols ** 1.1))
                w_norm = (inv_vol_w / inv_vol_w.sum()) * max_exp

                # Cap single weight (e.g. 28% max)
                w_norm = np.minimum(w_norm, self.max_single_weight)
                if w_norm.sum() > 0:
                    w_norm = (w_norm / w_norm.sum()) * max_exp

                for s, weight in zip(active_list, w_norm):
                    normalized_weights.loc[t, s] = weight

        # 3. Volatility Scaling Overlay (Daniel & Moskowitz / Risk-managed Momentum)
        est_port_vol = np.sqrt((normalized_weights ** 2 * vol_df ** 2).sum(axis=1))
        est_port_vol = est_port_vol.replace(0, np.nan).fillna(self.target_annual_vol)

        vol_scale = (self.target_annual_vol / est_port_vol).clip(0.8, 1.2)
        scaled_weights = normalized_weights.multiply(vol_scale, axis=0)

        # Strictly enforce single-asset max weight ceiling
        scaled_weights = scaled_weights.clip(upper=self.max_single_weight)

        # Cap total exposure
        total_exp = scaled_weights.sum(axis=1)
        over_exposure = total_exp > self.max_total_exposure
        scaled_weights[over_exposure] = scaled_weights[over_exposure].div(total_exp[over_exposure], axis=0)

        cash_weight = 1.0 - scaled_weights.sum(axis=1)

        return PortfolioAllocationResult(
            target_weights=scaled_weights,
            portfolio_volatility=est_port_vol,
            cash_weight=cash_weight
        )
