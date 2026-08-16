"""
Monte Carlo Simulation & Block Bootstrapping Engine.
Evaluates the statistical probability of achieving >=30% CAGR under stochastic path variations.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("WhaleTracking.Validation")


@dataclass
class MonteCarloResult:
    n_simulations: int
    prob_cagr_above_target: float    # Statistical confidence (e.g. 92.4%)
    p_value: float                   # Empirical p-value
    target_cagr: float
    cagr_5th_percentile: float
    cagr_50th_percentile: float
    cagr_95th_percentile: float
    simulated_cagrs: np.ndarray
    simulated_max_dds: np.ndarray
    prob_ruin: float                 # Prob of Drawdown > 30%

    def to_dict(self) -> Dict[str, float]:
        return {
            "Simulations": int(self.n_simulations),
            "Target CAGR (%)": round(self.target_cagr * 100.0, 2),
            "Confidence (P >= Target) (%)": round(self.prob_cagr_above_target * 100.0, 2),
            "Empirical p-value": round(self.p_value, 4),
            "5th Percentile CAGR (%)": round(self.cagr_5th_percentile * 100.0, 2),
            "Median (50th) CAGR (%)": round(self.cagr_50th_percentile * 100.0, 2),
            "95th Percentile CAGR (%)": round(self.cagr_95th_percentile * 100.0, 2),
            "Risk of Ruin (MDD > 30%) (%)": round(self.prob_ruin * 100.0, 2)
        }


class MonteCarloSimulator:
    """Performs circular block bootstrapping on strategy returns to quantify statistical confidence."""

    def __init__(
        self,
        n_simulations: int = 2000,
        block_size: int = 10,        # 10-day blocks preserve autocorrelation
        target_cagr: float = 0.30,   # 30% annualized threshold
        random_seed: int = 42
    ):
        self.n_simulations = n_simulations
        self.block_size = block_size
        self.target_cagr = target_cagr
        self.random_seed = random_seed

    def run(self, equity_series: pd.Series) -> MonteCarloResult:
        """Run block bootstrap simulations on the historical equity curve."""
        daily_returns = equity_series.pct_change().dropna().values
        n_days = len(daily_returns)
        years = n_days / 252.0

        if n_days < 50:
            raise ValueError("Insufficient return data for Monte Carlo simulation.")

        rng = np.random.default_rng(self.random_seed)
        num_blocks = int(np.ceil(n_days / self.block_size))

        simulated_cagrs = np.zeros(self.n_simulations)
        simulated_max_dds = np.zeros(self.n_simulations)

        # Generate block bootstrap paths
        for i in range(self.n_simulations):
            # Pick random block start indices
            start_indices = rng.integers(0, n_days - self.block_size + 1, size=num_blocks)
            sampled_returns = []
            for idx in start_indices:
                sampled_returns.extend(daily_returns[idx : idx + self.block_size])
            sampled_returns = np.array(sampled_returns[:n_days])

            # Compound simulated path
            sim_equity = np.cumprod(1.0 + sampled_returns)
            sim_cagr = (sim_equity[-1] ** (1.0 / years)) - 1.0
            simulated_cagrs[i] = sim_cagr

            # Max drawdown of path
            cum_max = np.maximum.accumulate(sim_equity)
            dd = (sim_equity - cum_max) / cum_max
            simulated_max_dds[i] = abs(np.min(dd))

        # Statistical significance calculation
        success_count = np.sum(simulated_cagrs >= self.target_cagr)
        prob_target = success_count / self.n_simulations
        p_val = 1.0 - prob_target

        prob_ruin = np.sum(simulated_max_dds >= 0.30) / self.n_simulations

        p5 = float(np.percentile(simulated_cagrs, 5))
        p50 = float(np.percentile(simulated_cagrs, 50))
        p95 = float(np.percentile(simulated_cagrs, 95))

        logger.info(
            f"Monte Carlo ({self.n_simulations} runs): "
            f"P(CAGR >= {self.target_cagr*100:.1f}%) = {prob_target*100:.2f}%, "
            f"p-value = {p_val:.4f}, Median CAGR = {p50*100:.2f}%"
        )

        return MonteCarloResult(
            n_simulations=self.n_simulations,
            prob_cagr_above_target=prob_target,
            p_value=p_val,
            target_cagr=self.target_cagr,
            cagr_5th_percentile=p5,
            cagr_50th_percentile=p50,
            cagr_95th_percentile=p95,
            simulated_cagrs=simulated_cagrs,
            simulated_max_dds=simulated_max_dds,
            prob_ruin=prob_ruin
        )
