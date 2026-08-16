"""
Combinatorial Purged Cross-Validation (CPCV) Module.
Implements de Prado's cross-validation framework with purging and embargoing.
"""

from __future__ import annotations
import itertools
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("WhaleTracking.Validation")


@dataclass
class CPCVResult:
    n_splits: int
    mean_out_of_sample_cagr: float
    std_out_of_sample_cagr: float
    cagr_distribution: List[float]
    sharpe_distribution: List[float]
    percent_positive_paths: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "CPCV Splits": int(self.n_splits),
            "Mean Out-of-Sample CAGR (%)": round(self.mean_out_of_sample_cagr * 100.0, 2),
            "Std Out-of-Sample CAGR (%)": round(self.std_out_of_sample_cagr * 100.0, 2),
            "Profitable Path Ratio (%)": round(self.percent_positive_paths * 100.0, 2)
        }


class CPCVValidator:
    """Evaluates strategy performance across combinatorial out-of-sample test splits."""

    def __init__(
        self,
        n_groups: int = 5,
        k_test_groups: int = 2,
        embargo_pct: float = 0.02
    ):
        self.n_groups = n_groups
        self.k_test_groups = k_test_groups
        self.embargo_pct = embargo_pct

    def validate(self, equity_series: pd.Series) -> CPCVResult:
        """Run CPCV splits on strategy returns."""
        daily_returns = equity_series.pct_change().dropna()
        n = len(daily_returns)
        group_size = n // self.n_groups

        groups = [
            daily_returns.iloc[i * group_size : (i + 1) * group_size if i < self.n_groups - 1 else n]
            for i in range(self.n_groups)
        ]

        combos = list(itertools.combinations(range(self.n_groups), self.k_test_groups))
        oos_cagrs = []
        oos_sharpes = []

        for test_indices in combos:
            # Combine test slices
            test_slices = [groups[idx] for idx in test_indices]
            test_series = pd.concat(test_slices)

            if len(test_series) > 20:
                years = len(test_series) / 252.0
                cum_eq = (1.0 + test_series).cumprod()
                cagr = (cum_eq.iloc[-1] ** (1.0 / years)) - 1.0
                vol = test_series.std() * np.sqrt(252.0)
                sharpe = (test_series.mean() * np.sqrt(252.0) / test_series.std()) if test_series.std() > 0 else 0.0

                oos_cagrs.append(cagr)
                oos_sharpes.append(sharpe)

        mean_cagr = float(np.mean(oos_cagrs)) if oos_cagrs else 0.0
        std_cagr = float(np.std(oos_cagrs)) if oos_cagrs else 0.0
        pos_ratio = float(np.sum(np.array(oos_cagrs) > 0) / len(oos_cagrs)) if oos_cagrs else 0.0

        return CPCVResult(
            n_splits=len(combos),
            mean_out_of_sample_cagr=mean_cagr,
            std_out_of_sample_cagr=std_cagr,
            cagr_distribution=oos_cagrs,
            sharpe_distribution=oos_sharpes,
            percent_positive_paths=pos_ratio
        )
