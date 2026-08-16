"""
Deflated Sharpe Ratio (DSR) Module.
Calculates statistical significance of Sharpe Ratio adjusting for non-normality and selection bias.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_deflated_sharpe_ratio(
    daily_returns: pd.Series,
    n_trials: int = 20,
    benchmark_sharpe: float = 0.0
) -> float:
    """
    Computes the Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).
    Accounts for skewness, kurtosis, and multiple testing selection bias.
    """
    returns = daily_returns.dropna().values
    n = len(returns)
    if n < 30:
        return 0.50

    # Calculate empirical moments
    mean_ret = np.mean(returns)
    std_ret = np.std(returns, ddof=1)
    if std_ret <= 1e-8:
        return 0.0

    sr_annual = (mean_ret / std_ret) * np.sqrt(252.0)
    sr_daily = mean_ret / std_ret

    skewness = float(pd.Series(returns).skew())
    kurtosis = float(pd.Series(returns).kurtosis())  # Excess kurtosis

    # Variance of Sharpe estimator under non-normality
    var_sr = (1.0 - skewness * sr_daily + ((kurtosis + 2.0) / 4.0) * (sr_daily ** 2)) / (n - 1)
    std_sr = np.sqrt(max(var_sr, 1e-8))

    # Expected maximum Sharpe under null hypothesis of N independent trials
    # Euler-Mascheroni constant approximation: E[max_N] ~ (1 - gamma)*Z^-1[1 - 1/N] + gamma*Z^-1[1 - 1/(N*e)]
    euler_gamma = 0.5772156649
    z_term = norm.ppf(1.0 - 1.0 / n_trials)
    z_term_e = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    expected_max_sr = ((1.0 - euler_gamma) * z_term + euler_gamma * z_term_e) * np.sqrt(252.0 / n)

    # Standardized test statistic
    dsr_stat = (sr_annual - expected_max_sr) / (std_sr * np.sqrt(252.0))
    dsr_pvalue = norm.cdf(dsr_stat)

    return float(np.clip(dsr_pvalue, 0.0, 1.0))
