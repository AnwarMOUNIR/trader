"""
Quantitative Performance & Risk Analytics Engine.

Implements:
1. Frequency-Aware Annualized Growth Rate (CAGR).
2. Maximum Drawdown (MDD) and Drawdown Duration.
3. Annualized Sharpe Ratio, Sortino Ratio, and Calmar Ratio.
4. Win Rate, Profit Factor, and Annual Portfolio Turnover.
5. Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (DSR) with statistical significance testing.
6. Execution Cost Tracking (Commissions, Slippage, Taxes).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class PerformanceMetrics:
    """Quantitative performance and risk metrics summary."""
    cagr: float
    total_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    max_drawdown_days: int
    win_rate: float
    profit_factor: float
    total_trades: int
    turnover_annual: float
    total_commission_paid: float
    total_slippage_paid: float
    total_tax_paid: float
    deflated_sharpe: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 3.0

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to a formatted dictionary for reporting."""
        return {
            "CAGR (%)": round(self.cagr * 100.0, 2),
            "Total Return (%)": round(self.total_return * 100.0, 2),
            "Annual Volatility (%)": round(self.annual_volatility * 100.0, 2),
            "Sharpe Ratio": round(self.sharpe_ratio, 2),
            "Sortino Ratio": round(self.sortino_ratio, 2),
            "Calmar Ratio": round(self.calmar_ratio, 2),
            "Max Drawdown (%)": round(self.max_drawdown * 100.0, 2),
            "Max Drawdown (Days)": int(self.max_drawdown_days),
            "Win Rate (%)": round(self.win_rate * 100.0, 2),
            "Profit Factor": round(self.profit_factor, 2),
            "Total Trades": int(self.total_trades),
            "Annual Turnover (x)": round(self.turnover_annual, 2),
            "Total Commission ($)": round(self.total_commission_paid, 2),
            "Total Slippage ($)": round(self.total_slippage_paid, 2),
            "Total Tax Paid ($)": round(self.total_tax_paid, 2),
            "Deflated Sharpe Ratio": round(self.deflated_sharpe, 4),
            "Skewness": round(self.skewness, 2),
            "Kurtosis": round(self.kurtosis, 2)
        }


def detect_annualization_factor(equity_series: pd.Series) -> float:
    """
    Infer annualization factor M based on series DatetimeIndex frequency:
    - 5m bars: M = 252 * 78 = 19,656
    - 15m bars: M = 252 * 26 = 6,552
    - 30m bars: M = 252 * 13 = 3,276
    - 60m bars: M = 252 * 6.5 = 1,638
    - Daily bars: M = 252
    """
    if len(equity_series) < 3 or not isinstance(equity_series.index, pd.DatetimeIndex):
        return 252.0

    try:
        diffs = pd.Series(equity_series.index).diff().dropna()
        median_sec = diffs.median().total_seconds()
        
        if median_sec <= 420:  # <= 7 min (5m)
            return 19656.0
        elif median_sec <= 1200:  # <= 20 min (15m)
            return 6552.0
        elif median_sec <= 2700:  # <= 45 min (30m)
            return 3276.0
        elif median_sec <= 7200:  # <= 2h (60m)
            return 1638.0
        else:
            return 252.0
    except Exception:
        return 252.0


def deflated_sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    trials: int = 20,
    annualization_factor: float = 252.0,
    risk_free_rate: float = 0.02
) -> float:
    """
    Computes Bailey & Lopez de Prado (2014) Deflated Sharpe Ratio (DSR).
    Evaluates statistical significance adjusted for multiple testing selection bias,
    skewness, and fat-tailed kurtosis.
    """
    rets = np.asarray(returns, dtype=float)
    rets = rets[~np.isnan(rets)]
    k = len(rets)
    if k < 5:
        return 0.50

    rf_bar = (1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0
    excess_rets = rets - rf_bar
    std = float(np.std(rets, ddof=1))

    if std <= 1e-8:
        return 0.50

    # Non-annualized sample Sharpe ratio
    sr_hat = float(np.mean(excess_rets) / std)

    # Skewness and kurtosis
    skew = float(stats.skew(rets)) if k >= 8 else 0.0
    kurt = float(stats.kurtosis(rets, fisher=False)) if k >= 8 else 3.0  # Pearson kurtosis (normal = 3)

    # Expected maximum Sharpe ratio under null hypothesis
    euler_gamma = 0.5772156649
    n_trials = max(trials, 1)
    if n_trials > 1:
        sr_0 = float(
            (1.0 - euler_gamma) * stats.norm.ppf(1.0 - 1.0 / n_trials)
            + euler_gamma * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        )
    else:
        sr_0 = 0.0

    # Variance of Sharpe ratio estimator
    sr_var = (1.0 - skew * sr_hat + ((kurt - 1.0) / 4.0) * (sr_hat ** 2))
    sr_var = max(sr_var, 1e-6)

    # DSR Test statistic
    z_stat = (sr_hat - sr_0) * np.sqrt(k - 1) / np.sqrt(sr_var)
    dsr_pvalue = float(stats.norm.cdf(z_stat))

    return float(np.clip(dsr_pvalue, 0.0, 1.0))


def calculate_metrics(
    equity_series: pd.Series,
    trade_log: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.02,
    annualization_factor: Optional[float] = None,
    trials: int = 20
) -> PerformanceMetrics:
    """
    Compute comprehensive quantitative performance and risk metrics from an equity series.
    """
    if equity_series.empty or len(equity_series) < 2:
        return PerformanceMetrics(
            cagr=0.0, total_return=0.0, annual_volatility=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            max_drawdown=0.0, max_drawdown_days=0, win_rate=0.0,
            profit_factor=0.0, total_trades=0, turnover_annual=0.0,
            total_commission_paid=0.0, total_slippage_paid=0.0, total_tax_paid=0.0,
            deflated_sharpe=0.0, skewness=0.0, kurtosis=3.0
        )

    # Clean equity series
    clean_equity = equity_series.ffill().bfill()
    start_val = float(clean_equity.iloc[0])
    end_val = float(clean_equity.iloc[-1])

    if start_val <= 0:
        start_val = 1.0
        end_val = max(end_val, 0.0)

    # Determine annualization factor M
    M = annualization_factor if annualization_factor is not None else detect_annualization_factor(clean_equity)
    M = max(M, 1.0)

    # Bar-by-bar returns
    returns = clean_equity.pct_change().dropna()
    n_bars = len(returns)
    years = max(n_bars / M, 0.01)

    # 1. Returns & CAGR
    total_return = (end_val / start_val) - 1.0
    if end_val <= 0:
        cagr = -1.0
    else:
        cagr = ((end_val / start_val) ** (1.0 / years)) - 1.0

    # 2. Volatility
    ret_std = float(returns.std())
    ann_vol = ret_std * np.sqrt(M) if ret_std > 0 else 0.0

    # 3. Annualized Sharpe Ratio
    rf_bar = (1.0 + risk_free_rate) ** (1.0 / M) - 1.0
    excess_returns = returns - rf_bar
    mean_excess = float(excess_returns.mean())
    if ret_std > 1e-16:
        sharpe = (mean_excess / ret_std) * np.sqrt(M)
    elif mean_excess > 0:
        sharpe = (mean_excess / 1e-6) * np.sqrt(M)
    else:
        sharpe = 0.0

    # 4. Sortino Ratio (Downside semi-deviation below risk-free rate)
    downside_diff = np.minimum(0.0, excess_returns)
    downside_var = float(np.mean(downside_diff ** 2))
    downside_std = np.sqrt(downside_var) * np.sqrt(M)
    if downside_std > 1e-16:
        sortino = (mean_excess * M) / downside_std
    elif mean_excess > 0:
        sortino = (mean_excess * M) / 1e-6
    else:
        sortino = 0.0

    # 5. Drawdown Analytics
    cum_max = clean_equity.cummax()
    drawdown = (clean_equity - cum_max) / cum_max
    max_dd = abs(float(drawdown.min())) if not drawdown.empty else 0.0

    # Drawdown Duration
    is_underwater = drawdown < 0.0
    if is_underwater.any():
        dd_groups = (~is_underwater).cumsum()[is_underwater]
        max_dd_bars = int(dd_groups.value_counts().max()) if not dd_groups.empty else 0
        # Convert max drawdown bars to approximate days
        bars_per_day = max(M / 252.0, 1.0)
        max_dd_days = int(np.ceil(max_dd_bars / bars_per_day))
    else:
        max_dd_days = 0

    # 6. Calmar Ratio
    if max_dd > 1e-6:
        calmar = cagr / max_dd
    elif cagr > 0:
        calmar = cagr / 1e-6
    else:
        calmar = 0.0

    # 7. Distribution Statistics (Skewness, Kurtosis, Deflated Sharpe)
    skew = float(stats.skew(returns.values)) if n_bars >= 8 else 0.0
    kurt = float(stats.kurtosis(returns.values, fisher=False)) if n_bars >= 8 else 3.0
    dsr = deflated_sharpe_ratio(returns, trials=trials, annualization_factor=M, risk_free_rate=risk_free_rate)

    # 8. Trade Log Analytics
    win_rate = 0.0
    profit_factor = 0.0
    total_trades = 0
    total_comm = 0.0
    total_slip = 0.0
    total_tax = 0.0
    annual_turnover = 0.0

    if trade_log is not None and not trade_log.empty:
        total_trades = len(trade_log)
        pnl_col = "realized_pnl" if "realized_pnl" in trade_log.columns else "pnl"
        
        if pnl_col in trade_log.columns:
            winning_trades = trade_log[trade_log[pnl_col] > 0]
            losing_trades = trade_log[trade_log[pnl_col] < 0]

            win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
            gross_profits = float(winning_trades[pnl_col].sum())
            gross_losses = abs(float(losing_trades[pnl_col].sum()))

            if gross_losses > 1e-6:
                profit_factor = gross_profits / gross_losses
            else:
                profit_factor = gross_profits if gross_profits > 0 else 0.0

        if "commission" in trade_log.columns:
            total_comm = float(trade_log["commission"].sum())
        if "slippage" in trade_log.columns:
            total_slip = float(trade_log["slippage"].sum())
        if "tax" in trade_log.columns:
            total_tax = float(trade_log["tax"].sum())

        val_col = "value" if "value" in trade_log.columns else "order_value"
        if val_col in trade_log.columns:
            total_traded_val = float(trade_log[val_col].sum())
            annual_turnover = (total_traded_val / (start_val * years)) if start_val > 0 else 0.0

    return PerformanceMetrics(
        cagr=cagr,
        total_return=total_return,
        annual_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_days=max_dd_days,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=total_trades,
        turnover_annual=annual_turnover,
        total_commission_paid=total_comm,
        total_slippage_paid=total_slip,
        total_tax_paid=total_tax,
        deflated_sharpe=dsr,
        skewness=skew,
        kurtosis=kurt
    )
