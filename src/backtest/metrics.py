"""
Institutional Performance & Risk Metrics Engine.
Computes CAGR, Sharpe, Sortino, Calmar, Max Drawdown, Win Rate, and Turnover.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PerformanceMetrics:
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

    def to_dict(self) -> Dict[str, float]:
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
            "Total Tax Paid ($)": round(self.total_tax_paid, 2)
        }


def calculate_metrics(
    equity_series: pd.Series,
    trade_log: Optional[pd.DataFrame] = None,
    risk_free_rate: float = 0.02
) -> PerformanceMetrics:
    """Compute standard quantitative performance metrics from an equity curve."""
    if equity_series.empty or len(equity_series) < 2:
        return PerformanceMetrics(
            cagr=0.0, total_return=0.0, annual_volatility=0.0,
            sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
            max_drawdown=0.0, max_drawdown_days=0, win_rate=0.0,
            profit_factor=0.0, total_trades=0, turnover_annual=0.0,
            total_commission_paid=0.0, total_slippage_paid=0.0, total_tax_paid=0.0
        )

    # Daily returns
    daily_returns = equity_series.pct_change().dropna()
    n_days = len(daily_returns)
    years = max(n_days / 252.0, 0.01)

    # 1. Returns & CAGR
    start_val = equity_series.iloc[0]
    end_val = equity_series.iloc[-1]
    total_return = (end_val / start_val) - 1.0
    cagr = ((end_val / start_val) ** (1.0 / years)) - 1.0

    # 2. Volatility
    ann_vol = daily_returns.std() * np.sqrt(252.0)
    ann_vol = max(ann_vol, 1e-6)

    # 3. Sharpe Ratio
    rf_daily = (1.0 + risk_free_rate) ** (1.0 / 252.0) - 1.0
    excess_returns = daily_returns - rf_daily
    sharpe = (excess_returns.mean() / daily_returns.std()) * np.sqrt(252.0) if daily_returns.std() > 0 else 0.0

    # 4. Sortino Ratio (Downside deviation)
    downside_returns = daily_returns[daily_returns < 0.0]
    downside_std = downside_returns.std() * np.sqrt(252.0) if len(downside_returns) > 0 else 1e-6
    sortino = (excess_returns.mean() * 252.0) / downside_std if downside_std > 0 else 0.0

    # 5. Drawdown Analytics
    cum_max = equity_series.cummax()
    drawdown = (equity_series - cum_max) / cum_max
    max_dd = abs(float(drawdown.min()))

    # Drawdown duration in days
    is_underwater = drawdown < 0.0
    dd_groups = (~is_underwater).cumsum()[is_underwater]
    max_dd_days = dd_groups.value_counts().max() if not dd_groups.empty else 0

    # 6. Calmar Ratio
    calmar = cagr / max_dd if max_dd > 0 else 0.0

    # 7. Trade Log Statistics
    win_rate = 0.0
    profit_factor = 0.0
    total_trades = 0
    total_comm = 0.0
    total_slip = 0.0
    total_tax = 0.0
    annual_turnover = 0.0

    if trade_log is not None and not trade_log.empty:
        total_trades = len(trade_log)
        winning_trades = trade_log[trade_log["realized_pnl"] > 0]
        losing_trades = trade_log[trade_log["realized_pnl"] < 0]

        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0.0
        gross_profits = winning_trades["realized_pnl"].sum()
        gross_losses = abs(losing_trades["realized_pnl"].sum())
        profit_factor = (gross_profits / gross_losses) if gross_losses > 0 else (gross_profits if gross_profits > 0 else 0.0)

        total_comm = trade_log["commission"].sum()
        total_slip = trade_log["slippage"].sum()
        total_tax = trade_log.get("tax", pd.Series(0.0)).sum()

        total_traded_value = trade_log["value"].sum()
        annual_turnover = (total_traded_value / (start_val * years)) if start_val > 0 else 0.0

    return PerformanceMetrics(
        cagr=cagr,
        total_return=total_return,
        annual_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        max_drawdown=max_dd,
        max_drawdown_days=int(max_dd_days),
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=total_trades,
        turnover_annual=annual_turnover,
        total_commission_paid=total_comm,
        total_slippage_paid=total_slip,
        total_tax_paid=total_tax
    )
