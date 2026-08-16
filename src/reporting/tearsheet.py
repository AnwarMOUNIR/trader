"""
Comprehensive Visual Reporting & Tearsheet Generation Engine.
Creates multi-panel institutional performance figures and exports summary artifacts.
"""

from __future__ import annotations
import json
import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from ..backtest.engine import BacktestResult
from ..validation.monte_carlo import MonteCarloResult
from ..validation.cpcv import CPCVResult

logger = logging.getLogger("WhaleTracking.Reporting")


def generate_full_report(
    backtest_res: BacktestResult,
    mc_res: Optional[MonteCarloResult] = None,
    cpcv_res: Optional[CPCVResult] = None,
    output_dir: str = "output"
) -> Dict[str, str]:
    """Generates visual charts and summary text files, returning paths to created artifacts."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig = plt.figure(figsize=(18, 14), dpi=150)
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    eq = backtest_res.equity_series
    bm = backtest_res.benchmark_equity
    dates = eq.index

    # 1. Cumulative Performance Chart (Top-Left)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(dates, eq, label="Whale-Tracking Momentum Strategy", color="#1f77b4", linewidth=2.0)
    ax1.plot(dates, bm, label="Benchmark (S&P 500 Proxy)", color="#7f7f7f", linestyle="--", linewidth=1.5)
    ax1.set_title(f"Cumulative Portfolio Value (Initial: ${eq.iloc[0]:,.0f} -> Final: ${eq.iloc[-1]:,.0f})", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Portfolio Value ($ USD)")
    ax1.legend(loc="upper left", frameon=True)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # 2. Drawdown Underwater Chart (Top-Right)
    ax2 = fig.add_subplot(gs[0, 1])
    cum_max = eq.cummax()
    drawdown = (eq - cum_max) / cum_max * 100.0
    ax2.fill_between(dates, drawdown, 0.0, color="#d62728", alpha=0.35)
    ax2.plot(dates, drawdown, color="#d62728", linewidth=1.2)
    ax2.set_title(f"Portfolio Drawdown Underwater (Max DD: {backtest_res.metrics.max_drawdown*100:.2f}%)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_ylim(bottom=min(-35.0, drawdown.min() * 1.15), top=2.0)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # 3. Monthly Returns Heatmap (Middle-Left)
    ax3 = fig.add_subplot(gs[1, 0])
    daily_ret = eq.pct_change().dropna()
    monthly_ret = daily_ret.resample("ME").apply(lambda r: (1.0 + r).prod() - 1.0) * 100.0
    monthly_df = pd.DataFrame({
        "Year": monthly_ret.index.year,
        "Month": monthly_ret.index.month,
        "Return": monthly_ret.values
    })
    pivot_table = monthly_df.pivot(index="Year", columns="Month", values="Return").fillna(0.0)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Reindex columns to ensure all 12 months appear
    pivot_table = pivot_table.reindex(columns=range(1, 13), fill_value=0.0)
    pivot_table.columns = month_labels

    sns.heatmap(
        pivot_table,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn",
        center=0.0,
        cbar=False,
        ax=ax3,
        linewidths=0.5
    )
    ax3.set_title("Monthly Returns Heatmap (%)", fontsize=11, fontweight="bold")
    ax3.set_xlabel("")

    # 4. Rolling 6-Month Sharpe Ratio (Middle-Right)
    ax4 = fig.add_subplot(gs[1, 1])
    roll_mean = daily_ret.rolling(126, min_periods=40).mean()
    roll_std = daily_ret.rolling(126, min_periods=40).std()
    roll_sharpe = (roll_mean / roll_std.replace(0, np.nan)) * np.sqrt(252.0)
    ax4.plot(dates[1:], roll_sharpe, color="#2ca02c", linewidth=1.5)
    ax4.axhline(1.5, color="#1f77b4", linestyle="--", alpha=0.7, label="Target Sharpe (1.50)")
    ax4.set_title(f"Rolling 6-Month Sharpe Ratio (Full Period Sharpe: {backtest_res.metrics.sharpe_ratio:.2f})", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Annualized Sharpe Ratio")
    ax4.legend(loc="upper left")
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # 5. Monte Carlo Simulated CAGR Distribution (Bottom-Left)
    ax5 = fig.add_subplot(gs[2, 0])
    if mc_res is not None:
        cagrs_pct = mc_res.simulated_cagrs * 100.0
        sns.histplot(cagrs_pct, kde=True, color="#9467bd", bins=40, ax=ax5, stat="density")
        ax5.axvline(30.0, color="#d62728", linestyle="--", linewidth=2.0, label="30% CAGR Target")
        ax5.axvline(mc_res.cagr_5th_percentile * 100.0, color="#ff7f0e", linestyle=":", label=f"5th Pct ({mc_res.cagr_5th_percentile*100:.1f}%)")
        ax5.axvline(mc_res.cagr_50th_percentile * 100.0, color="#2ca02c", linewidth=1.5, label=f"Median ({mc_res.cagr_50th_percentile*100:.1f}%)")
        ax5.set_title(
            f"Monte Carlo Distribution ({mc_res.n_simulations:,} runs) — Confidence: {mc_res.prob_cagr_above_target*100:.1f}% (p={mc_res.p_value:.3f})",
            fontsize=11, fontweight="bold"
        )
        ax5.set_xlabel("Simulated Annualized Return CAGR (%)")
        ax5.legend(loc="upper right", fontsize=9)
    else:
        ax5.text(0.5, 0.5, "Monte Carlo validation not executed", ha="center", va="center")

    # 6. Asset Class / Market Allocation Over Time (Bottom-Right)
    ax6 = fig.add_subplot(gs[2, 1])
    weights = backtest_res.weights_df
    us_cols = [c for c in weights.columns if ".BVC" not in c]
    bvc_cols = [c for c in weights.columns if ".BVC" in c]

    us_total_w = weights[us_cols].sum(axis=1) * 100.0
    bvc_total_w = weights[bvc_cols].sum(axis=1) * 100.0
    cash_w = (100.0 - us_total_w - bvc_total_w).clip(lower=0.0)

    ax6.stackplot(
        weights.index,
        us_total_w,
        bvc_total_w,
        cash_w,
        labels=["US Equities Exposure", "Casablanca (BVC) Exposure", "Cash & Settlement Queue"],
        colors=["#1f77b4", "#ff7f0e", "#2ca02c"],
        alpha=0.8
    )
    ax6.set_title("Dynamic Portfolio Exposure & Market Allocation (%)", fontsize=11, fontweight="bold")
    ax6.set_ylabel("Allocation (%)")
    ax6.set_ylim(0.0, 100.0)
    ax6.legend(loc="lower left", fontsize=9)
    ax6.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # Save tearsheet figure
    tearsheet_path = out_path / "tearsheet.png"
    fig.savefig(tearsheet_path, bbox_inches="tight")
    plt.close(fig)

    # Save JSON summary metrics
    report_dict = {
        "strategy_metrics": backtest_res.metrics.to_dict(),
        "monte_carlo_validation": mc_res.to_dict() if mc_res else {},
        "cpcv_validation": cpcv_res.to_dict() if cpcv_res else {}
    }
    json_path = out_path / "performance_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=4)

    # Save Markdown / Text summary table
    txt_path = out_path / "performance_summary.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  WHALE-TRACKING MOMENTUM QUANTITATIVE TRADING STRATEGY\n")
        f.write("=" * 70 + "\n\n")
        f.write("KEY PERFORMANCE METRICS:\n")
        f.write("-" * 40 + "\n")
        for k, v in backtest_res.metrics.to_dict().items():
            f.write(f"{k:<30}: {v}\n")
        if mc_res:
            f.write("\nMONTE CARLO STATISTICAL AUDIT (2,000 SIMULATIONS):\n")
            f.write("-" * 40 + "\n")
            for k, v in mc_res.to_dict().items():
                f.write(f"{k:<30}: {v}\n")
        if cpcv_res:
            f.write("\nCOMBINATORIAL PURGED CROSS-VALIDATION (CPCV):\n")
            f.write("-" * 40 + "\n")
            for k, v in cpcv_res.to_dict().items():
                f.write(f"{k:<30}: {v}\n")
        f.write("=" * 70 + "\n")

    return {
        "tearsheet_png": str(tearsheet_path),
        "report_json": str(json_path),
        "summary_txt": str(txt_path)
    }
