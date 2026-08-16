"""
Main Execution Script for Whale-Tracking Momentum Quantitative Trading System.
Runs data loading, feature engineering, backtesting, statistical validation, and reporting.
"""

from __future__ import annotations
import sys
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import MarketDataLoader, UniverseConfig
from src.signals.whale_detector import WhaleDetector
from src.portfolio.allocator import PortfolioAllocator
from src.execution.simulator import ExecutionModel, AlmgrenChrissConfig
from src.backtest.engine import BacktestEngine
from src.validation.monte_carlo import MonteCarloSimulator
from src.validation.cpcv import CPCVValidator
from src.validation.deflated_sharpe import calculate_deflated_sharpe_ratio
from src.reporting.tearsheet import generate_full_report


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def run_pipeline():
    setup_logging()
    logger = logging.getLogger("WhaleTracking.Main")
    logger.info("=" * 80)
    logger.info("LAUNCHING QUANTITATIVE WHALE-TRACKING MOMENTUM TRADING SYSTEM")
    logger.info("Dual-Universe: US Equities (NYSE/NASDAQ) & Casablanca Stock Exchange (BVC)")
    logger.info("=" * 80)

    # 1. Initialize Universe Configuration
    config = UniverseConfig(
        start_date="2020-01-01",
        end_date="2026-01-01",
        cache_dir="data_cache"
    )

    # 2. Configure Alpha Engine & Portfolio Rules
    detector = WhaleDetector(
        volume_window=30,
        z_score_threshold=2.2,
        rvol_threshold=2.2,
        min_momentum=0.10,
        contamination=0.08,
        use_isolation_forest=True
    )

    allocator = PortfolioAllocator(
        target_annual_vol=0.32,       # 32% target volatility
        max_single_weight=0.34,       # 34% max allocation per top leader
        min_weight_threshold=0.04,
        holding_period_days=25,       # 5-week swing accumulation horizon
        max_total_exposure=1.00
    )

    exec_config = AlmgrenChrissConfig(
        us_linear_impact_eta=0.03,
        us_permanent_impact_gamma=0.01,
        us_brokerage_bps=3.0,
        bvc_linear_impact_eta=0.15,
        bvc_permanent_impact_gamma=0.08,
        bvc_brokerage_bps=15.0,
        moroccan_ir_tax_rate=0.15,
        settlement_days_bvc=3
    )
    exec_model = ExecutionModel(exec_config)

    # 3. Instantiate and Run Backtest Simulation Engine
    engine = BacktestEngine(
        universe_config=config,
        whale_detector=detector,
        portfolio_allocator=allocator,
        execution_model=exec_model,
        initial_capital=100_000.0,
        rebalance_frequency=5        # Weekly rebalance review with daily trailing stop guard
    )

    backtest_res = engine.run(use_cache=True)
    metrics = backtest_res.metrics

    # 4. Statistical Validation (Monte Carlo & CPCV)
    logger.info("Running 2,500-iteration Block-Bootstrap Monte Carlo simulations...")
    mc_sim = MonteCarloSimulator(
        n_simulations=2500,
        block_size=10,
        target_cagr=0.30,
        random_seed=42
    )
    mc_res = mc_sim.run(backtest_res.equity_series)

    logger.info("Running Combinatorial Purged Cross-Validation (CPCV)...")
    cpcv_validator = CPCVValidator(n_groups=5, k_test_groups=2)
    cpcv_res = cpcv_validator.validate(backtest_res.equity_series)

    # Deflated Sharpe Ratio
    daily_returns = backtest_res.equity_series.pct_change().dropna()
    dsr = calculate_deflated_sharpe_ratio(daily_returns, n_trials=30)

    # 5. Generate Visual Tearsheets & Output Reports
    logger.info("Generating performance tearsheets and visual artifacts in output/...")
    report_paths = generate_full_report(backtest_res, mc_res, cpcv_res, output_dir="output")

    # 6. Display Executive Summary
    print("\n" + "=" * 80)
    print("                      EXECUTIVE PERFORMANCE TEARSHEET")
    print("=" * 80)
    print(f"  Annualized Return (CAGR)      : {metrics.cagr*100:.2f}%  (Target >= 30.0%)")
    print(f"  Total Return (6 Years)        : {metrics.total_return*100:.2f}%")
    print(f"  Annualized Volatility         : {metrics.annual_volatility*100:.2f}%")
    print(f"  Sharpe Ratio (Rf=2%)          : {metrics.sharpe_ratio:.2f}")
    print(f"  Sortino Ratio (Downside)      : {metrics.sortino_ratio:.2f}")
    print(f"  Calmar Ratio (CAGR/MDD)       : {metrics.calmar_ratio:.2f}")
    print(f"  Maximum Drawdown (MDD)        : {metrics.max_drawdown*100:.2f}%")
    print(f"  Win Rate                      : {metrics.win_rate*100:.2f}%")
    print(f"  Profit Factor                 : {metrics.profit_factor:.2f}")
    print(f"  Total Trades Executed         : {metrics.total_trades:,}")
    print(f"  Total Commissions Paid        : ${metrics.total_commission_paid:,.2f}")
    print(f"  Total Slippage Impact (A-C)   : ${metrics.total_slippage_paid:,.2f}")
    print(f"  Total Moroccan IR Tax (15%)   : ${metrics.total_tax_paid:,.2f}")
    print("-" * 80)
    print("                  STATISTICAL SIGNIFICANCE & ROBUSTNESS")
    print("-" * 80)
    print(f"  Monte Carlo Confidence (CAGR >= 30%) : {mc_res.prob_cagr_above_target*100:.2f}% (Target >= 90.0%)")
    print(f"  Monte Carlo Empirical p-value       : {mc_res.p_value:.4f} (p < 0.10 threshold)")
    print(f"  5th Percentile Simulated CAGR       : {mc_res.cagr_5th_percentile*100:.2f}%")
    print(f"  Median (50th Pct) Simulated CAGR    : {mc_res.cagr_50th_percentile*100:.2f}%")
    print(f"  95th Percentile Simulated CAGR      : {mc_res.cagr_95th_percentile*100:.2f}%")
    print(f"  Deflated Sharpe Ratio (DSR)         : {dsr:.4f}")
    print(f"  CPCV Mean Out-of-Sample CAGR        : {cpcv_res.mean_out_of_sample_cagr*100:.2f}%")
    print(f"  CPCV Profitable Path Ratio          : {cpcv_res.percent_positive_paths*100:.2f}%")
    print("=" * 80)
    print(f"Artifacts exported:")
    print(f"  - High-Res Tearsheet: {report_paths['tearsheet_png']}")
    print(f"  - JSON Report:        {report_paths['report_json']}")
    print(f"  - Text Summary:       {report_paths['summary_txt']}")
    print("=" * 80 + "\n")

    return backtest_res, mc_res, cpcv_res


if __name__ == "__main__":
    run_pipeline()
