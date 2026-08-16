"""
Unit and Integration Tests for Whale-Tracking Quantitative System.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
import pandas as pd

from src.data.loader import MarketDataLoader, UniverseConfig, AssetMarket
from src.signals.whale_detector import WhaleDetector
from src.portfolio.allocator import PortfolioAllocator
from src.execution.simulator import ExecutionModel, AlmgrenChrissConfig
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import calculate_metrics
from src.validation.monte_carlo import MonteCarloSimulator
from src.validation.cpcv import CPCVValidator
from src.validation.deflated_sharpe import calculate_deflated_sharpe_ratio


@pytest.fixture
def mock_market_data():
    """Generate deterministic mock OHLCV dataframe for tests."""
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="B")
    n = len(dates)
    np.random.seed(42)

    returns = np.random.normal(0.001, 0.015, n)
    close = 100.0 * np.cumprod(1.0 + returns)
    high = close * 1.01
    low = close * 0.99
    open_p = (high + low) / 2.0
    volume = np.random.uniform(5e6, 15e6, n)

    # Inject volume surge & price rise
    volume[100:110] *= 4.0
    close[100:110] *= 1.05

    df = pd.DataFrame({
        "Open": open_p,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
        "MarketCap": 50e9,
        "Currency": "USD",
        "Market": "US"
    }, index=dates)
    return df


def test_data_loader():
    config = UniverseConfig(
        us_symbols=["AAPL", "NVDA"],
        bvc_symbols=["ATW.BVC"],
        start_date="2023-01-01",
        end_date="2024-01-01"
    )
    loader = MarketDataLoader(config)
    df_us = loader.fetch_single_asset("AAPL", AssetMarket.US)
    assert df_us is not None
    assert "Close" in df_us.columns
    assert "Volume" in df_us.columns
    assert len(df_us) > 50

    df_bvc = loader.fetch_single_asset("ATW.BVC", AssetMarket.BVC)
    assert df_bvc is not None
    assert df_bvc["Currency"].iloc[0] == "MAD"


def test_whale_detector_signals(mock_market_data):
    detector = WhaleDetector()
    processed = detector.process_asset(mock_market_data, market="US")

    assert "Vol_ZScore" in processed.columns
    assert "RVOL" in processed.columns
    assert "Whale_Signal" in processed.columns
    assert "Conviction_Multiplier" in processed.columns
    assert "Mom_Score" in processed.columns

    # Verify signals are boolean 0/1
    assert set(processed["Whale_Signal"].unique()).issubset({0, 1})


def test_portfolio_allocator(mock_market_data):
    detector = WhaleDetector()
    sig1 = detector.process_asset(mock_market_data, market="US")

    mock2 = mock_market_data.copy()
    mock2["Close"] *= 1.2
    sig2 = detector.process_asset(mock2, market="US")

    allocator = PortfolioAllocator(max_single_weight=0.25, max_total_exposure=1.0)
    res = allocator.compute_weights({"ASSET_A": sig1, "ASSET_B": sig2})

    weights = res.target_weights
    assert (weights >= 0.0).all().all()
    assert (weights.sum(axis=1) <= 1.0001).all()
    assert (weights["ASSET_A"] <= 0.2501).all()


def test_execution_model():
    exec_model = ExecutionModel()
    res = exec_model.execute_order(
        symbol="NVDA",
        market="US",
        side="BUY",
        order_value_usd=10_000.0,
        current_price=100.0,
        daily_volume=5_000_000.0
    )
    assert res.executed_shares > 0
    assert res.executed_price >= 100.0  # Slippage impact
    assert res.total_commission > 0

    # Test BVC T+3 settlement and 15% IR tax
    res_bvc = exec_model.execute_order(
        symbol="ATW.BVC",
        market="BVC",
        side="SELL",
        order_value_usd=5_000.0,
        current_price=50.0,
        daily_volume=50_000.0,
        avg_entry_price=40.0,
        day_index=1
    )
    assert res_bvc.tax_withheld > 0.0  # Moroccan 15% IR tax
    assert len(exec_model.settlement_queue) == 1

    # Check cash release at day 4 (day 1 + 3)
    released = exec_model.process_daily_settlement(4)
    assert released > 0


def test_metrics_calculation():
    dates = pd.date_range("2023-01-01", "2024-01-01", freq="B")
    equity = pd.Series(100_000.0 * (1.30 ** (np.arange(len(dates)) / 252.0)), index=dates)
    metrics = calculate_metrics(equity)

    assert metrics.cagr >= 0.29
    assert metrics.max_drawdown <= 0.05
    assert metrics.sharpe_ratio > 0


def test_monte_carlo_validation():
    dates = pd.date_range("2022-01-01", "2025-01-01", freq="B")
    # Upward trending series with positive drift
    np.random.seed(42)
    daily_ret = np.random.normal(0.0014, 0.012, len(dates))
    equity = pd.Series(100_000.0 * np.cumprod(1.0 + daily_ret), index=dates)

    mc = MonteCarloSimulator(n_simulations=500, target_cagr=0.30)
    res = mc.run(equity)

    assert res.n_simulations == 500
    assert 0.0 <= res.prob_cagr_above_target <= 1.0
    assert res.median_cagr if hasattr(res, 'median_cagr') else res.cagr_50th_percentile is not None


def test_cpcv_and_deflated_sharpe():
    dates = pd.date_range("2022-01-01", "2024-01-01", freq="B")
    daily_ret = pd.Series(np.random.normal(0.0012, 0.010, len(dates)), index=dates)
    equity = 100_000.0 * (1.0 + daily_ret).cumprod()

    cpcv = CPCVValidator(n_groups=4, k_test_groups=2)
    cpcv_res = cpcv.validate(equity)
    assert cpcv_res.n_splits == 6
    assert len(cpcv_res.cagr_distribution) == 6

    dsr = calculate_deflated_sharpe_ratio(daily_ret, n_trials=10)
    assert 0.0 <= dsr <= 1.0
