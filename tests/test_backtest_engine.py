"""
Comprehensive Unit & Integration Test Suite for Milestone M4.
High-Frequency Intraday Backtester, Microstructure Frictions & Paper Simulator.

Covers:
1. Almgren-Chriss Market Impact Model (OrderBookSimulator / ExecutionModel).
2. Casablanca Stock Exchange BVC Microstructure Adapter (T+3, 15% IR Tax, +/-10% Circuit Breakers).
3. Quantitative Performance & Risk Metrics (CAGR, MDD, Sharpe, Sortino, Calmar, DSR).
4. Discrete-Event Intraday Backtest Engine (5m-60m and Daily Simulation).
5. Alpaca Paper Trading REST Client with Server-Side Bracket Orders.
6. Autonomous Dry-Run Simulator & Execution Scheduler with Telemetry Snapshots.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pytest

# Ensure repository root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backtest.order_book import (
    OrderBookSimulator,
    ExecutionModel,
    AlmgrenChrissConfig,
    ExecutionResult,
)
from src.backtest.bvc_adapter import (
    BVCAdapter,
    BVCMicrostructureHandler,
    BVCPriceLimitResult,
)
from src.backtest.metrics import (
    PerformanceMetrics,
    calculate_metrics,
    deflated_sharpe_ratio,
    detect_annualization_factor,
)
from src.backtest.intraday_engine import (
    IntradayBacktestEngine,
    BacktestEngine,
    BacktestResult,
)
from src.execution.alpaca_client import (
    AlpacaPaperTrader,
    AlpacaClient,
    OrderResult,
)
from src.execution.dry_run_simulator import (
    DryRunSimulator,
    ExecutionCycleReport,
)
from src.execution.scheduler import ExecutionScheduler
from src.data.loader import UniverseConfig


# ============================================================================
# 1. Almgren-Chriss Market Impact & Order Book Tests
# ============================================================================

def test_almgren_chriss_market_impact_calculation():
    """Verify non-linear temporary and permanent impact computation."""
    cfg = AlmgrenChrissConfig(
        us_linear_impact_eta=0.05,
        us_permanent_impact_gamma=0.02,
        us_impact_alpha=1.0,
        us_impact_beta=1.0,
        us_half_spread_bps=2.0,
        us_max_participation_rate=0.25
    )
    sim = OrderBookSimulator(cfg)

    part_rate, spread_rate, temp_imp, perm_imp = sim.calculate_market_impact(
        order_shares=1000,
        bar_volume=10000,
        market="US"
    )

    assert pytest.approx(part_rate, 1e-4) == 0.10
    assert pytest.approx(spread_rate, 1e-5) == 0.0002  # 2 bps
    assert pytest.approx(temp_imp, 1e-4) == 0.05 * 0.10  # 0.005
    assert pytest.approx(perm_imp, 1e-4) == 0.02 * 0.10  # 0.002


def test_almgren_chriss_participation_rate_clamping():
    """Verify participation rate is capped at max participation ceiling (e.g. 25% US, 10% BVC)."""
    cfg = AlmgrenChrissConfig(us_max_participation_rate=0.25, bvc_max_participation_rate=0.10)
    sim = OrderBookSimulator(cfg)

    # Large US order (50% of bar volume)
    part_us, _, _, _ = sim.calculate_market_impact(5000, 10000, market="US")
    assert part_us == 0.25

    # Large BVC order (50% of bar volume)
    part_bvc, _, _, _ = sim.calculate_market_impact(5000, 10000, market="BVC")
    assert part_bvc == 0.10


def test_order_execution_slippage_and_direction():
    """Verify BUY order increases execution price, SELL decreases execution price."""
    sim = OrderBookSimulator()
    price = 100.0
    val = 10_000.0

    # BUY Order
    res_buy: ExecutionResult = sim.execute_order(
        symbol="NVDA",
        market="US",
        side="BUY",
        order_value_usd=val,
        current_price=price,
        daily_volume=100_000
    )
    assert res_buy.executed_price > price
    assert res_buy.total_slippage > 0.0
    assert res_buy.total_commission > 0.0
    assert res_buy.order_side == "BUY"

    # SELL Order
    res_sell: ExecutionResult = sim.execute_order(
        symbol="NVDA",
        market="US",
        side="SELL",
        order_value_usd=val,
        current_price=price,
        daily_volume=100_000
    )
    assert res_sell.executed_price < price
    assert res_sell.total_slippage > 0.0
    assert res_sell.order_side == "SELL"


def test_order_execution_zero_and_edge_inputs():
    """Verify order simulator handles zero value / zero price gracefully."""
    sim = OrderBookSimulator()
    res = sim.execute_order("AAPL", "US", "BUY", order_value_usd=0.0, current_price=150.0, daily_volume=10000)
    assert res.executed_shares == 0.0
    assert res.total_cost == 0.0


# ============================================================================
# 2. Casablanca Stock Exchange BVC Microstructure Adapter Tests
# ============================================================================

def test_bvc_circuit_breaker_continuous_and_fixing():
    """Verify +/-10% continuous and +/-6% fixing circuit breaker clamping."""
    adapter = BVCAdapter(continuous_circuit_breaker=0.10, fixing_circuit_breaker=0.06)
    ref_price = 500.0

    # 1. Normal in-band price (520.0, +4%)
    res_norm = adapter.check_circuit_breaker(520.0, ref_price, session_type="continuous")
    assert not res_norm.is_limit_hit
    assert res_norm.clamped_price == 520.0
    assert res_norm.limit_type == "NORMAL"

    # 2. Continuous Limit Up (+15% attempt -> clamped at +10% = 550.0)
    res_up = adapter.check_circuit_breaker(575.0, ref_price, session_type="continuous")
    assert res_up.is_limit_hit
    assert pytest.approx(res_up.clamped_price, 1e-4) == 550.0
    assert res_up.limit_type == "LIMIT_UP"

    # 3. Continuous Limit Down (-15% attempt -> clamped at -10% = 450.0)
    res_down = adapter.check_circuit_breaker(425.0, ref_price, session_type="continuous")
    assert res_down.is_limit_hit
    assert pytest.approx(res_down.clamped_price, 1e-4) == 450.0
    assert res_down.limit_type == "LIMIT_DOWN"

    # 4. Fixing Limit Up (+8% attempt -> clamped at +6% = 530.0)
    res_fix = adapter.check_circuit_breaker(540.0, ref_price, session_type="fixing")
    assert res_fix.is_limit_hit
    assert pytest.approx(res_fix.clamped_price, 1e-4) == 530.0


def test_bvc_t3_settlement_queue_latency():
    """Verify cash proceeds from Casablanca asset sales are delayed by T+3 trading days."""
    adapter = BVCAdapter(settlement_latency_days=3)

    # Sale on Day 0: $5,000 net proceeds
    adapter.queue_settlement_proceeds("order_1", "ATW.BVC", 5000.0, current_day_index=0)
    # Sale on Day 1: $3,000 net proceeds
    adapter.queue_settlement_proceeds("order_2", "BCP.BVC", 3000.0, current_day_index=1)

    assert adapter.get_pending_settlement_total() == 8000.0

    # Day 0: No cash released
    cash_d0, _ = adapter.process_settlement_queue(current_day_index=0)
    assert cash_d0 == 0.0

    # Day 2: Still not passed T+3 (order_1 is due on Day 3)
    cash_d2, _ = adapter.process_settlement_queue(current_day_index=2)
    assert cash_d2 == 0.0

    # Day 3: Order 1 ($5,000) is released
    cash_d3, items_d3 = adapter.process_settlement_queue(current_day_index=3)
    assert cash_d3 == 5000.0
    assert len(items_d3) == 1
    assert adapter.get_pending_settlement_total() == 3000.0

    # Day 4: Order 2 ($3,000) is released
    cash_d4, items_d4 = adapter.process_settlement_queue(current_day_index=4)
    assert cash_d4 == 3000.0
    assert adapter.get_pending_settlement_total() == 0.0


def test_bvc_moroccan_ir_15_percent_tax_withholding():
    """Verify 15% IR tax withholding on net realized capital gains only."""
    adapter = BVCAdapter(ir_tax_rate=0.15)

    # Profitable sale: Buy at 100, Sell 100 shares at 120 (Gross Gain = 2000, Commission = 50)
    net_pnl, tax = adapter.calculate_capital_gains_tax(
        executed_shares=100,
        executed_price=120.0,
        avg_entry_price=100.0,
        commission=50.0
    )
    gross_gain = 100 * (120.0 - 100.0) - 50.0  # 1950.0
    expected_tax = 1950.0 * 0.15  # 292.5
    expected_net = 1950.0 - expected_tax  # 1657.5

    assert pytest.approx(tax, 1e-4) == expected_tax
    assert pytest.approx(net_pnl, 1e-4) == expected_net

    # Loss sale: Buy at 100, Sell at 90 -> Tax must be 0.0
    net_loss, tax_loss = adapter.calculate_capital_gains_tax(
        executed_shares=100,
        executed_price=90.0,
        avg_entry_price=100.0,
        commission=20.0
    )
    assert tax_loss == 0.0
    assert net_loss < 0.0


# ============================================================================
# 3. Quantitative Performance & Risk Metrics Tests
# ============================================================================

def test_metrics_cagr_and_frequency_detection():
    """Verify CAGR calculation across daily and intraday annualization factors."""
    # 252 daily bars growing from 100k to 140k (40% in 1 year)
    dates = pd.date_range("2025-01-01", periods=252, freq="D", tz="UTC")
    values = np.linspace(100_000, 140_000, 252)
    daily_equity = pd.Series(values, index=dates)

    metrics = calculate_metrics(daily_equity)
    assert pytest.approx(metrics.total_return, 1e-3) == 0.40
    assert pytest.approx(metrics.cagr, 1e-2) == 0.40

    # 5m intraday series (e.g. 78 bars = 1 day)
    intraday_dates = pd.date_range("2026-01-01 14:30", periods=78, freq="5min", tz="UTC")
    intraday_eq = pd.Series(np.linspace(100_000, 101_000, 78), index=intraday_dates)
    factor = detect_annualization_factor(intraday_eq)
    assert factor == 19656.0


def test_metrics_sharpe_sortino_and_drawdown():
    """Verify Sharpe, Sortino, Calmar, and Drawdown analytics."""
    np.random.seed(42)
    # Steady upward path with occasional dips
    dates = pd.date_range("2025-01-01", periods=252, freq="D", tz="UTC")
    returns = np.random.normal(0.0015, 0.008, 252)
    # Inject a 5% pullback
    returns[100:105] = -0.015
    eq_curve = 100_000 * np.cumprod(1.0 + returns)
    eq_series = pd.Series(eq_curve, index=dates)

    metrics = calculate_metrics(eq_series)
    assert metrics.cagr > 0.0
    assert metrics.sharpe_ratio > 1.0
    assert metrics.sortino_ratio > 1.0
    assert 0.0 < metrics.max_drawdown < 0.20
    assert metrics.max_drawdown_days > 0

    m_dict = metrics.to_dict()
    assert "CAGR (%)" in m_dict
    assert "Sharpe Ratio" in m_dict
    assert "Max Drawdown (%)" in m_dict


def test_deflated_sharpe_ratio_statistical_significance():
    """Verify Bailey & Lopez de Prado DSR returns valid probability p-value."""
    np.random.seed(123)
    rets = np.random.normal(0.001, 0.01, 500)
    dsr = deflated_sharpe_ratio(rets, trials=20, annualization_factor=252.0)
    assert 0.0 <= dsr <= 1.0


# ============================================================================
# 4. Discrete-Event Intraday Backtest Engine Tests
# ============================================================================

def test_intraday_backtest_engine_run_with_synthetic_data():
    """Verify bar-by-bar discrete simulation loop generates full BacktestResult."""
    dates = pd.date_range("2026-01-01", periods=100, freq="D", tz="UTC")
    
    # Create synthetic price series for US and BVC assets
    df_nvda = pd.DataFrame({
        "Open": np.linspace(100, 150, 100),
        "High": np.linspace(102, 152, 100),
        "Low": np.linspace(98, 148, 100),
        "Close": np.linspace(100, 150, 100),
        "Volume": np.full(100, 500_000)
    }, index=dates)

    df_bvc = pd.DataFrame({
        "Open": np.linspace(400, 480, 100),
        "High": np.linspace(405, 485, 100),
        "Low": np.linspace(395, 475, 100),
        "Close": np.linspace(400, 480, 100),
        "Volume": np.full(100, 50_000)
    }, index=dates)

    data_dict = {
        "NVDA": df_nvda,
        "ATW.BVC": df_bvc,
        "SPY": df_nvda.copy()
    }

    target_weights = pd.DataFrame({
        "NVDA": [0.30] * 100,
        "ATW.BVC": [0.20] * 100
    }, index=dates)

    engine = IntradayBacktestEngine(initial_capital=100_000.0, rebalance_frequency=5)
    result: BacktestResult = engine.run(
        data_dict=data_dict,
        target_weights=target_weights,
        initial_capital=100_000.0
    )

    assert isinstance(result, BacktestResult)
    assert len(result.equity_series) == 100
    assert result.equity_series.iloc[-1] > 100_000.0
    assert isinstance(result.metrics, PerformanceMetrics)
    assert result.metrics.cagr > 0.0
    assert not result.positions_df.empty
    assert len(result.trade_log) >= 2


def test_intraday_engine_stop_loss_trigger():
    """Verify stop-loss triggers on a gap-down crash and executes with slippage."""
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    
    # Asset gaps down severely on day 5
    closes = [100.0, 102.0, 101.0, 103.0, 85.0, 84.0, 85.0, 86.0, 87.0, 88.0]
    opens = [100.0, 102.0, 101.0, 103.0, 86.0, 84.0, 85.0, 86.0, 87.0, 88.0]
    lows = [99.0, 101.0, 100.0, 102.0, 83.0, 83.0, 84.0, 85.0, 86.0, 87.0]
    highs = [101.0, 103.0, 102.0, 104.0, 87.0, 85.0, 86.0, 87.0, 88.0, 89.0]
    
    df_crash = pd.DataFrame({
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": closes,
        "Volume": np.full(10, 100_000)
    }, index=dates)

    target_weights = pd.DataFrame({
        "CRASH": [0.30] * 10
    }, index=dates)

    engine = IntradayBacktestEngine(initial_capital=100_000.0, stop_loss_pct=0.048)
    result = engine.run(
        data_dict={"CRASH": df_crash},
        target_weights=target_weights,
        initial_capital=100_000.0
    )

    # Check if STOP_LOSS trade record was logged
    stops = result.trade_log[result.trade_log["side"] == "STOP_LOSS"]
    assert len(stops) >= 1
    assert stops.iloc[0]["symbol"] == "CRASH"


# ============================================================================
# 5. Alpaca Paper Client & Bracket Order Tests
# ============================================================================

def test_alpaca_client_bracket_order_generation():
    """Verify bracket order calculates stop-loss capped at -4.8% max and take-profit at +1.5 ATR."""
    client = AlpacaPaperTrader(max_stop_loss_pct=0.048)
    price = 200.0
    atr = 5.0  # ATR = $5.00 -> SL = 200 - 5 = 195 (2.5% < 4.8%), TP = 200 + 7.5 = 207.5

    res: OrderResult = client.submit_bracket_order(
        symbol="NVDA",
        qty=50,
        side="BUY",
        current_price=price,
        atr=atr
    )

    assert res.is_dry_run
    assert res.symbol == "NVDA"
    assert res.qty == 50
    assert res.side == "BUY"
    assert pytest.approx(res.stop_price, 1e-2) == 195.0
    assert pytest.approx(res.take_profit_price, 1e-2) == 207.5


def test_alpaca_client_bracket_order_max_stop_cap():
    """Verify stop loss is strictly clamped to max 4.8% even if ATR is very large."""
    client = AlpacaPaperTrader(max_stop_loss_pct=0.048)
    price = 100.0
    large_atr = 20.0  # 20% ATR -> Must be capped at 4.8% ($95.20)

    res = client.submit_bracket_order(
        symbol="TSLA",
        qty=10,
        side="BUY",
        current_price=price,
        atr=large_atr
    )
    assert pytest.approx(res.stop_price, 1e-2) == 95.20


def test_alpaca_client_execute_decision_workflow():
    """Verify execute_decision properly translates BUY, SELL, HOLD actions."""
    client = AlpacaPaperTrader()

    # BUY decision
    buy_res = client.execute_decision(
        symbol="AAPL",
        action="BUY",
        target_weight=0.25,
        current_price=180.0,
        atr=3.0
    )
    assert buy_res.side == "BUY"
    assert buy_res.qty > 0

    # HOLD decision
    hold_res = client.execute_decision(
        symbol="AAPL",
        action="HOLD",
        target_weight=0.0,
        current_price=180.0
    )
    assert hold_res.side == "HOLD"
    assert hold_res.status == "NO_ACTION"


# ============================================================================
# 6. Autonomous Dry-Run Simulator & Scheduler Tests
# ============================================================================

def test_dry_run_simulator_cycle_and_telemetry_files(tmp_path):
    """Verify DryRunSimulator runs an autonomous cycle and writes valid output telemetry JSONs."""
    out_dir = tmp_path / "output"
    sim = DryRunSimulator(output_dir=str(out_dir), initial_capital=100_000.0)

    dates = pd.date_range("2026-01-01", periods=40, freq="D", tz="UTC")
    synth_df = pd.DataFrame({
        "Open": np.linspace(100, 120, 40),
        "High": np.linspace(102, 122, 40),
        "Low": np.linspace(98, 118, 40),
        "Close": np.linspace(100, 120, 40),
        "Volume": np.full(40, 500_000)
    }, index=dates)

    report: ExecutionCycleReport = sim.run_cycle(data_dict={"NVDA": synth_df, "SPY": synth_df.copy()})

    assert report.status == "SUCCESS"
    assert report.account_equity > 0.0

    # Check that telemetry files exist on disk
    snap_file = out_dir / "current_signals_snapshot.json"
    exec_log_file = out_dir / "live_execution_log.json"

    assert snap_file.exists()
    assert exec_log_file.exists()

    with open(snap_file, "r") as f:
        data = json.load(f)
        assert "account_equity" in data
        assert "target_weights" in data
        assert data["is_dry_run"] is True


def test_scheduler_market_open_checks():
    """Verify ExecutionScheduler correctly identifies US / BVC market hours and weekend halts."""
    scheduler = ExecutionScheduler(force_market_open=False)

    # 1. Saturday check -> Closed
    sat_dt = datetime(2026, 8, 29, 15, 0, tzinfo=timezone.utc)  # Saturday
    assert not scheduler.is_market_open(market="US", current_dt=sat_dt)
    assert not scheduler.is_market_open(market="BVC", current_dt=sat_dt)

    # 2. Wednesday 15:00 UTC -> US open (13:30 - 21:00 UTC)
    wed_dt = datetime(2026, 8, 26, 15, 0, tzinfo=timezone.utc)
    assert scheduler.is_market_open(market="US", current_dt=wed_dt)

    # 3. Wednesday 10:00 UTC -> BVC open (08:30 - 14:30 UTC)
    wed_morning = datetime(2026, 8, 26, 10, 0, tzinfo=timezone.utc)
    assert scheduler.is_market_open(market="BVC", current_dt=wed_morning)

    # 4. Wednesday 22:00 UTC -> Both closed
    wed_night = datetime(2026, 8, 26, 22, 0, tzinfo=timezone.utc)
    assert not scheduler.is_market_open(market="US", current_dt=wed_night)
    assert not scheduler.is_market_open(market="BVC", current_dt=wed_night)


def test_scheduler_step_and_single_iteration(tmp_path):
    """Verify scheduler can execute single step without external API keys."""
    sim = DryRunSimulator(output_dir=str(tmp_path / "output"))
    scheduler = ExecutionScheduler(dry_run_simulator=sim, force_market_open=True)

    report = scheduler.step()
    assert report.is_dry_run is True
    assert report.account_equity > 0.0


def test_backtest_result_interface_contract_properties():
    """Verify BacktestResult exposes .equity_curve and .trades aliases."""
    dates = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    eq_series = pd.Series(np.linspace(100_000, 110_000, 10), index=dates)
    cash_series = pd.Series(np.linspace(50_000, 60_000, 10), index=dates)
    trade_df = pd.DataFrame([{"symbol": "NVDA", "side": "BUY", "shares": 10, "price": 100.0}])
    metrics = calculate_metrics(eq_series, trade_df)

    res = BacktestResult(
        equity_series=eq_series,
        cash_series=cash_series,
        positions_df=pd.DataFrame(),
        weights_df=pd.DataFrame(),
        trade_log=trade_df,
        metrics=metrics,
        benchmark_equity=eq_series,
        signals_dict={}
    )

    assert res.equity_curve.equals(eq_series)
    assert res.trades.equals(trade_df)
    summary = res.summary()
    assert "CAGR (%)" in summary
    assert "Sharpe Ratio" in summary


def test_intraday_engine_dnn_signals_target_weights_integration():
    """Verify IntradayBacktestEngine ingests DNN model output with target_weight column."""
    dates = pd.date_range("2026-01-01", periods=50, freq="D", tz="UTC")
    df_aapl = pd.DataFrame({
        "Open": np.linspace(150, 180, 50),
        "High": np.linspace(152, 182, 50),
        "Low": np.linspace(148, 178, 50),
        "Close": np.linspace(150, 180, 50),
        "Volume": np.full(50, 200_000)
    }, index=dates)

    # Mock DNN output DataFrame matching Interface Contract #3
    dnn_signals_aapl = pd.DataFrame({
        "action": ["BUY"] * 50,
        "buy_prob": [0.75] * 50,
        "hold_prob": [0.20] * 50,
        "sell_prob": [0.05] * 50,
        "target_weight": [0.30] * 50,
        "expected_return": [0.03] * 50,
        "confidence": [0.85] * 50
    }, index=dates)

    engine = IntradayBacktestEngine(initial_capital=100_000.0, rebalance_frequency=2)
    result = engine.run(
        data={"AAPL": df_aapl},
        signals={"AAPL": dnn_signals_aapl},
        initial_capital=100_000.0
    )

    assert isinstance(result, BacktestResult)
    assert result.equity_curve.iloc[-1] > 100_000.0
    assert len(result.trades) >= 1
    assert result.metrics.cagr > 0.0


def test_intraday_engine_run_simulation_alias():
    """Verify run_simulation wrapper executes properly."""
    dates = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
    df_msft = pd.DataFrame({
        "Open": np.linspace(300, 330, 30),
        "High": np.linspace(305, 335, 30),
        "Low": np.linspace(295, 325, 30),
        "Close": np.linspace(300, 330, 30),
        "Volume": np.full(30, 300_000)
    }, index=dates)

    target_weights = pd.DataFrame({"MSFT": [0.25] * 30}, index=dates)
    engine = IntradayBacktestEngine(initial_capital=50_000.0)
    result = engine.run_simulation(
        intraday_data={"MSFT": df_msft},
        target_weights=target_weights,
        initial_capital=50_000.0
    )

    assert result.metrics.total_return > 0.0


def test_almgren_chriss_sqrt_impact_power():
    """Verify square-root law (alpha = 0.5) for non-linear temporary impact."""
    cfg = AlmgrenChrissConfig(
        us_linear_impact_eta=0.04,
        us_impact_alpha=0.5,
        us_half_spread_bps=2.0
    )
    sim = OrderBookSimulator(cfg)

    # 4% participation rate: sqrt(0.04) = 0.20 -> temp_impact = 0.04 * 0.20 = 0.008
    part_rate, spread_rate, temp_imp, perm_imp = sim.calculate_market_impact(
        order_shares=400,
        bar_volume=10_000,
        market="US"
    )
    assert pytest.approx(part_rate, 1e-4) == 0.04
    assert pytest.approx(temp_imp, 1e-4) == 0.04 * np.sqrt(0.04)


def test_bvc_currency_conversion_and_reset():
    """Verify BVCAdapter currency conversion and state reset."""
    adapter = BVCAdapter(mad_usd_fx_rate=0.10)
    assert adapter.convert_mad_to_usd(1000.0) == 100.0
    assert adapter.convert_usd_to_mad(100.0) == 1000.0

    adapter.queue_settlement_proceeds("ord_1", "ATW.BVC", 500.0, 0)
    assert adapter.get_pending_settlement_total() == 500.0

    adapter.reset()
    assert adapter.get_pending_settlement_total() == 0.0
    assert adapter.cumulative_tax_paid_usd == 0.0


def test_scheduler_multiple_iterations_and_delay(tmp_path):
    """Verify scheduler can execute multiple iterations in sequence."""
    sim = DryRunSimulator(output_dir=str(tmp_path / "output"))
    scheduler = ExecutionScheduler(dry_run_simulator=sim, force_market_open=True)

    reports = scheduler.run_iterations(max_iterations=2, sleep_seconds=0)
    assert len(reports) == 2
    assert all(r.is_dry_run for r in reports)

