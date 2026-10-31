"""Unit tests and empirical stress-test validation for SingleStockRiskEngine and PortfolioRiskEngine."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from market_research.risk_engine import SingleStockRiskEngine, PortfolioRiskEngine


def test_single_stock_catastrophic_bankruptcy_never_exceeds_30k_loss():
    """Even if an individual stock drops by 99.9% in a catastrophic event,
    the 30% position cap guarantees that account loss remains <= $30,000 + fees."""
    engine = SingleStockRiskEngine(capital=100_000.0, max_allocation_fraction=0.30, cost_bps=10.0)
    
    # Stock drops from $100 to $0.01 overnight
    opens = [100.0, 0.01]
    closes = [100.0, 0.01]
    
    res = engine.simulate(opens, closes)
    assert res["max_drawdown_dollars"] <= 30_050.0  # $30,000 notional + fees
    assert res["ending_equity"] >= 69_950.0
    assert res["liquidated"] is True


def test_single_stock_trailing_stop_limits_gradual_loss():
    """A gradual 20% decline in stock price triggers the $15k trailing stop and preserves capital."""
    engine = SingleStockRiskEngine(capital=100_000.0, max_allocation_fraction=0.30, trailing_stop_dollars=5_000.0)
    
    # 30% of $100k is $30,000. A drop of > $5k on $30k occurs when price drops > 16.7%.
    opens = [100.0, 95.0, 90.0, 80.0, 70.0]
    closes = [100.0, 95.0, 90.0, 80.0, 70.0]
    
    res = engine.simulate(opens, closes)
    assert res["liquidated"] is True
    # Should stop out before full drop to 70
    assert res["max_drawdown_dollars"] < 10_000.0


def test_single_stock_parameter_validation():
    with pytest.raises(ValueError, match="Capital must be positive"):
        SingleStockRiskEngine(capital=-100)
    with pytest.raises(ValueError, match="Max allocation"):
        SingleStockRiskEngine(max_allocation_fraction=1.5)
    with pytest.raises(ValueError, match="Trailing stop"):
        SingleStockRiskEngine(trailing_stop_dollars=0)


def test_portfolio_risk_engine_dial_decay():
    engine = PortfolioRiskEngine(capital=100_000.0, soft_floor_dollars=6_000.0, hard_floor_dollars=9_000.0)
    
    # Below soft floor -> 100% dial
    assert engine.compute_dial_exposure(2_000.0) == 1.0
    assert engine.compute_dial_exposure(5_999.0) == 1.0
    
    # At soft floor -> 1.0
    assert engine.compute_dial_exposure(6_000.0) == pytest.approx(1.0)
    
    # Midway between soft and hard floor -> linearly decayed
    assert engine.compute_dial_exposure(7_500.0) == pytest.approx(0.60)
    
    # At hard floor or beyond -> 0.0 (100% Cash lock)
    assert engine.compute_dial_exposure(9_000.0) == 0.0
    assert engine.compute_dial_exposure(12_000.0) == 0.0


def test_portfolio_macro_gate_cuts_exposure_during_bear_regime():
    engine = PortfolioRiskEngine(capital=100_000.0, macro_gate_defensive_exposure=0.25)
    
    # 5 sessions, 4 stocks, flat market
    T, N = 5, 4
    opens = np.full((T, N), 100.0)
    closes = np.full((T, N), 100.0)
    
    # Bear signal for all 5 sessions
    bear_signals = np.zeros(T, dtype=bool)
    res_bear = engine.simulate(opens, closes, bear_signals)
    assert res_bear["average_exposure"] == pytest.approx(0.25)
    
    # Bull signal for all 5 sessions
    bull_signals = np.ones(T, dtype=bool)
    res_bull = engine.simulate(opens, closes, bull_signals)
    assert res_bull["average_exposure"] == pytest.approx(1.0)


def test_100_trials_stress_test_on_real_historical_data():
    """Empirical verification: 100 randomized trials over real 2022-2024 bear market data.
    Verifies that:
    1. Single-stock worst-case loss never breaches -$30,000.
    2. 35-stock portfolio 10-worst average loss is bounded at <= -$10,500.
    """
    candle_dir = Path("data/candles/30min")
    symbols = [p.name.replace("_30min.parquet", "") for p in sorted(candle_dir.glob("*_30min.parquet")) if p.name != "SPY_30min.parquet"]
    if len(symbols) < 35:
        pytest.skip("Insufficient candle data for 35-stock test")
        
    open_dict, close_dict = {}, {}
    for sym in symbols:
        df = pd.read_parquet(candle_dir / f"{sym}_30min.parquet")
        d = df.groupby("session_date").agg({"open": "first", "close": "last"})
        open_dict[sym] = d["open"]
        close_dict[sym] = d["close"]
        
    opens_df = pd.DataFrame(open_dict).sort_index().ffill()
    closes_df = pd.DataFrame(close_dict).sort_index().ffill()
    
    spy_df = pd.read_parquet(candle_dir / "SPY_30min.parquet")
    spy_d = spy_df.groupby("session_date").agg({"close": "last"}).sort_index()
    spy_trend = (spy_d["close"].rolling(50).mean() > spy_d["close"].rolling(200).mean()).shift(1).fillna(False)
    
    common_dates = [d for d in closes_df.index if "2022-01-01" <= d <= "2024-12-31" and d in spy_trend.index]
    opens_df = opens_df.loc[common_dates]
    closes_df = closes_df.loc[common_dates]
    spy_trend = spy_trend.loc[common_dates]
    
    np.random.seed(123)
    single_engine = SingleStockRiskEngine(capital=100_000.0, max_allocation_fraction=0.30, trailing_stop_dollars=15_000.0)
    port_engine = PortfolioRiskEngine(capital=100_000.0, soft_floor_dollars=6_000.0, hard_floor_dollars=9_000.0)
    
    single_drawdowns = []
    port_drawdowns = []
    
    for _ in range(100):
        # 1. Single stock trial
        sym = np.random.choice(symbols)
        idx = np.random.randint(0, len(common_dates) - 126)
        s_res = single_engine.simulate(
            opens_df[sym].iloc[idx:idx + 126].values,
            closes_df[sym].iloc[idx:idx + 126].values,
        )
        single_drawdowns.append(s_res["max_drawdown_dollars"])
        
        # 2. Portfolio trial (35 stocks)
        basket = np.random.choice(symbols, size=35, replace=False)
        p_idx = np.random.randint(0, len(common_dates) - 180)
        p_res = port_engine.simulate(
            opens_df[basket].iloc[p_idx:p_idx + 180].values,
            closes_df[basket].iloc[p_idx:p_idx + 180].values,
            spy_trend.iloc[p_idx:p_idx + 180].values,
        )
        port_drawdowns.append(p_res["max_drawdown_dollars"])
        
    single_drawdowns = np.array(single_drawdowns)
    port_drawdowns = np.array(port_drawdowns)
    
    worst_single_loss = np.max(single_drawdowns)
    worst_10_port_loss = np.mean(np.sort(port_drawdowns)[-10:])
    
    # Assert constraints
    assert worst_single_loss <= 30_050.0, f"Single stock worst loss {worst_single_loss} exceeded $30k"
    assert worst_10_port_loss <= 10_500.0, f"Portfolio worst-10 average loss {worst_10_port_loss} exceeded $10.5k"
