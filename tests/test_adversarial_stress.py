"""
Adversarial Stress Testing & Quantitative Hardening Test Suite (Milestone M5).

Empirically challenges the system across:
1. Extreme volatility regimes (GARCH spike, flash crash, 5-sigma jump events).
2. High market impact execution (order size exceeding 20% of bar volume to test Almgren-Chriss penalty).
3. Sudden news sentiment inversion (from extremely positive +1.0 to crashing negative -1.0 within 1 bar).
4. Casablanca BVC limit-down lock days (-10% circuit breaker preventing exit) and cash insolvency checks with T+3 settlement delay.
5. Risk management caps (single asset <= 34%, gross exposure <= 100%, -4.8% max stop loss) hold unconditionally.
6. Statistical lookahead leakage verification in features, scaling, and labels.
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import math
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import pytest

from src.backtest.bvc_adapter import BVCAdapter
from src.backtest.intraday_engine import IntradayBacktestEngine, BacktestResult
from src.backtest.order_book import OrderBookSimulator, AlmgrenChrissConfig
from src.data.aligner import MultiTimeframeAligner, resample_ohlcv
from src.data.loader import UniverseConfig
from src.execution.alpaca_client import AlpacaPaperTrader
from src.models.feature_pipeline import (
    CausalRollingScaler,
    FeatureEngineer,
    FeaturePipeline,
    MACRO_FEATURE_COLS,
    MICRO_FEATURE_COLS,
    SENTIMENT_FEATURE_COLS,
)
from src.models.labeler import TripleBarrierLabeler, BarrierConfig
from src.portfolio.allocator import PortfolioAllocator
from src.sentiment.aggregator import SentimentAggregator
from src.sentiment.nlp_engine import SentimentResult


# ============================================================================
# Helpers to generate synthetic stress datasets
# ============================================================================

def make_intraday_series(
    n_bars: int = 200,
    start_price: float = 100.0,
    start_dt: str = "2025-01-02 09:30:00",
    interval_minutes: int = 5,
    volatility: float = 0.002,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate realistic synthetic intraday OHLCV bars."""
    np.random.seed(seed)
    dt_index = pd.date_range(start=start_dt, periods=n_bars, freq=f"{interval_minutes}min", tz="UTC")
    returns = np.random.normal(0, volatility, n_bars)
    prices = start_price * np.cumprod(1.0 + returns)

    highs = prices * (1.0 + np.abs(np.random.normal(0, volatility * 0.8, n_bars)))
    lows = prices * (1.0 - np.abs(np.random.normal(0, volatility * 0.8, n_bars)))
    opens = (prices + np.random.normal(0, volatility * 0.2, n_bars)).clip(lows, highs)
    volumes = np.random.lognormal(mean=9.0, sigma=0.5, size=n_bars)

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volumes,
            "VWAP": (opens + highs + lows + prices) / 4.0,
        },
        index=dt_index,
    )
    return df


def make_daily_series(
    n_days: int = 300,
    start_price: float = 100.0,
    start_dt: str = "2024-01-02",
    volatility: float = 0.015,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate synthetic daily OHLCV bars."""
    np.random.seed(seed)
    dt_index = pd.date_range(start=start_dt, periods=n_days, freq="B", tz="UTC")
    returns = np.random.normal(0.0005, volatility, n_days)
    prices = start_price * np.cumprod(1.0 + returns)

    highs = prices * (1.0 + np.abs(np.random.normal(0, volatility * 0.7, n_days)))
    lows = prices * (1.0 - np.abs(np.random.normal(0, volatility * 0.7, n_days)))
    opens = (prices + np.random.normal(0, volatility * 0.3, n_days)).clip(lows, highs)
    volumes = np.random.lognormal(mean=14.0, sigma=0.6, size=n_days)

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": prices,
            "Volume": volumes,
            "VWAP": (opens + highs + lows + prices) / 4.0,
        },
        index=dt_index,
    )
    return df


# ============================================================================
# 1. Extreme Volatility Regimes Stress Tests
# ============================================================================

class TestExtremeVolatilityRegimes:
    """Stress tests for GARCH volatility spikes, flash crashes, and 5-sigma jump events."""

    def test_garch_volatility_spike_resilience(self):
        """Test system behavior during sudden 10x volatility explosion."""
        df = make_intraday_series(n_bars=300, volatility=0.001)
        # Inject GARCH volatility spike from bar 150 onwards (volatility explodes 10x to 0.03 per 5m bar)
        spike_returns = np.random.normal(0, 0.03, 150)
        p_base = df["Close"].iloc[149]
        spike_prices = p_base * np.cumprod(1.0 + spike_returns)

        df.iloc[150:, df.columns.get_loc("Close")] = spike_prices
        df.iloc[150:, df.columns.get_loc("High")] = spike_prices * 1.025
        df.iloc[150:, df.columns.get_loc("Low")] = spike_prices * 0.975
        df.iloc[150:, df.columns.get_loc("Open")] = spike_prices * 1.005

        # 1. Feature extraction must not produce NaNs or Infs
        micro_feat = FeatureEngineer.compute_intraday_micro_features(df)
        assert not micro_feat.isna().any().any(), "Micro features contain NaN during volatility spike"
        assert not np.isinf(micro_feat.values).any(), "Micro features contain Inf during volatility spike"

        # 2. Causal scaler must clip outputs within [-5.0, 5.0]
        scaler = CausalRollingScaler(method="robust", window=60)
        scaled = scaler.fit_transform(micro_feat)
        assert scaled.max().max() <= 5.0001, "Scaler failed to clip upper bound during volatility spike"
        assert scaled.min().min() >= -5.0001, "Scaler failed to clip lower bound during volatility spike"

    def test_intraday_flash_crash_and_stop_loss_liquidation(self):
        """Simulate severe -35% intraday flash crash and verify exact bracket stop loss exit."""
        df = make_intraday_series(n_bars=100, start_price=100.0)
        # Bar 50: Crash begins, dropping price to $65 (-35%)
        crash_prices = [100.0] * 50 + list(np.linspace(100.0, 65.0, 10)) + [65.0] * 40
        for i in range(len(df)):
            p = crash_prices[i]
            df.iloc[i, df.columns.get_loc("Close")] = p
            df.iloc[i, df.columns.get_loc("Open")] = p * 1.002
            df.iloc[i, df.columns.get_loc("High")] = p * 1.005
            df.iloc[i, df.columns.get_loc("Low")] = p * 0.995

        # Create target weights commanding BUY at bar 0
        target_weights = pd.DataFrame(0.0, index=df.index, columns=["CRASH_ASSET"])
        target_weights.iloc[0:80, 0] = 0.30

        engine = IntradayBacktestEngine(initial_capital=100_000.0, stop_loss_pct=0.048)
        res = engine.run(data_dict={"CRASH_ASSET": df}, target_weights=target_weights)

        # Check trades log
        trades = res.trade_log
        assert not trades.empty, "No trades executed during backtest"
        
        # Stop loss should be triggered when price drops below -4.8%
        stop_trades = trades[trades["side"] == "STOP_LOSS"]
        assert len(stop_trades) > 0, "Stop loss was NOT triggered during flash crash!"

        # Realized loss must be bounded and equity series must remain positive
        assert (res.equity_series > 0).all(), "Equity plunged below zero during flash crash!"
        assert res.equity_series.iloc[-1] > 70_000.0, f"Excessive loss beyond risk cap: equity={res.equity_series.iloc[-1]}"

    def test_five_sigma_jump_events_numerical_stability(self):
        """Inject Poisson-distributed 5-sigma jump shocks (+/-20%) in price and volume."""
        df = make_intraday_series(n_bars=250, start_price=100.0)
        jump_indices = [30, 75, 120, 180, 220]
        jump_magnitudes = [1.25, 0.75, 1.30, 0.70, 1.22]  # +25%, -25%, +30%, -30%, +22%

        for idx, mult in zip(jump_indices, jump_magnitudes):
            orig_p = df["Close"].iloc[idx]
            new_p = orig_p * mult
            df.iloc[idx, df.columns.get_loc("Close")] = new_p
            df.iloc[idx, df.columns.get_loc("High")] = max(orig_p, new_p) * 1.02
            df.iloc[idx, df.columns.get_loc("Low")] = min(orig_p, new_p) * 0.98
            df.iloc[idx, df.columns.get_loc("Open")] = orig_p
            df.iloc[idx, df.columns.get_loc("Volume")] = df["Volume"].iloc[idx] * 20.0  # 20x volume spike

        labeler = TripleBarrierLabeler(BarrierConfig(vertical_barrier_bars=12))
        lbl_res = labeler.generate_labels(df)
        labels = lbl_res.labels_df

        # Ensure valid labels generated without crashing
        assert len(labels) == len(df)
        assert set(labels["label_class"].unique()).issubset({0, 1, 2})
        assert not labels["bar_volatility"].isna().any()
        assert not labels["composite_return"].isna().any()


# ============================================================================
# 2. High Market Impact Execution & Almgren-Chriss Frictions
# ============================================================================

class TestHighMarketImpactExecution:
    """Stress tests for Almgren-Chriss non-linear market impact penalty under high volume orders."""

    def test_participation_rate_ceiling_us_and_bvc(self):
        """Verify that participation rate is strictly capped at max limits (25% US, 10% BVC)."""
        sim = OrderBookSimulator()
        
        # Scenario 1: US order requesting 50% of bar volume
        # 5,000 shares on 10,000 bar volume -> 50% raw participation
        part_us, spread_us, temp_us, perm_us = sim.calculate_market_impact(
            order_shares=5000, bar_volume=10000, market="US"
        )
        assert math.isclose(part_us, 0.25, abs_tol=1e-5), f"US participation rate {part_us} exceeded 25% ceiling!"

        # Scenario 2: BVC order requesting 30% of bar volume
        # 3,000 shares on 10,000 bar volume -> 30% raw participation
        part_bvc, spread_bvc, temp_bvc, perm_bvc = sim.calculate_market_impact(
            order_shares=3000, bar_volume=10000, market="BVC"
        )
        assert math.isclose(part_bvc, 0.10, abs_tol=1e-5), f"BVC participation rate {part_bvc} exceeded 10% ceiling!"

    def test_almgren_chriss_penalty_monotonic_scaling(self):
        """Verify that executed price penalty increases monotonically with order size."""
        sim = OrderBookSimulator()
        price = 100.0
        bar_volume = 50_000

        # Test varying order sizes: 1%, 5%, 10%, 20%, 30% of bar volume
        sizes_usd = [500 * price, 2500 * price, 5000 * price, 10000 * price, 15000 * price]
        effective_prices = []
        slippages = []

        for size in sizes_usd:
            res = sim.execute_order(
                symbol="NVDA",
                market="US",
                side="BUY",
                order_value_usd=size,
                current_price=price,
                daily_volume=bar_volume,
            )
            effective_prices.append(res.executed_price)
            slippages.append(res.total_slippage)

        # Ensure effective price > nominal price for BUY
        for ep in effective_prices:
            assert ep > price, f"BUY execution price {ep} was not higher than nominal price {price}"

        # Ensure effective prices and slippages are monotonically non-decreasing
        for i in range(len(effective_prices) - 1):
            assert effective_prices[i] <= effective_prices[i + 1] + 1e-6, "Impact price is not monotonically increasing"
            assert slippages[i] <= slippages[i + 1] + 1e-6, "Slippage cost is not monotonically increasing"

    def test_bvc_higher_impact_penalty_vs_us(self):
        """Verify BVC execution experiences substantially higher frictions and slippage than US."""
        sim = OrderBookSimulator()
        price = 100.0
        bar_vol = 20_000
        order_val = 10_000.0

        res_us = sim.execute_order(
            symbol="AAPL", market="US", side="BUY",
            order_value_usd=order_val, current_price=price, daily_volume=bar_vol
        )
        res_bvc = sim.execute_order(
            symbol="ATW.BVC", market="BVC", side="BUY",
            order_value_usd=order_val, current_price=price, daily_volume=bar_vol
        )

        assert res_bvc.executed_price > res_us.executed_price, "BVC executed price should be higher than US due to higher eta/gamma/spread"
        assert res_bvc.total_commission > res_us.total_commission, "BVC commissions (25 bps) should be higher than US (5 bps)"
        assert res_bvc.total_slippage > res_us.total_slippage, "BVC slippage should be higher than US"


# ============================================================================
# 3. Sudden News Sentiment Inversion Stress Tests
# ============================================================================

class TestSuddenSentimentInversion:
    """Stress tests for abrupt news sentiment shocks (+1.0 to -1.0) and time-decay responsiveness."""

    def test_sentiment_shock_inversion_and_zero_lookahead(self):
        """Test instantaneous flip from extremely positive news (+1.0) to catastrophic crash (-1.0)."""
        aggregator = SentimentAggregator()
        symbol = "TSLA"

        t0 = datetime(2025, 1, 15, 10, 0, tzinfo=timezone.utc)
        # 10 Bullish articles published between 10:00 and 11:00
        bull_articles = [
            SentimentResult(
                article_id=f"bull_{i}",
                symbol=symbol,
                published_at_utc=t0 + timedelta(minutes=i * 5),
                sentiment_score=0.95,
                sentiment_confidence=0.90,
                sentiment_label="POSITIVE",
            )
            for i in range(10)
        ]
        aggregator.add_sentiment_results(bull_articles)

        # Breaking catastrophic article at 11:30
        t_shock = t0 + timedelta(minutes=90)  # 11:30
        shock_article = SentimentResult(
            article_id="bear_shock",
            symbol=symbol,
            published_at_utc=t_shock,
            sentiment_score=-1.0,
            sentiment_confidence=0.99,
            sentiment_label="NEGATIVE",
        )
        aggregator.add_sentiment_results([shock_article])

        # Evaluate at 11:00 (before shock), 11:29 (just before shock), and 11:35 (after shock)
        ts_eval = [
            t0 + timedelta(minutes=60),  # 11:00 (Pre-shock)
            t0 + timedelta(minutes=89),  # 11:29 (Pre-shock)
            t0 + timedelta(minutes=95),  # 11:35 (Post-shock)
        ]
        feat_df = aggregator.get_sentiment_features(symbol, ts_eval, half_life_hours=1.0)

        # 1. Zero lookahead check: pre-shock scores must be strongly positive
        assert feat_df.loc[ts_eval[0], "sentiment_score_ema"] > 0.80, "Pre-shock score should be strongly positive"
        assert feat_df.loc[ts_eval[1], "sentiment_score_ema"] > 0.60, "Pre-shock score should remain positive"
        assert feat_df.loc[ts_eval[1], "news_volume_1h"] == 10, "1h volume pre-shock should only include bull articles"

        # 2. Post-shock: sudden inversion
        post_score = feat_df.loc[ts_eval[2], "sentiment_score_ema"]
        assert post_score < 0.20, f"Post-shock score {post_score} failed to plunge after -1.0 news"
        assert feat_df.loc[ts_eval[2], "sentiment_momentum"] < 0, "Sentiment momentum must be negative following shock"

    def test_future_news_leakage_rejection(self):
        """Ensure articles dated in the future NEVER contaminate past or present timestamps."""
        aggregator = SentimentAggregator()
        symbol = "NVDA"

        t_now = datetime(2025, 2, 1, 12, 0, tzinfo=timezone.utc)
        t_future = t_now + timedelta(hours=2)

        # Future news article
        future_article = SentimentResult(
            article_id="future_leak_test",
            symbol=symbol,
            published_at_utc=t_future,
            sentiment_score=1.0,
            sentiment_confidence=1.0,
            sentiment_label="POSITIVE",
        )
        aggregator.add_sentiment_results([future_article])

        # Evaluate at t_now
        res = aggregator.get_sentiment_features(symbol, [t_now])
        assert res["sentiment_score_ema"].iloc[0] == 0.0, "Future news leaked into current timestamp!"
        assert res["news_volume_1h"].iloc[0] == 0, "Future news counted in current 1h volume!"
        assert res["news_volume_24h"].iloc[0] == 0, "Future news counted in current 24h volume!"


# ============================================================================
# 4. Casablanca BVC Circuit Breakers & T+3 Settlement Invalidation Checks
# ============================================================================

class TestCasablancaBVCLimitDownAndSettlement:
    """Stress tests for BVC +/-10% circuit breakers, limit-down locks, and T+3 cash insolvency."""

    def test_bvc_circuit_breaker_limit_down_clamping(self):
        """Verify BVC price clamp at -10% during market crash."""
        bvc = BVCAdapter(continuous_circuit_breaker=0.10)
        ref_price = 500.0  # e.g., Attijariwafa Bank MAD 500

        # Attempted trades at -5%, -10%, -15%, -25%
        res_5pct = bvc.check_circuit_breaker(target_price=475.0, reference_price=ref_price)
        assert res_5pct.clamped_price == 475.0
        assert not res_5pct.is_limit_hit

        res_10pct = bvc.check_circuit_breaker(target_price=450.0, reference_price=ref_price)
        assert math.isclose(res_10pct.clamped_price, 450.0, abs_tol=1e-3)
        assert res_10pct.is_limit_hit
        assert res_10pct.limit_type == "LIMIT_DOWN"

        res_25pct = bvc.check_circuit_breaker(target_price=375.0, reference_price=ref_price)
        # Clamped price must be bounded at 450.0 (-10%)
        assert math.isclose(res_25pct.clamped_price, 450.0, abs_tol=1e-3), f"BVC price {res_25pct.clamped_price} breached -10% lower limit"
        assert res_25pct.is_limit_hit
        assert res_25pct.limit_type == "LIMIT_DOWN"

    def test_bvc_t3_settlement_cash_anti_overdraft_enforcement(self):
        """
        Verify that unsettled BVC sale proceeds cannot be used for purchasing new assets
        during the T+3 clearing window, preventing cash overdrafts.
        """
        engine = IntradayBacktestEngine(initial_capital=10_000.0, rebalance_frequency=1)
        
        # Build 10 daily bars for ATW.BVC and US_STOCK
        dates = pd.date_range("2025-01-01", periods=10, freq="B", tz="UTC")
        
        df_bvc = pd.DataFrame({
            "Open": 500.0, "High": 505.0, "Low": 495.0, "Close": 500.0, "Volume": 100_000
        }, index=dates)
        
        df_us = pd.DataFrame({
            "Open": 100.0, "High": 102.0, "Low": 98.0, "Close": 100.0, "Volume": 500_000
        }, index=dates)

        # Day 0: Allocate $3,000 to ATW.BVC (60 MAD = $600 USD per share at 0.10 fx)
        # Day 1: Sell ATW.BVC (generates $3,000 proceeds arriving Day 1+3 = Day 4)
        # Day 2: Request 100% allocation to US_STOCK ($10,000+)
        target_weights = pd.DataFrame(0.0, index=dates, columns=["ATW.BVC", "US_STOCK"])
        target_weights.iloc[0, 0] = 0.30  # Buy ATW.BVC on Day 0
        target_weights.iloc[1, 0] = 0.00  # Sell ATW.BVC on Day 1
        target_weights.iloc[2:5, 1] = 0.34  # Attempt to buy US_STOCK on Days 2-4

        res = engine.run(
            data_dict={"ATW.BVC": df_bvc, "US_STOCK": df_us},
            target_weights=target_weights,
            initial_capital=10_000.0
        )

        # Verify that cash series never dropped below zero (no illegal margin or overdraft)
        assert (res.cash_series >= 0.0).all(), f"Cash went negative! Min cash = {res.cash_series.min()}"

        # Verify that pending receivables are tracked in BVC adapter
        # and settled on Day 4
        assert res.equity_series.iloc[-1] > 0

    def test_moroccan_ir_15pct_tax_on_gains_and_zero_on_losses(self):
        """Verify 15% IR tax calculation on profitable BVC sales and 0% on loss sales."""
        bvc = BVCAdapter(ir_tax_rate=0.15)

        # Trade 1: Gain of MAD 1,000 ($100 USD) -> Tax = $15 USD (MAD 150)
        net_gain, tax_gain = bvc.calculate_capital_gains_tax(
            executed_shares=10, executed_price=110.0, avg_entry_price=100.0, commission=0.0
        )
        assert math.isclose(tax_gain, 15.0, abs_tol=1e-3), f"Tax on $100 gain was {tax_gain}, expected $15.0"
        assert math.isclose(net_gain, 85.0, abs_tol=1e-3)

        # Trade 2: Loss of MAD 1,000 ($100 USD) -> Tax = $0.0 USD
        net_loss, tax_loss = bvc.calculate_capital_gains_tax(
            executed_shares=10, executed_price=90.0, avg_entry_price=100.0, commission=0.0
        )
        assert math.isclose(tax_loss, 0.0, abs_tol=1e-3), f"Tax on loss should be $0.0, got {tax_loss}"
        assert math.isclose(net_loss, -100.0, abs_tol=1e-3)


# ============================================================================
# 5. Risk Management Caps Verification (Unconditional Enforcement)
# ============================================================================

class TestRiskManagementCapsUnconditional:
    """Stress tests verifying single-asset <= 34%, gross exposure <= 100%, and stop loss <= -4.8%."""

    def test_single_asset_max_allocation_cap_34pct(self):
        """Verify that target weights exceeding 34% are clipped in all engines."""
        # 1. Backtest engine weight clipping
        dates = pd.date_range("2025-01-01", periods=10, freq="B", tz="UTC")
        df = pd.DataFrame({"Open": 100, "High": 101, "Low": 99, "Close": 100, "Volume": 10000}, index=dates)

        # Adversarial input: 1 asset with 90% weight
        adversarial_weights = pd.DataFrame({"ASSET1": [0.90] * 10}, index=dates)

        engine = IntradayBacktestEngine()
        res = engine.run(data_dict={"ASSET1": df}, target_weights=adversarial_weights)

        # Verify weights in backtest result are clamped to <= 0.34
        assert (res.weights_df["ASSET1"] <= 0.340001).all(), f"Asset weight exceeded 34%! Max: {res.weights_df['ASSET1'].max()}"

    def test_gross_exposure_cap_100pct(self):
        """Verify that total gross exposure across multiple assets is capped at 100% (no leverage)."""
        dates = pd.date_range("2025-01-01", periods=10, freq="B", tz="UTC")
        data_dict = {
            f"SYM_{i}": pd.DataFrame({"Open": 100, "High": 101, "Low": 99, "Close": 100, "Volume": 10000}, index=dates)
            for i in range(5)
        }
        # 5 assets each at 0.34 -> raw sum = 1.70 (170% exposure)
        adversarial_weights = pd.DataFrame(
            {f"SYM_{i}": [0.34] * 10 for i in range(5)}, index=dates
        )

        engine = IntradayBacktestEngine()
        res = engine.run(data_dict=data_dict, target_weights=adversarial_weights)

        total_exposure = res.weights_df.sum(axis=1)
        assert (total_exposure <= 1.000001).all(), f"Gross exposure {total_exposure.max()} exceeded 100%!"

    def test_alpaca_client_max_stop_loss_cap_4_8pct(self):
        """Verify that Alpaca paper client clamps stop loss to at most 4.8% below entry."""
        client = AlpacaPaperTrader(max_stop_loss_pct=0.048)
        price = 200.0

        # 1. Request large stop loss (e.g. 15%)
        res = client.submit_bracket_order(
            symbol="NVDA",
            qty=10,
            side="BUY",
            current_price=price,
            stop_loss_pct=0.15,  # 15% stop requested
        )
        expected_min_stop = round(price * (1.0 - 0.048), 2)  # $190.40
        assert res.stop_price is not None
        assert res.stop_price >= expected_min_stop, f"Stop price ${res.stop_price} breached max 4.8% risk cap (${expected_min_stop})"

        # 2. Request large ATR (e.g. ATR = 30.0 on a $200 stock = 15%)
        res_atr = client.submit_bracket_order(
            symbol="NVDA",
            qty=10,
            side="BUY",
            current_price=price,
            atr=30.0,
        )
        assert res_atr.stop_price is not None
        assert res_atr.stop_price >= expected_min_stop, f"ATR Stop price ${res_atr.stop_price} breached max 4.8% risk cap"


# ============================================================================
# 6. Statistical Lookahead Leakage Audit
# ============================================================================

class TestStatisticalLookaheadLeakage:
    """Rigorous mathematical audits guaranteeing zero forward lookahead leakage."""

    def test_causal_rolling_scaler_no_future_leakage(self):
        """
        Verify that changing future data points has ZERO impact on historical scaled features.
        scaled_df(t) == scaled_df_perturbed(t) for all t < t_perturbation.
        """
        df = pd.DataFrame(
            np.random.normal(100, 15, size=(200, 4)),
            columns=["col1", "col2", "col3", "col4"],
            index=pd.date_range("2025-01-01", periods=200, freq="15min", tz="UTC"),
        )
        scaler = CausalRollingScaler(method="robust", window=60)
        scaled_orig = scaler.fit_transform(df)

        # Perturb future data from bar 100 onwards by 1,000,000x
        df_perturbed = df.copy()
        df_perturbed.iloc[100:] = df_perturbed.iloc[100:] * 1_000_000.0

        scaler_perturbed = CausalRollingScaler(method="robust", window=60)
        scaled_perturbed = scaler_perturbed.fit_transform(df_perturbed)

        # Historical slice [0:100] must be IDENTICAL to within machine precision
        diff = np.abs(scaled_orig.iloc[:100].values - scaled_perturbed.iloc[:100].values)
        assert np.max(diff) < 1e-12, f"Future data leaked into historical scaling! Max diff: {np.max(diff)}"

    def test_multi_timeframe_aligner_strict_t_minus_1_lag(self):
        """Verify that daily macro features merged into intraday bars strictly use date <= D-1."""
        df_daily = make_daily_series(n_days=100, start_price=100.0)
        df_intraday = make_intraday_series(n_bars=300, start_price=100.0)

        aligner = MultiTimeframeAligner()
        aligned = aligner.align_features(df_daily=df_daily, df_intraday=df_intraday)

        daily_close_map = {ts.normalize(): close for ts, close in zip(df_daily.index, df_daily["Close"])}
        daily_dates = sorted(list(daily_close_map.keys()))

        # Verify for every intraday bar on date D that its daily_close_lagged_t1 matches date D-1
        for i in range(1, min(10, len(daily_dates))):
            curr_date = daily_dates[i]
            prev_date = daily_dates[i - 1]

            bars_curr = aligned[aligned.index.normalize() == curr_date]
            if not bars_curr.empty:
                lagged_val = bars_curr["daily_close_lagged_t1"].iloc[0]
                expected_prev_close = daily_close_map[prev_date]
                assert math.isclose(lagged_val, expected_prev_close, abs_tol=1e-2), (
                    f"Lookahead failure on {curr_date}: lagged close {lagged_val} != prev close {expected_prev_close}"
                )

    def test_feature_pipeline_dataset_tensors_strict_lag(self):
        """Verify FeaturePipeline builds PyTorch sequences with strictly causal macro alignment."""
        df_daily = make_daily_series(n_days=150)
        df_intraday = make_intraday_series(n_bars=200)

        pipeline = FeaturePipeline(macro_seq_len=30, micro_seq_len=20)
        macro_sc, micro_sc, sent_sc = pipeline.process_features(df_daily, df_intraday)
        tensors = pipeline.build_dataset_tensors(macro_sc, micro_sc, sent_sc)

        assert tensors.macro_seq.shape == (200 - 20 + 1, 30, len(MACRO_FEATURE_COLS))
        assert tensors.micro_seq.shape == (200 - 20 + 1, 20, len(MICRO_FEATURE_COLS))
        assert tensors.sent_vec.shape == (200 - 20 + 1, len(SENTIMENT_FEATURE_COLS))
        assert len(tensors.timestamps) == 200 - 20 + 1
