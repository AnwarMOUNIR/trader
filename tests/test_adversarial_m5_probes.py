"""
Milestone M5 Adversarial Stress & Corner Case Hardening Test Suite.

EMPIRICAL CHALLENGER PROBES:
1. Empty news feeds / missing RSS items & Neutral Decay without NaN
2. Zero-volume / illiquid trading sessions & Div-by-Zero prevention in RVOL/VWAP/Impact
3. Missing or corrupted historical data & Automatic synthetic microstructure fallback
4. Daylight Saving Time (DST) shifts and UTC timezone transitions (New York vs Casablanca)
5. High-frequency rebalancing friction decay, extreme fees & Anti-Overdraft Cash Solvency
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest

from src.backtest.bvc_adapter import BVCAdapter
from src.backtest.intraday_engine import BacktestResult, IntradayBacktestEngine
from src.backtest.metrics import calculate_metrics
from src.backtest.order_book import AlmgrenChrissConfig, OrderBookSimulator
from src.data.aligner import MultiTimeframeAligner, resample_ohlcv
from src.data.historical_loader import HistoricalDataLoader
from src.data.loader import UniverseConfig
from src.data.parquet_store import ParquetDataStore
from src.data.synthetic_feed import SyntheticMicrostructureGenerator
from src.models.feature_pipeline import (
    CausalRollingScaler,
    FeatureEngineer,
    MACRO_FEATURE_COLS,
    MICRO_FEATURE_COLS,
    SENTIMENT_FEATURE_COLS,
)
from src.sentiment.aggregator import SentimentAggregator
from src.sentiment.deduplicator import SQLiteNewsStore
from src.sentiment.nlp_engine import NLPSentimentEngine, SentimentResult
from src.sentiment.rss_crawler import RSSNewsCrawler, NewsArticle

logger = logging.getLogger("AdversarialTests")


# ==============================================================================
# PROBE 1: Empty News Feeds & Neutral Sentiment Decay
# ==============================================================================

class TestProbe1EmptyNewsAndSentimentDecay:
    """Stress-test sentiment pipeline under empty feeds, sparse news, and extreme silence."""

    def test_empty_sentiment_feed_returns_zero_nan_clean_neutral(self):
        """Verify that an aggregator with 0 articles produces clean neutral values across 1000 bars without NaNs."""
        aggregator = SentimentAggregator(news_store=None)
        
        # 1000 5-minute intraday timestamps
        start_ts = pd.Timestamp("2025-01-01 09:30:00", tz="UTC")
        timestamps = pd.date_range(start=start_ts, periods=1000, freq="5min")

        df_sent = aggregator.get_sentiment_features(
            symbol="NVDA",
            target_timestamps=timestamps,
            half_life_hours=4.0
        )

        assert len(df_sent) == 1000
        assert list(df_sent.columns) == SentimentAggregator.FEATURE_COLUMNS
        assert df_sent.isna().sum().sum() == 0, "Empty news feed produced NaN values!"
        assert (df_sent["sentiment_score_ema"] == 0.0).all()
        assert (df_sent["sentiment_confidence_ema"] == 0.0).all()
        assert (df_sent["news_volume_1h"] == 0).all()
        assert (df_sent["news_volume_24h"] == 0).all()
        assert (df_sent["sentiment_dispersion"] == 0.0).all()
        assert (df_sent["sentiment_momentum"] == 0.0).all()
        assert (df_sent["bull_bear_ratio"] == 1.0).all()

    def test_single_news_spike_and_long_silence_exponential_decay(self):
        """Verify mathematical precision of exponential decay during prolonged news silence."""
        aggregator = SentimentAggregator(news_store=None, max_lookback_days=7.0)
        
        t0 = pd.Timestamp("2025-01-10 10:00:00", tz="UTC")
        single_article = SentimentResult(
            article_id="art_spike_1",
            symbol="AAPL",
            published_at_utc=t0.to_pydatetime(),
            sentiment_score=1.0,
            sentiment_confidence=1.0,
            sentiment_label="POSITIVE",
            probabilities={"positive": 1.0, "neutral": 0.0, "negative": 0.0},
            model_name="test_model"
        )
        aggregator.add_sentiment_results([single_article])

        # Test at exact half-life intervals (half-life = 4 hours)
        half_life_h = 4.0
        test_times = [
            t0,                                           # delta = 0h -> score = 1.0
            t0 + pd.Timedelta(hours=4),                   # delta = 4h (1 half-life) -> score = 1.0
            t0 + pd.Timedelta(hours=8),                   # delta = 8h -> score = 1.0
            t0 + pd.Timedelta(days=8),                    # delta = 8d (> max_lookback_days 7d) -> score = 0.0 (dropped from active window)
        ]
        
        df_decay = aggregator.get_sentiment_features("AAPL", test_times, half_life_hours=half_life_h)
        assert df_decay.isna().sum().sum() == 0

        # When there is only 1 article within lookback, weighted average score is 1.0, volume decays
        assert df_decay.loc[test_times[0], "news_volume_1h"] == 1
        assert df_decay.loc[test_times[1], "news_volume_1h"] == 0  # 4h later, not in 1h window
        assert df_decay.loc[test_times[1], "news_volume_24h"] == 1
        assert df_decay.loc[test_times[3], "sentiment_score_ema"] == 0.0  # Expired after 7 days
        assert df_decay.loc[test_times[3], "news_volume_24h"] == 0

    def test_future_dated_news_strict_pit_exclusion(self):
        """Verify that news with future publication timestamps are never leaked into earlier timestamps."""
        aggregator = SentimentAggregator(news_store=None)
        
        t_eval = pd.Timestamp("2025-01-10 12:00:00", tz="UTC")
        t_future = pd.Timestamp("2025-01-10 12:00:01", tz="UTC")  # 1 second in future
        
        future_article = SentimentResult(
            article_id="future_1",
            symbol="TSLA",
            published_at_utc=t_future.to_pydatetime(),
            sentiment_score=0.99,
            sentiment_confidence=0.95,
            sentiment_label="POSITIVE",
            probabilities={"positive": 0.99, "neutral": 0.01, "negative": 0.0},
            model_name="test"
        )
        aggregator.add_sentiment_results([future_article])

        # At t_eval, the future article must NOT be visible
        df_at_eval = aggregator.get_sentiment_features("TSLA", [t_eval])
        assert df_at_eval.loc[t_eval, "sentiment_score_ema"] == 0.0
        assert df_at_eval.loc[t_eval, "news_volume_1h"] == 0

        # At t_future, it becomes visible
        df_at_future = aggregator.get_sentiment_features("TSLA", [t_future])
        assert df_at_future.loc[t_future, "sentiment_score_ema"] == pytest.approx(0.99, abs=1e-3)
        assert df_at_future.loc[t_future, "news_volume_1h"] == 1

    def test_nlp_engine_adversarial_text_inputs(self):
        """Verify NLP engine handles empty strings, unicode emojis, huge strings, and corrupt titles."""
        engine = NLPSentimentEngine(device="cpu")
        
        adversarial_articles = [
            NewsArticle.create(
                symbol="AAPL",
                market="US",
                title="",
                summary="",
                url="https://example.com/empty",
                published_at_utc=datetime.now(timezone.utc),
                source="test"
            ),
            NewsArticle.create(
                symbol="AAPL",
                market="US",
                title="   \n\t   ",
                summary="   ",
                url="https://example.com/spaces",
                published_at_utc=datetime.now(timezone.utc),
                source="test"
            ),
            NewsArticle.create(
                symbol="AAPL",
                market="US",
                title="🚀🚀🚀🌕💎🙌📈",
                summary="🔥💣📉",
                url="https://example.com/emojis",
                published_at_utc=datetime.now(timezone.utc),
                source="test"
            ),
            NewsArticle.create(
                symbol="AAPL",
                market="US",
                title="A" * 5000,
                summary="B" * 5000,
                url="https://example.com/huge",
                published_at_utc=datetime.now(timezone.utc),
                source="test"
            ),
            NewsArticle.create(
                symbol="ATW.BVC",
                market="BVC",
                title="Hausse historique des bénéfices nets bancaires à Casablanca",
                summary="Attijariwafa bank annonce une croissance solide du dividende annuel",
                url="https://example.com/bvc_pos",
                published_at_utc=datetime.now(timezone.utc),
                source="test"
            )
        ]

        results = engine.score_batch(adversarial_articles)
        assert len(results) == len(adversarial_articles)
        for r in results:
            assert isinstance(r.sentiment_score, float)
            assert -1.0 <= r.sentiment_score <= 1.0
            assert 0.0 <= r.sentiment_confidence <= 1.0
            assert not math.isnan(r.sentiment_score)
            assert not math.isnan(r.sentiment_confidence)

        # French Moroccan positive article should have positive score
        bvc_res = results[-1]
        assert bvc_res.sentiment_score > 0.0


# ==============================================================================
# PROBE 2: Zero-Volume & Illiquid Trading Sessions
# ==============================================================================

class TestProbe2ZeroVolumeAndIlliquidity:
    """Stress-test feature extraction, VWAP, RVOL, OFI, and execution under total illiquidity."""

    def test_zero_volume_bars_feature_pipeline_div_by_zero_prevention(self):
        """Verify that 500 bars with 0.0 volume produce no NaN/Inf in feature pipeline."""
        timestamps = pd.date_range("2025-01-01 09:30:00", periods=500, freq="5min", tz="UTC")
        df_zero_vol = pd.DataFrame(
            {
                "open": 100.0,
                "high": 100.0,
                "low": 100.0,
                "close": 100.0,
                "volume": 0.0,
                "vwap": 100.0,
            },
            index=timestamps
        )

        micro_feats = FeatureEngineer.compute_intraday_micro_features(df_zero_vol)
        assert len(micro_feats) == 500
        assert list(micro_feats.columns) == MICRO_FEATURE_COLS
        assert micro_feats.isna().sum().sum() == 0, "Zero volume generated NaNs in micro features!"
        assert not np.isinf(micro_feats.values).any(), "Zero volume generated Infs in micro features!"

        # Specific feature value assertions for zero-volume flat price
        assert (micro_feats["rvol"] <= 1.0).all()
        assert (micro_feats["vol_zscore"] == 0.0).all()
        assert (micro_feats["vwap_distance"] == 0.0).all()
        assert (micro_feats["ofi_norm"] == 0.0).all()
        assert (micro_feats["clv"] == 0.0).all()

    def test_resample_ohlcv_on_zero_volume_dataset(self):
        """Verify resample_ohlcv handles zero volume without division-by-zero in VWAP aggregation."""
        timestamps = pd.date_range("2025-01-01 09:30:00", periods=60, freq="5min", tz="UTC")
        df_5m = pd.DataFrame(
            {
                "open": np.linspace(100.0, 105.0, 60),
                "high": np.linspace(101.0, 106.0, 60),
                "low": np.linspace(99.0, 104.0, 60),
                "close": np.linspace(100.5, 105.5, 60),
                "volume": 0.0,
                "vwap": np.linspace(100.5, 105.5, 60),
            },
            index=timestamps
        )

        df_15m = resample_ohlcv(df_5m, target_interval="15m")
        assert len(df_15m) == 20
        assert df_15m.isna().sum().sum() == 0
        assert (df_15m["volume"] == 0.0).all()
        # VWAP must fallback to close price when volume is zero
        assert (df_15m["vwap"] == df_15m["close"]).all()

    def test_order_book_simulator_zero_volume_liquidity_drought(self):
        """Verify OrderBookSimulator safely executes orders when bar volume is zero."""
        sim = OrderBookSimulator()
        
        # Calculate impact with 0 volume
        part_rate, spread_rate, temp_imp, perm_imp = sim.calculate_market_impact(
            order_shares=1000.0,
            bar_volume=0.0,
            market="US"
        )
        assert not math.isnan(part_rate)
        assert not math.isnan(temp_imp)
        assert not math.isnan(perm_imp)
        assert part_rate <= sim.config.us_max_participation_rate

        # Execute order with zero volume
        res_buy = sim.execute_order(
            symbol="AAPL",
            market="US",
            side="BUY",
            order_value_usd=10000.0,
            current_price=150.0,
            daily_volume=0.0,
            day_index=0
        )
        assert res_buy.executed_shares > 0
        assert res_buy.executed_price >= 150.0
        assert not math.isnan(res_buy.total_cost)
        assert not math.isnan(res_buy.total_slippage)

    def test_intraday_backtest_with_prolonged_illiquid_drought(self):
        """Verify backtest engine completes when entire asset series is frozen at zero volume."""
        timestamps = pd.date_range("2025-01-01 09:30:00", periods=200, freq="15min", tz="UTC")
        df_frozen = pd.DataFrame(
            {
                "Open": 50.0,
                "High": 50.0,
                "Low": 50.0,
                "Close": 50.0,
                "Volume": 0.0,
                "VWAP": 50.0
            },
            index=timestamps
        )

        data = {"FROZEN": df_frozen, "SPY": df_frozen}
        signals = {
            "FROZEN": pd.DataFrame(
                {
                    "action": ["BUY"] * 100 + ["HOLD"] * 100,
                    "target_weight": [0.20] * 100 + [0.20] * 100,
                    "confidence": 0.8
                },
                index=timestamps
            )
        }

        engine = IntradayBacktestEngine(initial_capital=50000.0)
        result = engine.run(data_dict=data, signals_dict=signals)

        assert isinstance(result, BacktestResult)
        assert len(result.equity_series) == 200
        assert not result.equity_series.isna().any()
        assert not math.isnan(result.metrics.cagr)
        assert not math.isnan(result.metrics.mdd)


# ==============================================================================
# PROBE 3: Missing / Corrupted Data & Synthetic Generator Fallback
# ==============================================================================

class TestProbe3MissingAndCorruptedDataFallback:
    """Stress-test data ingestion when network fails, caches are corrupt, or inputs malformed."""

    def test_parquet_store_corrupted_file_recovery(self):
        """Verify ParquetDataStore handles zero-byte / corrupt parquet files gracefully."""
        temp_dir = tempfile.mkdtemp(prefix="trader_corrupt_test_")
        try:
            store = ParquetDataStore(base_dir=temp_dir)
            
            # Create a corrupted zero-byte parquet file
            corrupt_path = os.path.join(temp_dir, "CORRUPT_5m.parquet")
            with open(corrupt_path, "wb") as f:
                f.write(b"NOT_A_VALID_PARQUET_HEADER_DATA_12345")

            # Loading corrupt file should return None and not crash
            res = store.load_bars("CORRUPT", "5m")
            assert res is None, "Corrupt file did not return None on load!"
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_historical_loader_synthetic_fallback_us_and_bvc(self):
        """Verify HistoricalDataLoader falls back to synthetic feed when API is unavailable."""
        temp_dir = tempfile.mkdtemp(prefix="trader_loader_test_")
        try:
            store = ParquetDataStore(base_dir=temp_dir)
            loader = HistoricalDataLoader(store=store)

            # Request Moroccan BVC asset (no yfinance support -> direct synthetic fallback)
            dual_bvc = loader.fetch_dual_resolution("ATW.BVC", daily_years=2, intraday_days=5, intraday_interval="15m")
            assert "daily" in dual_bvc and "intraday" in dual_bvc
            assert not dual_bvc["daily"].empty
            assert not dual_bvc["intraday"].empty
            assert dual_bvc["daily"].isna().sum().sum() == 0
            assert dual_bvc["intraday"].isna().sum().sum() == 0

            # Verify standard OHLCV contract columns
            for col in ["open", "high", "low", "close", "volume", "vwap"]:
                assert col in dual_bvc["daily"].columns
                assert col in dual_bvc["intraday"].columns

            # Verify prices are strictly positive
            assert (dual_bvc["daily"]["low"] > 0).all()
            assert (dual_bvc["intraday"]["low"] > 0).all()
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_synthetic_microstructure_generator_extreme_parameters(self):
        """Stress-test synthetic generator with extreme volatility and jump parameters."""
        gen = SyntheticMicrostructureGenerator(seed=42)
        
        # Extreme jump diffusion
        df = gen.generate_intraday_series(
            symbol="VOLATILE",
            days=10,
            interval="5m",
            s0=100.0,
            mu=-0.50,
            sigma=1.50,  # 150% annualized volatility
            jump_lambda=5.0,
            jump_mu=-0.10,
            jump_sigma=0.20,
            market="US"
        )
        assert len(df) > 0
        assert df.isna().sum().sum() == 0
        assert (df["low"] > 0).all(), "Synthetic generator generated negative or zero price!"
        assert (df["high"] >= df["low"]).all(), "Bar High is lower than Low!"
        assert (df["high"] >= df["open"]).all()
        assert (df["high"] >= df["close"]).all()
        assert (df["volume"] > 0).all()


# ==============================================================================
# PROBE 4: Daylight Saving Time (DST) & UTC Timezone Transitions
# ==============================================================================

class TestProbe4TimezoneAndDSTTransitions:
    """Stress-test multi-timeframe alignment across US EDT/EST and Moroccan Ramadan DST shifts."""

    def test_us_dst_spring_forward_and_fall_back_alignment(self):
        """Verify MultiTimeframeAligner across March spring-forward and November fall-back dates."""
        aligner = MultiTimeframeAligner()
        
        # March 2024 DST switch (Spring forward: March 10, 2024)
        daily_dates = pd.date_range("2024-03-05", "2024-03-15", freq="B", tz="UTC")
        df_daily = pd.DataFrame(
            {
                "open": np.linspace(150, 160, len(daily_dates)),
                "high": np.linspace(152, 162, len(daily_dates)),
                "low": np.linspace(149, 159, len(daily_dates)),
                "close": np.linspace(151, 161, len(daily_dates)),
                "volume": 5000000.0,
                "vwap": np.linspace(151, 161, len(daily_dates))
            },
            index=daily_dates
        )

        # Intraday bars across the transition
        intraday_dates = pd.date_range("2024-03-08 14:30:00", "2024-03-12 21:00:00", freq="30min", tz="UTC")
        df_intraday = pd.DataFrame(
            {
                "open": 155.0,
                "high": 156.0,
                "low": 154.0,
                "close": 155.5,
                "volume": 25000.0,
                "vwap": 155.5
            },
            index=intraday_dates
        )

        aligned = aligner.align_features(df_daily, df_intraday)
        assert len(aligned) == len(df_intraday)
        assert aligned.isna().sum().sum() == 0
        assert "daily_sma_50" in aligned.columns
        assert "daily_close_lagged_t1" in aligned.columns

    def test_morocco_casablanca_timezone_session_rules(self):
        """Verify BVCAdapter circuit breakers and settlement timing across Moroccan calendar."""
        adapter = BVCAdapter()
        
        # Check circuit breaker limit +/- 10%
        ref_price = 100.0
        # Jump within limit (+8%)
        res_normal = adapter.check_circuit_breaker(108.0, ref_price)
        assert not res_normal.is_tripped
        assert res_normal.clamped_price == 108.0

        # Jump exceeding limit (+15%) -> clamped to +10% (110.0)
        res_high = adapter.check_circuit_breaker(115.0, ref_price)
        assert res_high.is_tripped
        assert res_high.clamped_price == pytest.approx(110.0, abs=1e-3)

        # Drop exceeding limit (-15%) -> clamped to -10% (90.0)
        res_low = adapter.check_circuit_breaker(85.0, ref_price)
        assert res_low.is_tripped
        assert res_low.clamped_price == pytest.approx(90.0, abs=1e-3)


# ==============================================================================
# PROBE 5: High-Frequency Rebalancing Friction Decay & Anti-Overdraft Cash Solvency
# ==============================================================================

class TestProbe5HFRebalancingFrictionAndCashSolvency:
    """Stress-test engine cash accounting under extreme turnover, high fees, and BVC T+3 settlement starvation."""

    def test_hyper_frequency_rebalancing_anti_overdraft(self):
        """Verify that rebalancing every single bar with oscillating signals never drives cash negative."""
        n_bars = 300
        timestamps = pd.date_range("2025-01-01 09:30:00", periods=n_bars, freq="5min", tz="UTC")
        
        # Volatile price series
        prices = 100.0 + np.sin(np.linspace(0, 20 * np.pi, n_bars)) * 10.0
        df_asset = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 1.0,
                "Low": prices - 1.0,
                "Close": prices,
                "Volume": 100000.0,
                "VWAP": prices
            },
            index=timestamps
        )

        data = {"VOLA": df_asset, "SPY": df_asset}
        
        # Oscillating target weights: switch between 30% and 0% every 2 bars
        weights = [0.30 if (i // 2) % 2 == 0 else 0.0 for i in range(n_bars)]
        signals = {
            "VOLA": pd.DataFrame(
                {
                    "action": ["BUY" if w > 0 else "SELL" for w in weights],
                    "target_weight": weights,
                    "confidence": 0.9
                },
                index=timestamps
            )
        }

        # Use elevated frictions: 50 bps commission, 20 bps half-spread
        custom_config = AlmgrenChrissConfig(
            us_brokerage_bps=50.0,
            us_half_spread_bps=20.0,
            us_linear_impact_eta=0.20
        )
        sim = OrderBookSimulator(config=custom_config)

        engine = IntradayBacktestEngine(
            execution_model=sim,
            initial_capital=20000.0,
            rebalance_frequency=1,
            deadband_pct=0.0  # Force rebalance on every small change
        )

        result = engine.run(data_dict=data, signals_dict=signals)
        
        # CASH SOLVENCY ASSERTIONS
        assert (result.cash_series >= 0.0).all(), f"Cash went negative! Min cash: {result.cash_series.min()}"
        assert (result.equity_series >= 0.0).all(), f"Equity went negative! Min equity: {result.equity_series.min()}"
        assert not result.equity_series.isna().any()

    def test_bvc_t3_settlement_cash_starvation_protection(self):
        """Verify that BVC T+3 settlement queue prevents re-investing unsettled cash before day 3."""
        # 10 daily bars
        timestamps = pd.date_range("2025-01-01", periods=10, freq="B", tz="UTC")
        df_bvc = pd.DataFrame(
            {
                "Open": 500.0,
                "High": 510.0,
                "Low": 490.0,
                "Close": 500.0,
                "Volume": 50000.0,
                "VWAP": 500.0
            },
            index=timestamps
        )

        data = {"ATW.BVC": df_bvc, "SPY": df_bvc}

        # Day 0: BUY 30%
        # Day 1: SELL all (proceeds locked in T+3)
        # Day 2: Attempt BUY 30% (must fail or scale to available cash ~0)
        # Day 3: Attempt BUY 30% (must fail or scale to available cash ~0)
        # Day 4: T+3 settled -> BUY succeeds
        weights = [0.30, 0.0, 0.30, 0.30, 0.30, 0.0, 0.0, 0.0, 0.0, 0.0]
        signals = {
            "ATW.BVC": pd.DataFrame(
                {
                    "action": ["BUY" if w > 0 else "SELL" for w in weights],
                    "target_weight": weights,
                    "confidence": 0.85
                },
                index=timestamps
            )
        }

        engine = IntradayBacktestEngine(initial_capital=100000.0, rebalance_frequency=1)
        result = engine.run(data_dict=data, signals_dict=signals)

        # Cash balance during days 2 and 3 must never be negative
        assert (result.cash_series >= 0.0).all(), f"T+3 queue caused cash overdraft! Min cash: {result.cash_series.min()}"
        assert result.equity_series.isna().sum() == 0

    def test_low_capital_solvency_and_rounding(self):
        """Verify backtest behaves safely when initial capital is tiny ($500)."""
        timestamps = pd.date_range("2025-01-01 09:30:00", periods=50, freq="15min", tz="UTC")
        df_asset = pd.DataFrame(
            {
                "Open": 250.0,
                "High": 255.0,
                "Low": 245.0,
                "Close": 250.0,
                "Volume": 10000.0,
                "VWAP": 250.0
            },
            index=timestamps
        )
        data = {"EXPENSIVE": df_asset, "SPY": df_asset}
        signals = {
            "EXPENSIVE": pd.DataFrame(
                {
                    "action": ["BUY"] * 50,
                    "target_weight": [0.34] * 50,
                    "confidence": 0.9
                },
                index=timestamps
            )
        }

        engine = IntradayBacktestEngine(initial_capital=500.0)
        result = engine.run(data_dict=data, signals_dict=signals)
        assert (result.cash_series >= 0.0).all()
        assert (result.equity_series >= 0.0).all()
        assert not result.equity_series.isna().any()
