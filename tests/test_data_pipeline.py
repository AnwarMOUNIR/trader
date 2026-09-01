"""
Comprehensive Unit & Integration Test Suite for Milestone M1 Data Pipeline.
Covers:
- Synthetic Microstructure Generator (Jump-Diffusion, U-curve volume, BVC circuit breakers)
- Columnar Parquet Storage (ZSTD compression, Column pruning, SQLite metadata catalog)
- Historical Data Loader (30Y daily + 5m-60m intraday dual resolution, caching, fallback)
- Multi-Timeframe Aligner & Resampler (Zero lookahead verification, vectorized OHLCV resampling)
"""

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pytest

from src.data import (
    ASSET_PRESETS,
    AssetMarket,
    HistoricalDataLoader,
    MarketMicrostructureParams,
    MultiTimeframeAligner,
    ParquetDataStore,
    SyntheticMicrostructureGenerator,
    UniverseConfig,
    resample_ohlcv,
)


@pytest.fixture
def temp_cache_dir():
    """Create and clean up a temporary directory for Parquet data and SQLite catalog."""
    temp_dir = tempfile.mkdtemp(prefix="test_data_cache_")
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_daily_df():
    """Generate a deterministic 2-year daily OHLCV dataset."""
    generator = SyntheticMicrostructureGenerator(seed=42)
    return generator.generate_daily_series(
        symbol="AAPL",
        start_date="2024-01-01",
        end_date="2025-12-31",
        market="US"
    )


@pytest.fixture
def sample_intraday_df():
    """Generate a deterministic 10-day 5m intraday OHLCV dataset."""
    generator = SyntheticMicrostructureGenerator(seed=42)
    return generator.generate_intraday_series(
        symbol="AAPL",
        days=10,
        interval="5m",
        market="US"
    )


# ============================================================================
# 1. Synthetic Microstructure Generator Tests
# ============================================================================

def test_synthetic_generator_daily_properties(sample_daily_df):
    """Test mathematical and structural properties of synthetic daily series."""
    df = sample_daily_df
    assert not df.empty
    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.tz is not None
    assert str(df.index.tz) == "UTC"

    required_cols = ["open", "high", "low", "close", "volume", "vwap", "trade_count", "market", "currency"]
    for col in required_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Price positivity
    assert (df["open"] > 0).all()
    assert (df["high"] > 0).all()
    assert (df["low"] > 0).all()
    assert (df["close"] > 0).all()
    assert (df["volume"] > 0).all()

    # OHLC Consistency
    assert (df["high"] >= df["open"]).all()
    assert (df["high"] >= df["close"]).all()
    assert (df["low"] <= df["open"]).all()
    assert (df["low"] <= df["close"]).all()


def test_synthetic_generator_reproducibility():
    """Test that seed guarantees identical outputs."""
    gen1 = SyntheticMicrostructureGenerator(seed=12345)
    gen2 = SyntheticMicrostructureGenerator(seed=12345)

    df1 = gen1.generate_daily_series("NVDA", start_date="2025-01-01", end_date="2025-06-01")
    df2 = gen2.generate_daily_series("NVDA", start_date="2025-01-01", end_date="2025-06-01")

    pd.testing.assert_frame_equal(df1, df2)


def test_synthetic_generator_bvc_circuit_breaker():
    """Test that Casablanca BVC returns respect regulatory +/-10% circuit breaker."""
    gen = SyntheticMicrostructureGenerator(seed=999)
    df_bvc = gen.generate_daily_series("ATW.BVC", start_date="2020-01-01", end_date="2025-01-01", market="BVC")

    assert df_bvc["currency"].iloc[0] == "MAD"
    assert df_bvc["market"].iloc[0] == "BVC"

    daily_pct_changes = df_bvc["close"].pct_change().dropna()
    # Ensure no single day exceeds 10.5% return (allowing minor rounding buffer)
    assert (daily_pct_changes <= 0.105).all(), f"Max positive return exceeded 10%: {daily_pct_changes.max()}"
    assert (daily_pct_changes >= -0.105).all(), f"Max negative return exceeded -10%: {daily_pct_changes.min()}"


def test_synthetic_generator_u_curve_volume_profile():
    """Test empirical U-shaped intraday volume profile."""
    gen = SyntheticMicrostructureGenerator()
    profile = gen.generate_intraday_volume_profile(78, a=3.5, b=0.35)

    assert len(profile) == 78
    assert np.isclose(np.sum(profile), 1.0, atol=1e-5)

    # U-shape verification: Open (first bar) and Close (last bar) should have higher volume than midday
    open_vol = profile[0]
    close_vol = profile[-1]
    mid_vol = profile[39]

    assert open_vol > mid_vol, f"Open volume {open_vol} should be > mid volume {mid_vol}"
    assert close_vol > mid_vol, f"Close volume {close_vol} should be > mid volume {mid_vol}"


def test_synthetic_generator_intraday_sessions():
    """Test US and BVC session bar counts and intervals."""
    gen = SyntheticMicrostructureGenerator(seed=55)

    # US 5m: 6.5 hours = 78 bars per day
    df_us = gen.generate_intraday_series("AAPL", days=2, interval="5m", market="US")
    assert len(df_us) == 78 * 2

    # Casablanca BVC 5m: 6.0 hours = 72 bars per day
    df_bvc = gen.generate_intraday_series("ATW.BVC", days=2, interval="5m", market="BVC")
    assert len(df_bvc) == 72 * 2


# ============================================================================
# 2. Parquet Data Store & SQLite Catalog Tests
# ============================================================================

def test_parquet_store_save_load_roundtrip(temp_cache_dir, sample_daily_df):
    """Test storing and loading Parquet files with ZSTD compression and metadata recording."""
    store = ParquetDataStore(base_dir=temp_cache_dir)
    symbol = "NVDA"
    timeframe = "daily"

    # Store
    file_path_str = store.store_bars(
        symbol=symbol,
        timeframe=timeframe,
        df=sample_daily_df,
        market="US",
        currency="USD",
        name="NVIDIA Corp",
        sector="Technology"
    )
    assert Path(file_path_str).exists()

    # Load full
    loaded_df = store.load_bars(symbol, timeframe)
    assert loaded_df is not None
    assert len(loaded_df) == len(sample_daily_df)
    assert list(loaded_df.columns) == list(sample_daily_df.columns)

    # Verify Checksum
    assert store.verify_checksum(symbol, timeframe) is True

    # Check Manifest Entry
    entry = store.get_manifest_entry(symbol, timeframe)
    assert entry is not None
    assert entry["symbol"] == "NVDA"
    assert entry["timeframe"] == "daily"
    assert entry["total_bars"] == len(sample_daily_df)
    assert entry["file_size_bytes"] > 0

    # Check Asset Registry
    catalog = store.list_catalog(market="US")
    assert len(catalog) == 1
    assert catalog[0]["symbol"] == "NVDA"
    assert catalog[0]["currency"] == "USD"


def test_parquet_store_column_pruning_and_date_filtering(temp_cache_dir, sample_daily_df):
    """Test efficient columnar loading with column selection and date slicing."""
    store = ParquetDataStore(base_dir=temp_cache_dir)
    store.store_bars("AAPL", "daily", sample_daily_df)

    # Column pruning
    pruned_df = store.load_bars("AAPL", "daily", columns=["close", "volume"])
    assert pruned_df is not None
    assert set(pruned_df.columns) == {"close", "volume"}

    # Date range filtering
    start_filter = "2024-06-01"
    end_filter = "2024-08-31"
    sliced_df = store.load_bars("AAPL", "daily", start=start_filter, end=end_filter)
    assert sliced_df is not None
    assert sliced_df.index.min() >= pd.to_datetime(start_filter, utc=True)
    assert sliced_df.index.max() <= pd.to_datetime(end_filter, utc=True)


def test_parquet_store_deletion(temp_cache_dir, sample_daily_df):
    """Test deletion of bars and manifest entries."""
    store = ParquetDataStore(base_dir=temp_cache_dir)
    store.store_bars("MSFT", "daily", sample_daily_df)
    assert store.has_bars("MSFT", "daily") is True

    # Delete specific timeframe
    store.delete_bars("MSFT", timeframe="daily")
    assert store.has_bars("MSFT", "daily") is False
    assert store.get_manifest_entry("MSFT", "daily") is None


def test_parquet_store_error_handling(temp_cache_dir):
    """Test error handling for non-existent assets and empty data."""
    store = ParquetDataStore(base_dir=temp_cache_dir)
    
    # Non-existent bar loading returns None
    assert store.load_bars("UNKNOWN_TICKER", "daily") is None
    assert store.has_bars("UNKNOWN_TICKER", "daily") is False

    # Storing empty dataframe raises ValueError
    with pytest.raises(ValueError, match="Cannot store empty DataFrame"):
        store.store_bars("EMPTY", "daily", pd.DataFrame())


# ============================================================================
# 3. Vectorized OHLCV Resampling Tests
# ============================================================================

def test_resample_ohlcv_accuracy(sample_intraday_df):
    """Test that resample_ohlcv aggregates 5m bars to 15m and 60m correctly."""
    df_5m = sample_intraday_df

    # Resample to 15m (each 15m bar combines 3 x 5m bars)
    df_15m = resample_ohlcv(df_5m, target_interval="15m")
    assert not df_15m.empty
    assert len(df_15m) == len(df_5m) // 3

    # Verify first 15m bar
    first_3_5m = df_5m.iloc[:3]
    first_15m = df_15m.iloc[0]

    assert first_15m["open"] == first_3_5m["open"].iloc[0]
    assert first_15m["close"] == first_3_5m["close"].iloc[-1]
    assert first_15m["high"] == first_3_5m["high"].max()
    assert first_15m["low"] == first_3_5m["low"].min()
    assert np.isclose(first_15m["volume"], first_3_5m["volume"].sum(), atol=1e-2)

    # Resample to 60m
    df_60m = resample_ohlcv(df_5m, target_interval="60m")
    assert not df_60m.empty
    assert len(df_60m) < len(df_15m)


# ============================================================================
# 4. Multi-Timeframe Aligner & Zero-Lookahead Leakage Tests
# ============================================================================

def test_multi_timeframe_aligner_features(sample_daily_df, sample_intraday_df):
    """Test calculation and merging of macro and micro features."""
    aligner = MultiTimeframeAligner()
    aligned_df = aligner.align_features(sample_daily_df, sample_intraday_df)

    assert not aligned_df.empty
    assert len(aligned_df) == len(sample_intraday_df)

    # Micro features presence
    assert "rvol_20" in aligned_df.columns
    assert "vol_zscore_30" in aligned_df.columns
    assert "vwap_deviation" in aligned_df.columns
    assert "ofi_proxy" in aligned_df.columns
    assert "clv" in aligned_df.columns

    # Lagged Macro features presence
    assert "daily_close_lagged_t1" in aligned_df.columns
    assert "daily_sma_50" in aligned_df.columns
    assert "daily_rsi_14" in aligned_df.columns
    assert "daily_vol_20d" in aligned_df.columns
    assert "daily_mom_12m" in aligned_df.columns


def test_multi_timeframe_aligner_strict_zero_lookahead():
    """
    CRITICAL TEST: Mathematically proves zero future lookahead leakage.
    Ensures that for day T intraday bars, the daily close from day T is NEVER visible,
    and the lagged daily close strictly matches day T-1.
    """
    # Create precise 5-day daily series with distinct recognizable closing prices
    dates_daily = pd.date_range("2025-01-06", periods=5, freq="B", tz="UTC")
    df_daily = pd.DataFrame({
        "open": [100.0, 110.0, 120.0, 130.0, 140.0],
        "high": [105.0, 115.0, 125.0, 135.0, 145.0],
        "low": [95.0, 105.0, 115.0, 125.0, 135.0],
        "close": [102.0, 112.0, 122.0, 132.0, 142.0],  # Day 0: 102, Day 1: 112, Day 2: 122, ...
        "volume": [1e6, 1e6, 1e6, 1e6, 1e6]
    }, index=dates_daily)

    # Create intraday bars for Day 1 through Day 4 (2 bars per day)
    intraday_times = []
    for day in dates_daily[1:]:
        intraday_times.append(day.replace(hour=14, minute=0))
        intraday_times.append(day.replace(hour=19, minute=0))

    df_intraday = pd.DataFrame({
        "open": [110.0, 111.0, 120.0, 121.0, 130.0, 131.0, 140.0, 141.0],
        "high": [112.0, 113.0, 122.0, 123.0, 132.0, 133.0, 142.0, 143.0],
        "low": [109.0, 110.0, 119.0, 120.0, 129.0, 130.0, 139.0, 140.0],
        "close": [111.0, 112.0, 121.0, 122.0, 131.0, 132.0, 141.0, 142.0],
        "volume": [5000, 5000, 5000, 5000, 5000, 5000, 5000, 5000],
        "vwap": [110.5, 111.5, 120.5, 121.5, 130.5, 131.5, 140.5, 141.5]
    }, index=pd.DatetimeIndex(intraday_times, tz="UTC"))

    aligner = MultiTimeframeAligner()
    aligned = aligner.align_features(df_daily, df_intraday)

    # Check Day 1 bars (2025-01-07):
    # Today's daily close is 112.0. The lagged close MUST be 102.0 (Day 0 close).
    day1_bars = aligned[aligned.index.normalize() == pd.to_datetime("2025-01-07", utc=True)]
    assert len(day1_bars) == 2
    assert (day1_bars["daily_close_lagged_t1"] == 102.0).all()
    assert (day1_bars["daily_close_lagged_t1"] != 112.0).all()

    # Check Day 2 bars (2025-01-08):
    # Today's daily close is 122.0. The lagged close MUST be 112.0 (Day 1 close).
    day2_bars = aligned[aligned.index.normalize() == pd.to_datetime("2025-01-08", utc=True)]
    assert len(day2_bars) == 2
    assert (day2_bars["daily_close_lagged_t1"] == 112.0).all()
    assert (day2_bars["daily_close_lagged_t1"] != 122.0).all()


# ============================================================================
# 5. Historical Data Loader Integration Tests
# ============================================================================

def test_historical_loader_dual_resolution_contract(temp_cache_dir):
    """Test HistoricalDataLoader master dual-resolution interface contract."""
    store = ParquetDataStore(base_dir=temp_cache_dir)
    loader = HistoricalDataLoader(store=store, rate_limit_delay_sec=0.01)

    result = loader.fetch_dual_resolution(
        symbol="NVDA",
        daily_years=5,
        intraday_days=5,
        intraday_interval="5m",
        use_cache=True
    )

    assert "daily" in result
    assert "intraday" in result

    df_daily = result["daily"]
    df_intraday = result["intraday"]

    # Daily checks
    assert not df_daily.empty
    assert len(df_daily) >= 250 * 4
    assert isinstance(df_daily.index, pd.DatetimeIndex)
    for col in ["open", "high", "low", "close", "volume", "vwap"]:
        assert col in df_daily.columns

    # Intraday checks
    assert not df_intraday.empty
    assert len(df_intraday) >= 78 * 4
    assert isinstance(df_intraday.index, pd.DatetimeIndex)
    for col in ["open", "high", "low", "close", "volume", "vwap"]:
        assert col in df_intraday.columns

    # Parquet cache verification
    assert store.has_bars("NVDA", "daily") is True
    assert store.has_bars("NVDA", "5m") is True


def test_historical_loader_30_year_daily_depth(temp_cache_dir):
    """Test fetching full 30 years of daily historical data."""
    store = ParquetDataStore(base_dir=temp_cache_dir)
    loader = HistoricalDataLoader(store=store, rate_limit_delay_sec=0.01)

    df_30y = loader.fetch_daily("AAPL", years=30, use_cache=True)
    assert not df_30y.empty
    # ~30 years * ~252 trading days = ~7,500 bars
    assert len(df_30y) >= 7000
    assert df_30y.index.min().year <= 1996


def test_historical_loader_moroccan_bvc_universe(temp_cache_dir):
    """Test Moroccan Casablanca BVC equity dual-resolution fetching and currency handling."""
    store = ParquetDataStore(base_dir=temp_cache_dir)
    loader = HistoricalDataLoader(store=store, rate_limit_delay_sec=0.01)

    bvc_data = loader.fetch_dual_resolution("ATW.BVC", daily_years=5, intraday_days=3, intraday_interval="15m")
    df_daily = bvc_data["daily"]
    df_intraday = bvc_data["intraday"]
    assert df_daily is not None
    assert df_intraday is not None

    assert bvc_data["daily"]["currency"].iloc[0] == "MAD"
    assert bvc_data["daily"]["market"].iloc[0] == "BVC"
    assert bvc_data["intraday"]["currency"].iloc[0] == "MAD"


def test_historical_loader_fetch_universe(temp_cache_dir):
    """Test fetching small multi-market universe."""
    store = ParquetDataStore(base_dir=temp_cache_dir)
    loader = HistoricalDataLoader(store=store, rate_limit_delay_sec=0.01)

    test_universe = ["AAPL", "ATW.BVC", "SPY"]
    universe_result = loader.fetch_universe(
        symbols=test_universe,
        daily_years=2,
        intraday_days=2,
        intraday_interval="30m",
        use_cache=True
    )

    assert len(universe_result) == 3
    for sym in test_universe:
        assert sym in universe_result
        assert "daily" in universe_result[sym]
        assert "intraday" in universe_result[sym]
        assert not universe_result[sym]["daily"].empty
        assert not universe_result[sym]["intraday"].empty
