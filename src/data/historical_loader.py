"""
Historical Data Loader for Dual-Universe & Dual-Resolution Equities.
Ingests 30 years of macro daily bars and high-frequency intraday bars (5m-60m)
for US (NYSE/NASDAQ) and Moroccan (Casablanca BVC) equities using yfinance / APIs
with exponential backoff, rate limiting, and seamless synthetic fallback.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from .aligner import resample_ohlcv
from .loader import AssetMarket, UniverseConfig
from .parquet_store import ParquetDataStore
from .synthetic_feed import SyntheticMicrostructureGenerator

try:
    import yfinance as yf
except ImportError:
    yf = None

logger = logging.getLogger("DataPipeline.HistoricalLoader")

T = TypeVar("T")


class HistoricalDataLoader:
    """
    Dual-Resolution Data Ingestion Engine.
    Handles 30-year secular daily data and high-frequency intraday data across
    US and Casablanca stock markets with Parquet caching and automatic fallback.
    """

    def __init__(
        self,
        store: Optional[ParquetDataStore] = None,
        config: Optional[UniverseConfig] = None,
        rate_limit_delay_sec: float = 0.25,
        max_retries: int = 3
    ):
        self.config = config or UniverseConfig()
        self.store = store or ParquetDataStore(base_dir=self.config.cache_dir)
        self.generator = SyntheticMicrostructureGenerator()
        self.rate_limit_delay_sec = rate_limit_delay_sec
        self.max_retries = max_retries

    def _determine_market_and_currency(self, symbol: str) -> Tuple[str, str]:
        """Determine asset market type and operating currency from symbol."""
        sym_clean = symbol.upper()
        if sym_clean.endswith(".BVC") or sym_clean.endswith("_BVC") or sym_clean in [s.upper() for s in self.config.bvc_symbols]:
            return "BVC", "MAD"
        return "US", "USD"

    def _execute_with_retry(
        self,
        func: Callable[[], T],
        operation_name: str,
        max_retries: Optional[int] = None
    ) -> Optional[T]:
        """Execute a remote network call with exponential backoff retries."""
        retries = max_retries or self.max_retries
        backoff = 1.0

        for attempt in range(1, retries + 1):
            try:
                time.sleep(self.rate_limit_delay_sec)
                result = func()
                return result
            except Exception as e:
                logger.warning(
                    f"Attempt {attempt}/{retries} failed for {operation_name}: {e}."
                )
                if attempt == retries:
                    logger.error(f"All {retries} retries exhausted for {operation_name}.")
                    return None
                time.sleep(backoff)
                backoff *= 2.0
        return None

    def fetch_daily(
        self,
        symbol: str,
        years: int = 30,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch up to 30 years of daily historical OHLCV data.
        1. Checks Parquet store.
        2. Tries public API / yfinance with retries.
        3. If unavailable, falls back to high-fidelity jump-diffusion synthetic feed.
        4. Saves to Parquet store.
        """
        market, currency = self._determine_market_and_currency(symbol)
        timeframe = "daily"

        # Check Cache
        if use_cache and self.store.has_bars(symbol, timeframe):
            cached_df = self.store.load_bars(symbol, timeframe, start=start_date, end=end_date)
            if cached_df is not None and not cached_df.empty:
                logger.debug(f"Loaded cached daily data for {symbol} ({len(cached_df)} bars).")
                return cached_df

        # Resolve date boundaries
        now = datetime.now(timezone.utc)
        if end_date is None:
            end_dt = now
            end_str = end_dt.strftime("%Y-%m-%d")
        else:
            end_str = end_date
            end_dt = pd.to_datetime(end_date, utc=True)

        if start_date is None:
            start_year = max(1995, end_dt.year - years)
            start_str = f"{start_year}-01-01"
        else:
            start_str = start_date

        df_daily: Optional[pd.DataFrame] = None

        # Try yfinance for US equities
        if market == "US" and yf is not None:
            def _download_yf() -> Optional[pd.DataFrame]:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(
                    start=start_str,
                    end=end_str,
                    auto_adjust=True,
                    timeout=10
                )
                if not hist.empty and len(hist) >= 20:
                    hist.columns = [c.lower() for c in hist.columns]
                    hist = hist[["open", "high", "low", "close", "volume"]].copy()
                    hist.index = pd.to_datetime(hist.index, utc=True)
                    hist["vwap"] = (hist["high"] + hist["low"] + hist["close"]) / 3.0
                    hist["market"] = "US"
                    hist["currency"] = "USD"
                    return hist
                return None

            df_daily = self._execute_with_retry(_download_yf, f"yfinance daily ({symbol})")

        # Fallback to high-fidelity synthetic generator if no data
        if df_daily is None or df_daily.empty:
            logger.info(f"Generating high-fidelity synthetic daily feed for {symbol} ({start_str} to {end_str}).")
            df_daily = self.generator.generate_daily_series(
                symbol=symbol,
                start_date=start_str,
                end_date=end_str,
                market=market
            )

        # Ensure schema standard
        df_daily.columns = [c.lower() for c in df_daily.columns]
        df_daily.sort_index(inplace=True)

        # Store in Parquet cache
        try:
            self.store.store_bars(
                symbol=symbol,
                timeframe=timeframe,
                df=df_daily,
                market=market,
                currency=currency
            )
        except Exception as e:
            logger.warning(f"Could not cache daily bars for {symbol}: {e}")

        # Slice to requested date window if needed
        if start_date is not None:
            df_daily = df_daily[df_daily.index >= pd.to_datetime(start_date, utc=True)]
        if end_date is not None:
            df_daily = df_daily[df_daily.index <= pd.to_datetime(end_date, utc=True)]

        return df_daily

    def fetch_intraday(
        self,
        symbol: str,
        days: int = 60,
        interval: str = "5m",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        Fetch high-resolution intraday OHLCV bars (5m, 15m, 30m, 60m).
        1. Checks Parquet store.
        2. Tries public API / yfinance (up to 60 days for 5m, 730 days for 60m).
        3. If unavailable, generates realistic microstructure intraday series with U-shaped volume.
        4. Saves to Parquet store.
        """
        market, currency = self._determine_market_and_currency(symbol)
        timeframe = interval.lower()

        # Check Cache
        if use_cache and self.store.has_bars(symbol, timeframe):
            cached_df = self.store.load_bars(symbol, timeframe, start=start_date, end=end_date)
            if cached_df is not None and not cached_df.empty:
                logger.debug(f"Loaded cached intraday {timeframe} data for {symbol} ({len(cached_df)} bars).")
                return cached_df

        df_intraday: Optional[pd.DataFrame] = None

        # Try yfinance for US equities
        if market == "US" and yf is not None:
            def _download_yf_intraday() -> Optional[pd.DataFrame]:
                ticker = yf.Ticker(symbol)
                # Determine period or date limits based on interval
                if timeframe in ["1m", "2m", "5m"]:
                    fetch_period = f"{min(days, 59)}d"
                else:
                    fetch_period = f"{min(days, 720)}d"

                hist = ticker.history(
                    period=fetch_period,
                    interval=timeframe if timeframe != "60m" else "1h",
                    auto_adjust=True,
                    timeout=10
                )
                if not hist.empty and len(hist) >= 10:
                    hist.columns = [c.lower() for c in hist.columns]
                    hist = hist[["open", "high", "low", "close", "volume"]].copy()
                    hist.index = pd.to_datetime(hist.index, utc=True)
                    # Compute VWAP
                    hist["vwap"] = (hist["high"] + hist["low"] + hist["close"]) / 3.0
                    hist["market"] = "US"
                    hist["currency"] = "USD"
                    return hist
                return None

            df_intraday = self._execute_with_retry(_download_yf_intraday, f"yfinance intraday ({symbol} {timeframe})")

        # Fallback to high-fidelity synthetic microstructure generator
        if df_intraday is None or df_intraday.empty:
            logger.info(
                f"Generating high-fidelity synthetic intraday feed for {symbol} "
                f"({timeframe}, {days} days)."
            )
            df_intraday = self.generator.generate_intraday_series(
                symbol=symbol,
                days=days,
                interval=timeframe,
                market=market,
                start_date=start_date,
                end_date=end_date
            )

        # Ensure schema standard
        df_intraday.columns = [c.lower() for c in df_intraday.columns]
        df_intraday.sort_index(inplace=True)

        # Store in Parquet cache
        try:
            self.store.store_bars(
                symbol=symbol,
                timeframe=timeframe,
                df=df_intraday,
                market=market,
                currency=currency
            )
        except Exception as e:
            logger.warning(f"Could not cache intraday bars for {symbol}: {e}")

        # Slice to requested dates if needed
        if start_date is not None:
            df_intraday = df_intraday[df_intraday.index >= pd.to_datetime(start_date, utc=True)]
        if end_date is not None:
            df_intraday = df_intraday[df_intraday.index <= pd.to_datetime(end_date, utc=True)]

        return df_intraday

    def fetch_dual_resolution(
        self,
        symbol: str,
        daily_years: int = 30,
        intraday_days: int = 60,
        intraday_interval: str = "5m",
        use_cache: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Master Dual-Resolution Fetcher Interface Contract.
        Returns:
            {
                "daily": df_daily (30 years daily bars),
                "intraday": df_intraday (high-resolution intraday bars)
            }
        Both indexed by UTC DatetimeIndex with columns:
        ['open', 'high', 'low', 'close', 'volume', 'vwap'].
        """
        df_daily = self.fetch_daily(symbol, years=daily_years, use_cache=use_cache)
        df_intraday = self.fetch_intraday(
            symbol,
            days=intraday_days,
            interval=intraday_interval,
            use_cache=use_cache
        )

        return {
            "daily": df_daily,
            "intraday": df_intraday
        }

    def fetch_universe(
        self,
        symbols: Optional[List[str]] = None,
        daily_years: int = 30,
        intraday_days: int = 60,
        intraday_interval: str = "5m",
        use_cache: bool = True
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Fetch dual-resolution market data for the complete investable universe.
        """
        target_symbols = symbols or (
            self.config.us_symbols + [self.config.benchmark_symbol] + self.config.bvc_symbols
        )

        universe_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        for sym in target_symbols:
            logger.info(f"Ingesting dual-resolution data for universe asset: {sym}")
            dual_data = self.fetch_dual_resolution(
                symbol=sym,
                daily_years=daily_years,
                intraday_days=intraday_days,
                intraday_interval=intraday_interval,
                use_cache=use_cache
            )
            universe_data[sym] = dual_data

        return universe_data
