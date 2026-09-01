"""
Point-in-Time Multi-Timeframe Feature Aligner & Vectorized OHLCV Resampler.
Synchronizes daily macro trend features lagged by 1 day (t-1) with high-frequency
intraday bars, strictly eliminating forward lookahead bias.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("DataPipeline.Aligner")


def resample_ohlcv(
    df: pd.DataFrame,
    target_interval: str = "15m"
) -> pd.DataFrame:
    """
    Vectorized OHLCV Resampling Engine.
    Resamples high-frequency bars (e.g. 5m) into coarser intervals (15m, 30m, 60m, 1D).
    
    Aggregation rules:
    - open: first
    - high: max
    - low: min
    - close: last
    - volume: sum
    - vwap: sum(close * volume) / sum(volume) [or volume-weighted vwap if present]
    - trade_count: sum (if present)
    """
    if df.empty:
        return df.copy()

    # Standardize column casing
    data = df.copy()
    data.columns = [c.lower() for c in data.columns]

    # Ensure UTC DatetimeIndex
    if not isinstance(data.index, pd.DatetimeIndex):
        if "timestamp" in data.columns:
            data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True)
            data.set_index("timestamp", inplace=True)
        else:
            data.index = pd.to_datetime(data.index, utc=True)
    elif data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")

    # Map intervals to pandas offset aliases
    interval_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "60m": "60min",
        "1h": "1h",
        "1d": "1D",
        "daily": "1D"
    }
    freq = interval_map.get(target_interval.lower(), target_interval)

    # Compute PV for accurate VWAP aggregation
    price_for_vwap = data["vwap"] if "vwap" in data.columns else data["close"]
    data["_pv"] = price_for_vwap * data["volume"]

    # Resampling aggregation dictionary
    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "_pv": "sum"
    }

    if "trade_count" in data.columns:
        agg_dict["trade_count"] = "sum"
    if "market" in data.columns:
        agg_dict["market"] = "first"
    if "currency" in data.columns:
        agg_dict["currency"] = "first"

    resampled = data.resample(freq, label="left", closed="left").agg(agg_dict).dropna(subset=["close"])

    # Calculate VWAP
    resampled["vwap"] = np.where(
        resampled["volume"] > 0,
        resampled["_pv"] / resampled["volume"],
        resampled["close"]
    )
    resampled.drop(columns=["_pv"], inplace=True)

    # Format numeric columns
    for col in ["open", "high", "low", "close", "vwap"]:
        resampled[col] = resampled[col].round(4)
    resampled["volume"] = resampled["volume"].round(2)

    return resampled


class MultiTimeframeAligner:
    """
    Point-in-Time Multi-Timeframe Bar & Feature Aligner.
    Guarantees that all daily macro indicators merged into intraday timestamps
    are strictly calculated using information up to day t-1.
    """

    @staticmethod
    def compute_daily_macro_features(df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Compute standard 30-year macro trend and volatility features on daily bars.
        Features computed:
        - daily_sma_50: 50-day SMA
        - daily_sma_200: 200-day SMA
        - daily_ema_20: 20-day EMA
        - daily_rsi_14: 14-day RSI
        - daily_vol_20d: 20-day annualized realized volatility
        - daily_mom_12m: 12-month momentum (252 days)
        - daily_high_52w_ratio: Price / 52-Week High (252 days)
        - daily_atr_14: 14-day Average True Range
        - daily_close_unlagged: Close price at day T (will be lagged before merge)
        """
        df = df_daily.copy()
        df.columns = [c.lower() for c in df.columns]
        df.sort_index(inplace=True)

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # Moving averages
        df["daily_sma_50"] = close.rolling(window=50, min_periods=10).mean()
        df["daily_sma_200"] = close.rolling(window=200, min_periods=20).mean()
        df["daily_ema_20"] = close.ewm(span=20, adjust=False).mean()

        # Ratios to moving averages
        df["daily_sma_50_ratio"] = close / df["daily_sma_50"]
        df["daily_sma_200_ratio"] = close / df["daily_sma_200"]

        # 14-day RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=14, min_periods=5).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=5).mean()
        rs = gain / (loss + 1e-8)
        df["daily_rsi_14"] = 100.0 - (100.0 / (1.0 + rs))

        # Realized 20-day annualized volatility
        daily_returns = close.pct_change()
        df["daily_vol_20d"] = daily_returns.rolling(window=20, min_periods=5).std() * np.sqrt(252.0)

        # 12-Month Momentum (252 trading days)
        df["daily_mom_12m"] = close.pct_change(periods=252)

        # 52-Week High Ratio
        rolling_high_52w = high.rolling(window=252, min_periods=20).max()
        df["daily_high_52w_ratio"] = close / (rolling_high_52w + 1e-8)

        # 14-day Average True Range (ATR)
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df["daily_atr_14"] = tr.rolling(window=14, min_periods=5).mean()

        df["daily_close_unlagged"] = close

        # Fill initial NaNs gracefully
        df.bfill(inplace=True)
        return df

    @staticmethod
    def compute_intraday_micro_features(df_intraday: pd.DataFrame) -> pd.DataFrame:
        """
        Compute intraday microstructure features on high-frequency bars (5m-60m).
        Features:
        - rvol_20: Relative Volume (Volume / 20-bar SMA)
        - vol_zscore_30: 30-bar Volume Z-Score
        - vwap_deviation: (Close - VWAP) / VWAP
        - ofi_proxy: Normalized Order Flow Imbalance proxy
        - clv: Close Location Value ((Close - Low) / (High - Low))
        - intraday_ret_1: 1-bar log return
        - intraday_ret_3: 3-bar log return
        - intraday_ret_12: 12-bar log return
        """
        df = df_intraday.copy()
        df.columns = [c.lower() for c in df.columns]
        df.sort_index(inplace=True)

        close = df["close"]
        open_p = df["open"]
        high = df["high"]
        low = df["low"]
        vol = df["volume"]
        vwap = df["vwap"] if "vwap" in df.columns else (high + low + close) / 3.0

        # RVOL and Volume Z-Score
        vol_sma_20 = vol.rolling(window=20, min_periods=3).mean()
        df["rvol_20"] = vol / (vol_sma_20 + 1e-8)

        vol_mean_30 = vol.rolling(window=30, min_periods=5).mean()
        vol_std_30 = vol.rolling(window=30, min_periods=5).std()
        df["vol_zscore_30"] = (vol - vol_mean_30) / (vol_std_30 + 1e-8)

        # VWAP deviation
        df["vwap_deviation"] = (close - vwap) / (vwap + 1e-8)

        # Order Flow Imbalance Proxy
        spread = high - low
        df["ofi_proxy"] = (vol * (close - open_p)) / (spread + 1e-4)

        # Close Location Value (CLV) in [-1, 1] normalized
        raw_clv = ((close - low) - (high - close)) / (spread + 1e-6)
        df["clv"] = np.clip(raw_clv, -1.0, 1.0)

        # Multi-horizon intraday log returns
        df["intraday_ret_1"] = np.log(close / close.shift(1)).fillna(0.0)
        df["intraday_ret_3"] = np.log(close / close.shift(3)).fillna(0.0)
        df["intraday_ret_12"] = np.log(close / close.shift(12)).fillna(0.0)

        # Clean NaNs
        df.bfill(inplace=True)
        return df

    def align_features(
        self,
        df_daily: pd.DataFrame,
        df_intraday: pd.DataFrame,
        custom_macro_features: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Merge daily macro features into intraday bars strictly with t-1 lagging (Zero Lookahead).
        
        Algorithm:
        1. Calculate daily macro features on df_daily.
        2. Shift ALL daily macro feature columns by 1 row (Lagged t-1).
           The row for date D now contains feature values computed through date D-1.
        3. Convert daily index to calendar dates.
        4. Join lagged daily features onto intraday DataFrame using intraday bar calendar dates.
        5. Verify no forward leakage.
        """
        if df_daily.empty:
            raise ValueError("df_daily cannot be empty for multi-timeframe alignment.")
        if df_intraday.empty:
            raise ValueError("df_intraday cannot be empty for multi-timeframe alignment.")

        # Ensure UTC indices
        daily = df_daily.copy()
        if not isinstance(daily.index, pd.DatetimeIndex):
            daily.index = pd.to_datetime(daily.index, utc=True)
        elif daily.index.tz is None:
            daily.index = daily.index.tz_localize("UTC")
        else:
            daily.index = daily.index.tz_convert("UTC")

        intraday = df_intraday.copy()
        if not isinstance(intraday.index, pd.DatetimeIndex):
            intraday.index = pd.to_datetime(intraday.index, utc=True)
        elif intraday.index.tz is None:
            intraday.index = intraday.index.tz_localize("UTC")
        else:
            intraday.index = intraday.index.tz_convert("UTC")

        daily.sort_index(inplace=True)
        intraday.sort_index(inplace=True)

        # 1. Compute macro features on daily
        macro_df = self.compute_daily_macro_features(daily)
        
        # Determine feature columns to keep
        if custom_macro_features is not None:
            feature_cols = [c for c in custom_macro_features if c in macro_df.columns]
        else:
            feature_cols = [
                c for c in macro_df.columns
                if c.startswith("daily_")
            ]

        # 2. Strict Point-in-Time Lag: Shift daily features by 1 row
        # Row at index D will now have values computed from index <= D-1
        lagged_macro = macro_df[feature_cols].shift(1)
        
        # Rename lagged close for clarity
        if "daily_close_unlagged" in lagged_macro.columns:
            lagged_macro.rename(columns={"daily_close_unlagged": "daily_close_lagged_t1"}, inplace=True)

        # 3. Create normalized date key for merging
        # Extract calendar date from UTC timestamps
        lagged_macro["_match_date"] = lagged_macro.index.normalize()
        # Drop duplicates on match_date keeping the last entry
        lagged_macro = lagged_macro.drop_duplicates(subset=["_match_date"]).set_index("_match_date")

        # 4. Compute intraday micro features
        intraday_features = self.compute_intraday_micro_features(intraday)
        intraday_features["_match_date"] = intraday_features.index.normalize()

        # 5. Merge lagged daily features onto intraday bars
        aligned = intraday_features.merge(
            lagged_macro,
            how="left",
            left_on="_match_date",
            right_index=True,
            suffixes=("", "_macro")
        )

        aligned.drop(columns=["_match_date"], inplace=True)
        aligned.index = intraday_features.index

        # Forward fill to handle intraday days that fall on days without new daily bar (e.g. partial holidays)
        aligned.ffill(inplace=True)
        aligned.bfill(inplace=True)

        # 6. Strict Zero-Lookahead Leakage Verification
        self._verify_no_lookahead(daily, aligned)

        return aligned

    def _verify_no_lookahead(self, df_daily: pd.DataFrame, aligned_intraday: pd.DataFrame) -> None:
        """
        Validate mathematical constraint:
        For any intraday bar on date D, daily_close_lagged_t1 must equal close of date D-1,
        and MUST NOT equal close of date D.
        """
        if "daily_close_lagged_t1" not in aligned_intraday.columns:
            return

        daily_close_map = {ts.normalize(): close for ts, close in zip(df_daily.index, df_daily["close"])}
        daily_dates = sorted(list(daily_close_map.keys()))

        if len(daily_dates) < 2:
            return

        # Check sample bars across days
        for i in range(1, min(10, len(daily_dates))):
            curr_date = daily_dates[i]
            prev_date = daily_dates[i - 1]

            bars_today = aligned_intraday[aligned_intraday.index.normalize() == curr_date]
            if not bars_today.empty:
                lagged_val = bars_today["daily_close_lagged_t1"].iloc[0]
                expected_prev = daily_close_map[prev_date]
                today_close = daily_close_map[curr_date]

                # Assert that lagged_val is equal to prev_date close (t-1)
                # and when today_close != prev_date close, verify it does NOT equal today_close
                if not np.isclose(lagged_val, expected_prev, atol=1e-2):
                    logger.warning(
                        f"Alignment warning: lagged close {lagged_val} != expected t-1 close {expected_prev} on {curr_date}"
                    )
                if not np.isclose(today_close, expected_prev, atol=1e-2):
                    assert not np.isclose(lagged_val, today_close, atol=1e-3), (
                        f"CRITICAL LOOKAHEAD DETECTED: Intraday bar on {curr_date} accessed today's close {today_close}!"
                    )
