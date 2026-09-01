"""
Feature Engineering Pipeline for Multi-Horizon Deep Neural Network.

Extracts, normalizes, and sequences:
1. Macro Daily Features (30y history: SMA50/200, RSI14, Vol20d, Mom12m, 52w High Ratio, ATR14, SPY Regime)
2. Microstructure Intraday Features (5m/15m/60m: RVOL, Volume Z-Score, VWAP Distance, OFI, Spread Ratio, CLV, Momentum)
3. Point-in-Time Sentiment Features (7 continuous features from SentimentAggregator)
4. Causal Rolling RobustScaler / Z-Score with zero lookahead bias
5. Multi-Horizon Tensor Dataset Builder for PyTorch models
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.sentiment.aggregator import SentimentAggregator

logger = logging.getLogger("Models.FeaturePipeline")


MACRO_FEATURE_COLS = [
    "sma50_ratio",
    "sma200_ratio",
    "rsi14",
    "vol_20d",
    "mom_12m",
    "ratio_52w_high",
    "atr14_norm",
    "macro_trend_regime",
]

MICRO_FEATURE_COLS = [
    "rvol",
    "vol_zscore",
    "vwap_distance",
    "ofi_norm",
    "spread_ratio",
    "clv",
    "log_ret_1",
    "log_ret_3",
    "log_ret_12",
    "intra_vol_20",
    "vsa_absorption",
    "vsa_spring",
]

SENTIMENT_FEATURE_COLS = [
    "sentiment_score_ema",
    "sentiment_confidence_ema",
    "news_volume_1h",
    "news_volume_24h",
    "sentiment_dispersion",
    "sentiment_momentum",
    "bull_bear_ratio",
]


class CausalRollingScaler:
    """
    Causal Online / Rolling Scaler guaranteeing zero lookahead bias.
    Computes rolling median and IQR (RobustScaler) or rolling mean and std (Z-Score)
    using strictly historical observations up to time t (or t-1).
    """

    def __init__(
        self,
        method: str = "robust",
        window: int = 120,
        min_periods: int = 20,
        eps: float = 1e-6,
    ) -> None:
        self.method = method.lower()
        self.window = window
        self.min_periods = min_periods
        self.eps = eps

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute causal rolling normalization over DataFrame columns.
        Uses expanding window for initial samples and rolling window once window size is reached.
        """
        if df.empty:
            return df.copy()

        scaled_df = pd.DataFrame(index=df.index, columns=df.columns, dtype=np.float64)

        for col in df.columns:
            series = pd.to_numeric(df[col], errors="coerce").astype(np.float64)
            if self.method == "robust":
                # Rolling Median and Interquartile Range (IQR = Q75 - Q25)
                rolling_med = series.rolling(
                    window=self.window, min_periods=self.min_periods
                ).median().fillna(series.expanding(min_periods=1).median())
                
                rolling_q25 = series.rolling(
                    window=self.window, min_periods=self.min_periods
                ).quantile(0.25).fillna(series.expanding(min_periods=1).quantile(0.25))
                
                rolling_q75 = series.rolling(
                    window=self.window, min_periods=self.min_periods
                ).quantile(0.75).fillna(series.expanding(min_periods=1).quantile(0.75))
                
                iqr = np.maximum(rolling_q75 - rolling_q25, self.eps)
                scaled_col = (series - rolling_med) / iqr
            else:
                # Rolling Z-Score
                rolling_mean = series.rolling(
                    window=self.window, min_periods=self.min_periods
                ).mean().fillna(series.expanding(min_periods=1).mean())
                
                rolling_std = series.rolling(
                    window=self.window, min_periods=self.min_periods
                ).std().fillna(series.expanding(min_periods=1).std())
                
                rolling_std = np.maximum(rolling_std, self.eps)
                scaled_col = (series - rolling_mean) / rolling_std

            # Clip extreme outliers to [-5.0, 5.0] to protect gradient stability
            scaled_df[col] = np.clip(scaled_col.fillna(0.0), -5.0, 5.0)

        return scaled_df


class FeatureEngineer:
    """
    Computes domain-specific quantitative alpha features across macro, micro, and sentiment streams.
    """

    @staticmethod
    def _standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to Title case (Open, High, Low, Close, Volume)."""
        df_copy = df.copy()
        col_map = {}
        for c in df_copy.columns:
            c_str = str(c).lower()
            if c_str == "open":
                col_map[c] = "Open"
            elif c_str == "high":
                col_map[c] = "High"
            elif c_str == "low":
                col_map[c] = "Low"
            elif c_str == "close":
                col_map[c] = "Close"
            elif c_str == "volume":
                col_map[c] = "Volume"
            elif c_str == "vwap":
                col_map[c] = "VWAP"
        return df_copy.rename(columns=col_map)

    @staticmethod
    def compute_macro_features(
        daily_df: pd.DataFrame,
        spy_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Extract macro indicators from daily OHLCV bars.
        Features computed:
        - SMA50 ratio: Close / SMA50
        - SMA200 ratio: Close / SMA200
        - RSI14: 14-day Relative Strength Index (scaled to [-1, 1])
        - Vol20d: 20-day annualized volatility
        - Mom12m: 12-month (252-day) momentum ratio
        - Ratio52wHigh: Close / 252-day Rolling High
        - ATR14Norm: 14-day Average True Range normalized by Close
        - MacroTrendRegime: 1.0 if SPY >= SMA200 else 0.0 (or Close >= SMA200 if SPY not given)
        """
        if daily_df.empty:
            return pd.DataFrame(columns=MACRO_FEATURE_COLS)

        df = FeatureEngineer._standardize_ohlcv(daily_df).sort_index()
        close = df["Close"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)

        res = pd.DataFrame(index=df.index)

        # 1. 50-day and 200-day SMA Ratios
        sma50 = close.rolling(window=50, min_periods=5).mean()
        sma200 = close.rolling(window=200, min_periods=10).mean()
        res["sma50_ratio"] = (close / sma50).fillna(1.0) - 1.0
        res["sma200_ratio"] = (close / sma200).fillna(1.0) - 1.0

        # 2. RSI 14 (Scaled to [-1.0, 1.0])
        delta = close.diff()
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        roll_gain = pd.Series(gain, index=df.index).rolling(window=14, min_periods=2).mean()
        roll_loss = pd.Series(loss, index=df.index).rolling(window=14, min_periods=2).mean()
        rs = roll_gain / np.maximum(roll_loss, 1e-6)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        res["rsi14"] = ((rsi - 50.0) / 50.0).fillna(0.0).clip(-1.0, 1.0)

        # 3. 20-day Annualized Volatility
        log_ret = np.log(close / close.shift(1)).fillna(0.0)
        vol_20d = log_ret.rolling(window=20, min_periods=5).std() * np.sqrt(252.0)
        res["vol_20d"] = vol_20d.fillna(0.20)

        # 4. 12-Month Momentum (252 bars or max available)
        mom_12m = (close / close.shift(252).fillna(close.iloc[0])) - 1.0
        res["mom_12m"] = mom_12m.fillna(0.0).clip(-2.0, 2.0)

        # 5. 52-Week High Ratio (Close / Rolling Max High 252 bars)
        roll_high_52w = high.rolling(window=252, min_periods=10).max().fillna(high)
        res["ratio_52w_high"] = (close / np.maximum(roll_high_52w, 1e-6)).fillna(1.0).clip(0.0, 1.5)

        # 6. 14-day ATR Normalized
        prev_close = close.shift(1).fillna(close)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr14 = pd.Series(tr, index=df.index).rolling(window=14, min_periods=2).mean()
        res["atr14_norm"] = (atr14 / np.maximum(close, 1e-6)).fillna(0.02)

        # 7. SPY Macro Trend Regime
        if spy_df is not None and not spy_df.empty:
            spy_close = spy_df["Close"].reindex(df.index, method="ffill").fillna(close)
            spy_sma200 = spy_close.rolling(window=200, min_periods=10).mean()
            res["macro_trend_regime"] = np.where(spy_close >= spy_sma200, 1.0, 0.0)
        else:
            res["macro_trend_regime"] = np.where(close >= sma200, 1.0, 0.0)

        return res[MACRO_FEATURE_COLS].ffill().bfill().fillna(0.0)

    @staticmethod
    def compute_intraday_micro_features(
        intraday_df: pd.DataFrame,
        market_cap: float = 1e10,
    ) -> pd.DataFrame:
        """
        Extract intraday microstructure and VSA alpha features from 5m/15m/60m bars.
        Features computed:
        - RVOL: Relative Volume vs 20-bar rolling volume
        - VolZScore: 30-bar rolling volume Z-score
        - VWAPDistance: Deviation of Close from intraday cumulative VWAP
        - OFINorm: Normalized Order Flow Imbalance proxy
        - SpreadRatio: Bar high-low spread vs 20-bar average spread
        - CLV: Close Location Value (where close sits in bar range [0, 1])
        - LogRet1, LogRet3, LogRet12: Short-term multi-scale momentum
        - IntraVol20: 20-bar rolling intraday volatility
        - VSAAbsorption, VSASpring: Wyckoff volume-spread analysis pattern flags
        """
        if intraday_df.empty:
            return pd.DataFrame(columns=MICRO_FEATURE_COLS)

        df = FeatureEngineer._standardize_ohlcv(intraday_df).sort_index()
        close = df["Close"].astype(float)
        open_p = df["Open"].astype(float)
        high = df["High"].astype(float)
        low = df["Low"].astype(float)
        volume = df["Volume"].astype(float)

        res = pd.DataFrame(index=df.index)

        # 1. Relative Volume (RVOL)
        roll_vol20 = volume.rolling(window=20, min_periods=3).mean()
        res["rvol"] = (volume / np.maximum(roll_vol20, 1.0)).fillna(1.0).clip(0.0, 20.0)

        # 2. Volume Z-Score
        roll_vol_mean = volume.rolling(window=30, min_periods=5).mean()
        roll_vol_std = volume.rolling(window=30, min_periods=5).std()
        res["vol_zscore"] = (
            (volume - roll_vol_mean) / np.maximum(roll_vol_std, 1e-6)
        ).fillna(0.0).clip(-3.0, 5.0)

        # 3. Cumulative Intraday VWAP Distance
        # Group by trading date for intraday session VWAP resetting
        dates = df.index.normalize() if hasattr(df.index, "normalize") else pd.to_datetime(df.index).normalize()
        pv = (close + high + low) / 3.0 * volume
        cum_pv = pv.groupby(dates).cumsum()
        cum_vol = volume.groupby(dates).cumsum()
        vwap = cum_pv / np.maximum(cum_vol, 1e-6)
        res["vwap_distance"] = ((close - vwap) / np.maximum(vwap, 1e-6)).fillna(0.0).clip(-0.20, 0.20)

        # 4. Normalized Order Flow Imbalance (OFI Proxy)
        bar_range = np.maximum(high - low, 1e-6)
        price_delta = close - open_p
        ofi_raw = (volume * price_delta) / (bar_range * (market_cap / 1e8))
        res["ofi_norm"] = ofi_raw.fillna(0.0).clip(-10.0, 10.0)

        # 5. Candle Spread Ratio
        roll_range20 = bar_range.rolling(window=20, min_periods=3).mean()
        res["spread_ratio"] = (bar_range / np.maximum(roll_range20, 1e-6)).fillna(1.0).clip(0.1, 10.0)

        # 6. Close Location Value (CLV in [-1.0, 1.0])
        clv = (2.0 * close - high - low) / np.maximum(high - low, 1e-6)
        res["clv"] = clv.fillna(0.0).clip(-1.0, 1.0)

        # 7. Short-Term Log Returns
        log_ret_1 = np.log(close / close.shift(1)).fillna(0.0)
        log_ret_3 = np.log(close / close.shift(3)).fillna(0.0)
        log_ret_12 = np.log(close / close.shift(12)).fillna(0.0)
        res["log_ret_1"] = log_ret_1.clip(-0.15, 0.15)
        res["log_ret_3"] = log_ret_3.clip(-0.25, 0.25)
        res["log_ret_12"] = log_ret_12.clip(-0.40, 0.40)

        # 8. 20-bar Intraday Volatility
        res["intra_vol_20"] = log_ret_1.rolling(window=20, min_periods=3).std().fillna(0.01)

        # 9. Wyckoff VSA Flags (Absorption and Spring)
        # Absorption: High volume, narrow spread, close in upper half
        is_high_vol = res["rvol"] > 1.8
        is_narrow_spread = res["spread_ratio"] < 0.9
        is_high_close = res["clv"] > 0.3
        res["vsa_absorption"] = np.where(is_high_vol & is_narrow_spread & is_high_close, 1.0, 0.0)

        # Spring: Lower low than previous 5 bars, but close rebounds near high with above-average volume
        roll_low_5 = low.shift(1).rolling(window=5, min_periods=1).min()
        is_spring_low = low < roll_low_5
        res["vsa_spring"] = np.where(is_spring_low & is_high_close & (res["rvol"] > 1.2), 1.0, 0.0)

        return res[MICRO_FEATURE_COLS].ffill().bfill().fillna(0.0)


@dataclass
class DatasetTensors:
    """Packaged PyTorch tensors ready for Multi-Horizon DNN consumption."""
    macro_seq: torch.Tensor       # Shape: [B, macro_seq_len, D_macro]
    micro_seq: torch.Tensor       # Shape: [B, micro_seq_len, D_micro]
    sent_vec: torch.Tensor        # Shape: [B, D_sent]
    target_labels: Optional[torch.Tensor] = None   # Shape: [B] (SELL=0, HOLD=1, BUY=2)
    target_returns: Optional[torch.Tensor] = None  # Shape: [B, num_horizons]
    timestamps: Optional[pd.DatetimeIndex] = None


class MultiHorizonDataset(Dataset):
    """PyTorch Dataset yielding aligned macro, micro, sentiment inputs and labels."""

    def __init__(self, tensors: DatasetTensors):
        self.macro_seq = tensors.macro_seq
        self.micro_seq = tensors.micro_seq
        self.sent_vec = tensors.sent_vec
        self.target_labels = tensors.target_labels
        self.target_returns = tensors.target_returns
        self.length = len(self.macro_seq)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "macro_seq": self.macro_seq[idx],
            "micro_seq": self.micro_seq[idx],
            "sent_vec": self.sent_vec[idx],
        }
        if self.target_labels is not None:
            item["target_labels"] = self.target_labels[idx]
        if self.target_returns is not None:
            item["target_returns"] = self.target_returns[idx]
        return item


class FeaturePipeline:
    """
    End-to-end Feature Pipeline orchestrating:
    - Multi-timeframe feature extraction
    - Causal rolling scaling
    - NLP sentiment alignment
    - Sequence sliding-window construction for PyTorch tensors
    """

    def __init__(
        self,
        macro_seq_len: int = 60,
        micro_seq_len: int = 39,
        scaler_window: int = 120,
        scaler_method: str = "robust",
    ) -> None:
        self.macro_seq_len = macro_seq_len
        self.micro_seq_len = micro_seq_len
        self.scaler = CausalRollingScaler(method=scaler_method, window=scaler_window)
        self.sentiment_aggregator = SentimentAggregator()

    def process_features(
        self,
        daily_df: pd.DataFrame,
        intraday_df: pd.DataFrame,
        sentiment_df: Optional[pd.DataFrame] = None,
        spy_daily_df: Optional[pd.DataFrame] = None,
        market_cap: float = 1e10,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extract raw features and apply causal scaling with zero lookahead bias.
        Returns (scaled_macro_df, scaled_micro_df, scaled_sent_df).
        """
        # 1. Macro Features
        macro_raw = FeatureEngineer.compute_macro_features(daily_df, spy_daily_df)
        macro_scaled = self.scaler.fit_transform(macro_raw)

        # 2. Micro Features
        micro_raw = FeatureEngineer.compute_intraday_micro_features(intraday_df, market_cap=market_cap)
        micro_scaled = self.scaler.fit_transform(micro_raw)

        # 3. Sentiment Features
        if sentiment_df is not None and not sentiment_df.empty:
            # Reindex to intraday bar timestamps using forward fill
            sent_aligned = sentiment_df[SENTIMENT_FEATURE_COLS].reindex(
                intraday_df.index, method="ffill"
            ).fillna(0.0)
        else:
            # Default neutral sentiment DataFrame
            sent_aligned = pd.DataFrame(
                0.0, index=intraday_df.index, columns=SENTIMENT_FEATURE_COLS
            )
            sent_aligned["bull_bear_ratio"] = 1.0

        sent_scaled = self.scaler.fit_transform(sent_aligned)

        return macro_scaled, micro_scaled, sent_scaled

    def build_dataset_tensors(
        self,
        macro_scaled_df: pd.DataFrame,
        micro_scaled_df: pd.DataFrame,
        sent_scaled_df: pd.DataFrame,
        labels_df: Optional[pd.DataFrame] = None,
    ) -> DatasetTensors:
        """
        Assemble sequence windows into PyTorch tensors.
        For each intraday bar index i >= micro_seq_len - 1:
        - micro_seq: slice [i - micro_seq_len + 1 : i + 1]
        - macro_seq: macro daily bars available up to date of intraday bar i (lagged by 1 day to eliminate lookahead)
        - sent_vec: sentiment vector at bar i
        - labels: corresponding target labels & returns if provided
        """
        n_bars = len(micro_scaled_df)
        if n_bars < self.micro_seq_len:
            raise ValueError(
                f"Intraday bars ({n_bars}) insufficient for micro_seq_len ({self.micro_seq_len})"
            )

        micro_values = micro_scaled_df.values.astype(np.float32)
        sent_values = sent_scaled_df.values.astype(np.float32)
        macro_values = macro_scaled_df.values.astype(np.float32)
        macro_dates = macro_scaled_df.index.normalize() if hasattr(macro_scaled_df.index, "normalize") else pd.to_datetime(macro_scaled_df.index).normalize()
        micro_dates = micro_scaled_df.index.normalize() if hasattr(micro_scaled_df.index, "normalize") else pd.to_datetime(micro_scaled_df.index).normalize()

        valid_indices = []
        macro_seqs = []
        micro_seqs = []
        sent_vecs = []
        lbl_list = []
        ret_list = []

        # Map each date to available macro index
        macro_date_to_idx = {date: idx for idx, date in enumerate(macro_dates)}

        start_idx = self.micro_seq_len - 1
        for i in range(start_idx, n_bars):
            cur_date = micro_dates[i]
            
            # Find the most recent macro daily bar prior to or on cur_date (lagged point-in-time)
            # Use searchsorted to find macro index strictly strictly on or before cur_date
            macro_idx = np.searchsorted(macro_dates, cur_date, side="right") - 1
            if macro_idx < 0:
                macro_idx = 0

            # Construct macro sequence of length macro_seq_len
            macro_start = max(0, macro_idx - self.macro_seq_len + 1)
            macro_slice = macro_values[macro_start : macro_idx + 1]
            if len(macro_slice) < self.macro_seq_len:
                # Pad earlier entries by repeating first row
                pad_len = self.macro_seq_len - len(macro_slice)
                pad = np.repeat(macro_slice[:1], pad_len, axis=0) if len(macro_slice) > 0 else np.zeros((pad_len, macro_values.shape[1]), dtype=np.float32)
                macro_slice = np.vstack([pad, macro_slice])

            # Construct micro sequence of length micro_seq_len
            micro_slice = micro_values[i - self.micro_seq_len + 1 : i + 1]
            sent_vec = sent_values[i]

            macro_seqs.append(macro_slice)
            micro_seqs.append(micro_slice)
            sent_vecs.append(sent_vec)
            valid_indices.append(micro_scaled_df.index[i])

            if labels_df is not None:
                ts = micro_scaled_df.index[i]
                if ts in labels_df.index:
                    row = labels_df.loc[ts]
                    # Label: BUY=2, HOLD=1, SELL=0 (or 1, 0, -1 converted to 0, 1, 2)
                    raw_lbl = row.get("label_triple_barrier", 0)
                    if raw_lbl == -1:
                        class_idx = 0
                    elif raw_lbl == 0:
                        class_idx = 1
                    elif raw_lbl == 1:
                        class_idx = 2
                    else:
                        class_idx = int(raw_lbl)

                    lbl_list.append(class_idx)

                    # Multi-horizon returns [h3, h12, h78]
                    r3 = float(row.get("ret_h3", 0.0))
                    r12 = float(row.get("ret_h12", 0.0))
                    r78 = float(row.get("ret_h78", 0.0))
                    ret_list.append([r3, r12, r78])
                else:
                    lbl_list.append(1)  # Default HOLD
                    ret_list.append([0.0, 0.0, 0.0])

        macro_tensor = torch.tensor(np.array(macro_seqs, dtype=np.float32), dtype=torch.float32)
        micro_tensor = torch.tensor(np.array(micro_seqs, dtype=np.float32), dtype=torch.float32)
        sent_tensor = torch.tensor(np.array(sent_vecs, dtype=np.float32), dtype=torch.float32)

        lbl_tensor = (
            torch.tensor(np.array(lbl_list, dtype=np.int64), dtype=torch.long)
            if labels_df is not None
            else None
        )
        ret_tensor = (
            torch.tensor(np.array(ret_list, dtype=np.float32), dtype=torch.float32)
            if labels_df is not None
            else None
        )

        return DatasetTensors(
            macro_seq=macro_tensor,
            micro_seq=micro_tensor,
            sent_vec=sent_tensor,
            target_labels=lbl_tensor,
            target_returns=ret_tensor,
            timestamps=pd.DatetimeIndex(valid_indices),
        )
