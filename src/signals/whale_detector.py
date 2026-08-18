"""
Whale-Tracking Alpha Signal Engine.
Implements Volume Z-Score, Wyckoff VSA Absorption, Order Flow Imbalance (OFI),
Isolation Forest Machine Learning Anomaly Detection, and 52-Week High Momentum Crash Protection.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.signals.insider_scraper import InsiderIntelligenceEngine

logger = logging.getLogger("WhaleTracking.Signals")


@dataclass
class WhaleSignalResult:
    signals: pd.DataFrame
    raw_scores: pd.Series
    conviction_multipliers: pd.Series
    volatility: pd.Series


class WhaleDetector:
    """Multi-dimensional Institutional Smart-Money & Insider Anomaly Detector."""

    def __init__(
        self,
        volume_window: int = 30,
        z_score_threshold: float = 2.5,
        rvol_threshold: float = 2.5,
        min_momentum: float = 0.10,
        contamination: float = 0.08,
        use_isolation_forest: bool = True,
        use_insider_data: bool = True
    ):
        self.volume_window = volume_window
        self.z_score_threshold = z_score_threshold
        self.rvol_threshold = rvol_threshold
        self.min_momentum = min_momentum
        self.contamination = contamination
        self.use_isolation_forest = use_isolation_forest
        self.use_insider_data = use_insider_data
        self.insider_engine = InsiderIntelligenceEngine() if use_insider_data else None

    def process_asset(
        self,
        df: pd.DataFrame,
        market: str = "US",
        symbol: Optional[str] = None
    ) -> pd.DataFrame:
        """Calculate full feature matrix, whale signals, and insider convergence for an asset."""
        data = df.copy()

        # 1. Basic Returns & Historical Volatility
        data["Returns"] = data["Close"].pct_change().fillna(0.0)
        data["Vol_20D"] = data["Returns"].rolling(window=20, min_periods=10).std() * np.sqrt(252.0)
        data["Vol_20D"] = data["Vol_20D"].bfill().fillna(0.20).clip(lower=0.05)

        # ATR (Average True Range) for trailing volatility stops
        tr1 = data["High"] - data["Low"]
        tr2 = (data["High"] - data["Close"].shift(1)).abs()
        tr3 = (data["Low"] - data["Close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        data["ATR_14"] = tr.rolling(14, min_periods=5).mean().bfill()

        # Trend & Regime Filters (EMA 20 / SMA 50 / SMA 200)
        data["EMA_20"] = data["Close"].ewm(span=20, adjust=False).mean()
        data["SMA_20"] = data["Close"].rolling(20, min_periods=10).mean()
        data["SMA_50"] = data["Close"].rolling(50, min_periods=20).mean()
        data["SMA_200"] = data["Close"].rolling(200, min_periods=50).mean()
        data["Trend_Bullish"] = (data["Close"] > data["EMA_20"]) & (data["EMA_20"] >= data["SMA_50"].bfill() * 0.98)

        # 2. Volume Analytics & Z-Score
        roll_vol = data["Volume"].rolling(window=self.volume_window, min_periods=15)
        data["Vol_SMA"] = roll_vol.mean()
        data["Vol_Std"] = roll_vol.std().replace(0, np.nan).bfill().fillna(1.0)
        data["RVOL"] = data["Volume"] / data["Vol_SMA"].replace(0, np.nan).bfill()
        data["Vol_ZScore"] = (data["Volume"] - data["Vol_SMA"]) / data["Vol_Std"]
        data["Vol_ZScore"] = data["Vol_ZScore"].fillna(0.0)

        # Dollar Volume Filtering ($1M USD / 100K MAD)
        dollar_vol = data["Close"] * data["Volume"]
        if market == "US":
            data["Passes_Dollar_Volume"] = dollar_vol >= 1_000_000.0
        else:
            data["Passes_Dollar_Volume"] = dollar_vol >= 100_000.0

        # 3. 12-Month Momentum & 52-Week High Anchoring Neutralization
        lookback_mom = min(250, max(20, len(data) - 1))
        data["Close_252D_Ago"] = data["Close"].shift(lookback_mom).bfill().fillna(data["Close"].iloc[0])
        data["Mom_12M"] = (data["Close"] / data["Close_252D_Ago"].replace(0, np.nan)).fillna(1.0) - 1.0
        data["High_52W"] = data["High"].rolling(window=min(252, len(data)), min_periods=10).max()
        data["Ratio_52W"] = (data["Close"] / data["High_52W"]).clip(0.0, 1.0).fillna(1.0)

        # Neutralized Momentum Score (Daniel & Moskowitz momentum crash protection)
        data["Mom_Score"] = np.where(
            data["Mom_12M"] >= self.min_momentum,
            data["Mom_12M"] * (data["Ratio_52W"] ** 2),
            0.0
        )

        # 4. Wyckoff Volume Spread Analysis (VSA)
        data["Candle_Spread"] = (data["High"] - data["Low"]) / data["Close"]
        roll_spread = data["Candle_Spread"].rolling(window=20, min_periods=10).mean().bfill()
        data["Spread_Ratio"] = data["Candle_Spread"] / roll_spread.replace(0, np.nan).fillna(1.0)

        candle_range = (data["High"] - data["Low"]).replace(0, np.nan)
        data["Close_Location"] = ((data["Close"] - data["Low"]) / candle_range).fillna(0.5)

        # Wyckoff Absorption: High Volume but compressed spread with strong close -> Smart Money Accumulation
        data["Wyckoff_Absorption"] = (
            (data["Vol_ZScore"] >= 1.3) &
            (data["Close_Location"] >= 0.50) &
            (data["Spread_Ratio"] <= 1.4)
        ).astype(int)

        # Wyckoff Spring: Low test followed by strong recovery on volume
        low_20d = data["Low"].rolling(window=20, min_periods=10).min().shift(1)
        data["Wyckoff_Spring"] = (
            (data["Low"] < low_20d) &
            (data["Close_Location"] >= 0.60) &
            (data["RVOL"] >= 1.3)
        ).astype(int)

        # Breakout on Institutional Volume
        high_20d = data["High"].rolling(window=20, min_periods=10).max().shift(1)
        data["Volume_Breakout"] = (
            (data["Close"] >= high_20d * 0.99) &
            (data["Vol_ZScore"] >= 1.3)
        ).astype(int)

        # Wyckoff Upthrust (Bearish rejection): Disqualifier
        data["Wyckoff_Upthrust"] = (
            (data["High"] > high_20d) &
            (data["Close_Location"] <= 0.35) &
            (data["RVOL"] >= 1.8)
        ).astype(int)

        # 5. Order Flow Imbalance (OFI) normalized by Market Cap (Matched Filter)
        mcap = data.get("MarketCap", pd.Series(50e9, index=data.index))
        ofi_raw = data["Volume"] * (data["Close"] - data["Open"]) / (data["High"] - data["Low"] + 1e-5)
        data["OFI_Norm"] = (ofi_raw * data["Close"]) / mcap
        data["OFI_Norm_Z"] = (
            (data["OFI_Norm"] - data["OFI_Norm"].rolling(30, min_periods=10).mean())
            / data["OFI_Norm"].rolling(30, min_periods=10).std().replace(0, 1.0)
        ).fillna(0.0)

        # 6. Unsupervised Machine Learning Anomaly Score (Isolation Forest)
        if self.use_isolation_forest and len(data) >= 60:
            features = data[["Vol_ZScore", "RVOL", "Spread_Ratio", "Close_Location", "OFI_Norm_Z"]].fillna(0.0)
            try:
                clf = IsolationForest(
                    contamination=self.contamination,
                    random_state=42,
                    n_estimators=100
                )
                clf.fit(features)
                data["ML_Anomaly_Score"] = -clf.decision_function(features)
                data["ML_Is_Whale"] = (clf.predict(features) == -1).astype(int)
            except Exception as e:
                logger.warning(f"Isolation Forest error: {e}. Defaulting to rule-based.")
                data["ML_Anomaly_Score"] = 0.0
                data["ML_Is_Whale"] = 0
        else:
            data["ML_Anomaly_Score"] = 0.0
            data["ML_Is_Whale"] = 0

        # 7. Composite Whale Accumulation Entry Signal
        has_surge = (data["Vol_ZScore"] >= self.z_score_threshold) | (data["RVOL"] >= self.rvol_threshold) | (data["Volume_Breakout"] == 1)
        is_accumulating = (data["Wyckoff_Absorption"] == 1) | (data["Wyckoff_Spring"] == 1) | (data["ML_Is_Whale"] == 1)
        valid_direction = (data["Close_Location"] >= 0.45) & (data["Wyckoff_Upthrust"] == 0)

        entry_signal = (
            data["Passes_Dollar_Volume"] &
            (data["Mom_Score"] > 0) &
            data["Trend_Bullish"] &
            (has_surge | is_accumulating) &
            valid_direction
        ).astype(int)

        # Asymmetric Trend Tracking (Ride winners above EMA20, cut losses at -5%)
        in_position = np.zeros(len(data), dtype=int)
        pos = 0
        entry_p = 0.0
        peak_p = 0.0

        close_vals = data["Close"].values
        ema20_vals = data["EMA_20"].values
        entry_sig_vals = entry_signal.values
        upthrust_vals = data["Wyckoff_Upthrust"].values

        for i in range(len(data)):
            if entry_sig_vals[i] == 1 and pos == 0:
                pos = 1
                entry_p = close_vals[i]
                peak_p = close_vals[i]
            elif pos == 1:
                if close_vals[i] > peak_p:
                    peak_p = close_vals[i]

                # Asymmetric Payoff Exit Rules:
                # 1. Hard stop loss: -4.5% from entry (ultra-tight loss cutting)
                # 2. Profit preservation: for strong winners (+15%+), trail stop at -8% from peak
                # 3. Trend exhaustion: Close drops below EMA20 and breaks trend
                hit_hard_stop = close_vals[i] < entry_p * 0.955
                hit_profit_trail = (peak_p > entry_p * 1.15) and (close_vals[i] < peak_p * 0.92)
                hit_ema20_break = (close_vals[i] < ema20_vals[i] * 0.982)
                hit_distribution = (upthrust_vals[i] == 1) and (close_vals[i] < ema20_vals[i])

                if hit_hard_stop or hit_profit_trail or hit_ema20_break or hit_distribution:
                    pos = 0
                    entry_p = 0.0
                    peak_p = 0.0
            in_position[i] = pos

        data["Whale_Signal"] = entry_signal
        data["In_Whale_Position"] = in_position

        # 8. Insider & Regulatory Intelligence Convergence Overlay
        insider_boost = 1.0
        if self.use_insider_data and self.insider_engine and symbol and market == "US":
            try:
                insider_df = self.insider_engine.compute_insider_signals(symbol, data["Close"])
                if "Insider_Conviction_Boost" in insider_df:
                    insider_boost = float(insider_df["Insider_Conviction_Boost"].iloc[-1])
                    data["Insider_Score"] = insider_df["Insider_Score"]
                    data["Insider_Buy_Ratio"] = insider_df["Insider_Buy_Ratio"]
            except Exception as e:
                logger.warning(f"Could not compute insider convergence for {symbol}: {e}")
                data["Insider_Score"] = 0.0
                data["Insider_Buy_Ratio"] = 0.50
        else:
            data["Insider_Score"] = 0.0
            data["Insider_Buy_Ratio"] = 0.50

        # Composite Alpha Score: Convergence of Tape Momentum + Insider Smart Money Conviction
        composite_score = (
            data["Mom_Score"] * 3.5 +
            np.clip(data["Vol_ZScore"] / 1.0, 0.0, 6.0) +
            (data["Wyckoff_Absorption"] * 2.5) +
            (data["Volume_Breakout"] * 2.5) +
            np.clip(data["OFI_Norm_Z"] * 2.0, 0.0, 4.0)
        ) * insider_boost

        data["Composite_Alpha_Score"] = np.where(data["In_Whale_Position"] == 1, composite_score, 0.0)
        data["Conviction_Multiplier"] = np.where(data["In_Whale_Position"] == 1, 1.0 + np.clip(composite_score / 2.0, 0.0, 6.0), 0.0)

        return data
