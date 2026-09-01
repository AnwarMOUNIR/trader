"""
Triple-Barrier Labeler & Multi-Horizon Forward Return Generator.

Implements Marcos López de Prado's Triple-Barrier Method for institutional-grade financial labeling:
- Upper Profit Target Barrier (+1.5 * sigma_t)
- Lower Stop-Loss Target Barrier (-1.0 * sigma_t)
- Vertical Time Barrier (12 bars horizon)
- Multi-Horizon Continuous Forward Returns (h in {3, 12, 78} bars)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("Models.Labeler")


@dataclass
class BarrierConfig:
    """Configuration for Triple-Barrier Method."""
    upper_mult: float = 1.5           # +1.5 * sigma profit target
    lower_mult: float = 1.0           # -1.0 * sigma stop loss
    vertical_barrier_bars: int = 12   # Maximum holding period (bars)
    vol_window: int = 20              # Rolling volatility window
    horizons: Tuple[int, ...] = (3, 12, 78)  # Multi-horizon forecast steps
    horizon_weights: Tuple[float, ...] = (0.2, 0.3, 0.5)


@dataclass
class LabelResult:
    """Container for Triple-Barrier and Multi-Horizon labeling results."""
    labels_df: pd.DataFrame
    class_distribution: pd.Series
    mean_horizon_returns: pd.Series


class TripleBarrierLabeler:
    """
    Marcos López de Prado's Triple-Barrier Method Labeler.
    Generates point-in-time ground truth supervisory targets for deep neural networks.
    """

    def __init__(self, config: Optional[BarrierConfig] = None) -> None:
        self.config = config or BarrierConfig()

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
        return df_copy.rename(columns=col_map)

    def compute_volatility(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute dynamic rolling volatility sigma_t (ATR or standard deviation of returns).
        Uses rolling standard deviation of log returns protected with minimum floor.
        """
        df_std = self._standardize_ohlcv(df)
        close = df_std["Close"].astype(float)
        log_ret = np.log(close / close.shift(1)).fillna(0.0)
        vol = log_ret.rolling(window=self.config.vol_window, min_periods=3).std()
        
        # Volatility floor (0.002 = 20 bps) to prevent zero-width barriers
        return vol.fillna(0.01).clip(lower=0.002)

    def generate_labels(
        self,
        df: pd.DataFrame,
    ) -> LabelResult:
        """
        Apply Triple-Barrier labeling and compute multi-horizon forward returns.

        Parameters
        ----------
        df : pd.DataFrame
            Intraday or daily OHLCV DataFrame with UTC DatetimeIndex.

        Returns
        -------
        LabelResult
            labels_df containing:
            - "label_triple_barrier": +1 (BUY), 0 (HOLD), -1 (SELL)
            - "label_class": 2 (BUY), 1 (HOLD), 0 (SELL)
            - "action_name": "BUY", "HOLD", "SELL"
            - "touch_barrier": "UPPER", "LOWER", "VERTICAL"
            - "bar_volatility": sigma_t
            - "ret_h3", "ret_h12", "ret_h78": continuous forward returns
            - "composite_return": weighted multi-horizon return
        """
        if df.empty:
            empty_df = pd.DataFrame(
                columns=[
                    "label_triple_barrier",
                    "label_class",
                    "action_name",
                    "touch_barrier",
                    "bar_volatility",
                    "ret_h3",
                    "ret_h12",
                    "ret_h78",
                    "composite_return",
                ],
                index=df.index,
            )
            return LabelResult(empty_df, pd.Series(dtype=int), pd.Series(dtype=float))

        df_sorted = self._standardize_ohlcv(df).sort_index()
        n = len(df_sorted)
        close = df_sorted["Close"].values.astype(np.float64)
        high = df_sorted["High"].values.astype(np.float64)
        low = df_sorted["Low"].values.astype(np.float64)

        vol_series = self.compute_volatility(df_sorted).values.astype(np.float64)

        tb_labels = np.zeros(n, dtype=np.int32)       # +1, 0, -1
        class_labels = np.ones(n, dtype=np.int64)    # 2=BUY, 1=HOLD, 0=SELL
        touch_barriers = np.array(["VERTICAL"] * n, dtype=object)
        action_names = np.array(["HOLD"] * n, dtype=object)

        # Precompute multi-horizon forward returns: (P_{t+h} - P_t) / P_t
        h_returns: Dict[int, np.ndarray] = {}
        for h in self.config.horizons:
            ret_arr = np.zeros(n, dtype=np.float64)
            for t in range(n):
                if t + h < n:
                    ret_arr[t] = (close[t + h] - close[t]) / max(close[t], 1e-6)
                else:
                    ret_arr[t] = (close[-1] - close[t]) / max(close[t], 1e-6)
            h_returns[h] = ret_arr

        # Triple barrier evaluation loop
        vert_limit = self.config.vertical_barrier_bars
        u_mult = self.config.upper_mult
        l_mult = self.config.lower_mult

        for t in range(n):
            p0 = close[t]
            sigma = vol_series[t]
            upper_barrier = p0 * (1.0 + u_mult * sigma)
            lower_barrier = p0 * (1.0 - l_mult * sigma)

            max_t = min(t + vert_limit, n - 1)
            first_touch = "VERTICAL"
            lbl = 0

            # Scan future bars t+1 ... max_t
            for future_t in range(t + 1, max_t + 1):
                h_val = high[future_t]
                l_val = low[future_t]

                upper_hit = h_val >= upper_barrier
                lower_hit = l_val <= lower_barrier

                if upper_hit and lower_hit:
                    # Both hit in same bar; check bar close direction relative to open
                    if close[future_t] >= p0:
                        first_touch = "UPPER"
                        lbl = 1
                    else:
                        first_touch = "LOWER"
                        lbl = -1
                    break
                elif upper_hit:
                    first_touch = "UPPER"
                    lbl = 1
                    break
                elif lower_hit:
                    first_touch = "LOWER"
                    lbl = -1
                    break

            tb_labels[t] = lbl
            touch_barriers[t] = first_touch
            if lbl == 1:
                class_labels[t] = 2
                action_names[t] = "BUY"
            elif lbl == -1:
                class_labels[t] = 0
                action_names[t] = "SELL"
            else:
                class_labels[t] = 1
                action_names[t] = "HOLD"

        # Compute composite weighted return
        w_sum = sum(self.config.horizon_weights)
        norm_weights = [w / w_sum for w in self.config.horizon_weights]
        composite_ret = np.zeros(n, dtype=np.float64)
        for w, h in zip(norm_weights, self.config.horizons):
            composite_ret += w * h_returns[h]

        result_dict = {
            "label_triple_barrier": tb_labels,
            "label_class": class_labels,
            "action_name": action_names,
            "touch_barrier": touch_barriers,
            "bar_volatility": np.round(vol_series, 6),
            "composite_return": np.round(composite_ret, 6),
        }
        for h in self.config.horizons:
            result_dict[f"ret_h{h}"] = np.round(h_returns[h], 6)

        res_df = pd.DataFrame(result_dict, index=df_sorted.index)
        class_dist = res_df["action_name"].value_counts()
        mean_returns = res_df[[f"ret_h{h}" for h in self.config.horizons]].mean()

        return LabelResult(
            labels_df=res_df,
            class_distribution=class_dist,
            mean_horizon_returns=mean_returns,
        )
