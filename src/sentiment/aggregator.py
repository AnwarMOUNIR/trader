"""
Point-in-Time Exponential Time-Decayed Sentiment Aggregator.

Synthesizes discrete, irregularly timed sentiment scores into continuous, regularly spaced
sentiment feature time-series aligned with intraday and daily price bars with zero lookahead bias.
"""

from __future__ import annotations

import bisect
import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from src.sentiment.deduplicator import SQLiteNewsStore
from src.sentiment.nlp_engine import NLPSentimentEngine, SentimentResult
from src.sentiment.rss_crawler import NewsArticle

logger = logging.getLogger(__name__)


class SentimentAggregator:
    """
    Point-in-Time Exponential Decay Sentiment Aggregator.

    Computes 7 standardized sentiment features for any symbol and target DatetimeIndex:
    - sentiment_score_ema: Exponentially decaying weighted mean sentiment score in [-1.0, 1.0]
    - sentiment_confidence_ema: Exponentially decaying weighted mean confidence in [0.0, 1.0]
    - news_volume_1h: Count of articles published in (t - 1h, t]
    - news_volume_24h: Count of articles published in (t - 24h, t]
    - sentiment_dispersion: Weighted sentiment variance / narrative disagreement
    - sentiment_momentum: Narrative velocity change (delta vs prior rolling window)
    - bull_bear_ratio: Ratio of positive to negative stories over 24h window
    """

    FEATURE_COLUMNS = [
        "sentiment_score_ema",
        "sentiment_confidence_ema",
        "news_volume_1h",
        "news_volume_24h",
        "sentiment_dispersion",
        "sentiment_momentum",
        "bull_bear_ratio",
    ]

    def __init__(
        self,
        news_store: Optional[SQLiteNewsStore] = None,
        nlp_engine: Optional[NLPSentimentEngine] = None,
        max_lookback_days: float = 7.0,
    ) -> None:
        self.news_store = news_store
        self.nlp_engine = nlp_engine or NLPSentimentEngine()
        self.max_lookback_days = max_lookback_days

        # In-memory storage: symbol -> sorted list of (pub_timestamp_sec, score, confidence, is_pos, is_neg)
        self._symbol_sentiment_index: Dict[str, List[Tuple[float, float, float, bool, bool]]] = {}

    def add_sentiment_results(self, results: Sequence[SentimentResult]) -> None:
        """Add pre-scored SentimentResult instances into the aggregator index."""
        for res in results:
            sym = res.symbol.upper()
            pub_ts = (
                res.published_at_utc.timestamp()
                if res.published_at_utc.tzinfo
                else res.published_at_utc.replace(tzinfo=timezone.utc).timestamp()
            )
            score = float(res.sentiment_score)
            conf = float(res.sentiment_confidence)
            is_pos = score > 0.05 or res.sentiment_label == "POSITIVE"
            is_neg = score < -0.05 or res.sentiment_label == "NEGATIVE"

            if sym not in self._symbol_sentiment_index:
                self._symbol_sentiment_index[sym] = []

            self._symbol_sentiment_index[sym].append((pub_ts, score, conf, is_pos, is_neg))

        # Re-sort indexed entries by timestamp
        for sym in self._symbol_sentiment_index:
            self._symbol_sentiment_index[sym].sort(key=lambda x: x[0])

    def add_articles(self, articles: Sequence[NewsArticle]) -> List[SentimentResult]:
        """Score and add raw NewsArticle instances into the aggregator index."""
        results = self.nlp_engine.score_batch(articles)
        self.add_sentiment_results(results)
        if self.news_store:
            self.news_store.store_articles(articles)
            self.news_store.store_sentiment_results(results)
        return results

    def _load_from_store_if_needed(self, symbol: str) -> None:
        """Load historical scores from SQLite news_store if available and not yet loaded."""
        if self.news_store and symbol.upper() not in self._symbol_sentiment_index:
            scores = self.news_store.get_sentiment_scores(symbol=symbol.upper())
            if scores:
                loaded_results = []
                for s in scores:
                    pub_dt = datetime.fromisoformat(s["published_at_utc"])
                    if pub_dt.tzinfo is None:
                        pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                    loaded_results.append(
                        SentimentResult(
                            article_id=s["article_id"],
                            symbol=s["symbol"],
                            published_at_utc=pub_dt,
                            sentiment_score=float(s["sentiment_score"]),
                            sentiment_confidence=float(s["sentiment_confidence"]),
                            sentiment_label=s["sentiment_label"],
                            probabilities={
                                "positive": s.get("pos_prob", 0.0),
                                "neutral": s.get("neu_prob", 0.0),
                                "negative": s.get("neg_prob", 0.0),
                            },
                            model_name=s.get("model_name", "stored"),
                        )
                    )
                self.add_sentiment_results(loaded_results)

    def get_sentiment_features(
        self,
        symbol: str,
        target_timestamps: Union[pd.DatetimeIndex, Sequence[datetime]],
        half_life_hours: float = 4.0,
    ) -> pd.DataFrame:
        """
        Compute point-in-time exponentially decaying sentiment features for each timestamp in target_timestamps.

        Parameters
        ----------
        symbol : str
            Equity ticker symbol (e.g. "NVDA", "AAPL", "ATW.BVC").
        target_timestamps : pd.DatetimeIndex or Sequence[datetime]
            Target evaluation timestamps (bar start/close).
        half_life_hours : float, default 4.0
            Half-life decay horizon in hours (4.0 for intraday, 24.0 for daily).

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by target_timestamps with columns:
            ["sentiment_score_ema", "sentiment_confidence_ema", "news_volume_1h",
             "news_volume_24h", "sentiment_dispersion", "sentiment_momentum", "bull_bear_ratio"]
        """
        sym_clean = symbol.upper()
        self._load_from_store_if_needed(sym_clean)

        # Convert target_timestamps to UTC DatetimeIndex
        if not isinstance(target_timestamps, pd.DatetimeIndex):
            dt_index = pd.to_datetime(target_timestamps, utc=True)
        else:
            if target_timestamps.tz is None:
                dt_index = target_timestamps.tz_localize("UTC")
            else:
                dt_index = target_timestamps.tz_convert("UTC")

        n_bars = len(dt_index)
        if n_bars == 0:
            return pd.DataFrame(columns=self.FEATURE_COLUMNS, index=dt_index)

        # Initialize result arrays
        score_ema = np.zeros(n_bars, dtype=np.float64)
        conf_ema = np.zeros(n_bars, dtype=np.float64)
        vol_1h = np.zeros(n_bars, dtype=np.int32)
        vol_24h = np.zeros(n_bars, dtype=np.int32)
        dispersion = np.zeros(n_bars, dtype=np.float64)
        momentum = np.zeros(n_bars, dtype=np.float64)
        bull_bear = np.ones(n_bars, dtype=np.float64)  # Default 1.0 (neutral 1/1)

        symbol_records = self._symbol_sentiment_index.get(sym_clean, [])
        if not symbol_records:
            # No sentiment records available; return clean neutral baseline
            df_res = pd.DataFrame(
                {
                    "sentiment_score_ema": score_ema,
                    "sentiment_confidence_ema": conf_ema,
                    "news_volume_1h": vol_1h,
                    "news_volume_24h": vol_24h,
                    "sentiment_dispersion": dispersion,
                    "sentiment_momentum": momentum,
                    "bull_bear_ratio": bull_bear,
                },
                index=dt_index,
            )
            return df_res

        # Extract record arrays for high-speed indexing
        timestamps = np.array([r[0] for r in symbol_records], dtype=np.float64)
        scores = np.array([r[1] for r in symbol_records], dtype=np.float64)
        confs = np.array([r[2] for r in symbol_records], dtype=np.float64)
        is_pos_arr = np.array([r[3] for r in symbol_records], dtype=bool)
        is_neg_arr = np.array([r[4] for r in symbol_records], dtype=bool)

        half_life_sec = half_life_hours * 3600.0
        decay_lambda = math.log(2.0) / half_life_sec
        max_lookback_sec = self.max_lookback_days * 86400.0

        # Unit-agnostic timestamp extraction in POSIX seconds
        target_ts_seconds = np.array([t.timestamp() for t in dt_index], dtype=np.float64)

        for k, t_sec in enumerate(target_ts_seconds):
            # Strict Point-in-Time condition: publication timestamp <= target evaluation timestamp
            # Find insertion point using binary search
            idx_end = bisect.bisect_right(timestamps, t_sec)
            if idx_end == 0:
                # No news published strictly on or before t_sec
                continue

            # Lookback filter
            t_min = t_sec - max_lookback_sec
            idx_start = bisect.bisect_left(timestamps, t_min, 0, idx_end)

            if idx_start >= idx_end:
                continue

            # Active slice of articles
            sub_ts = timestamps[idx_start:idx_end]
            sub_scores = scores[idx_start:idx_end]
            sub_confs = confs[idx_start:idx_end]
            sub_pos = is_pos_arr[idx_start:idx_end]
            sub_neg = is_neg_arr[idx_start:idx_end]

            # Compute exponential weights: w_i = exp(-lambda * (t - t_i))
            dt_deltas = t_sec - sub_ts
            weights = np.exp(-decay_lambda * dt_deltas)
            total_weight = np.sum(weights)

            if total_weight > 0:
                weighted_score = np.sum(weights * sub_scores) / total_weight
                weighted_conf = np.sum(weights * sub_confs) / total_weight

                # Weighted dispersion (variance)
                disp = np.sum(weights * ((sub_scores - weighted_score) ** 2)) / total_weight

                score_ema[k] = np.clip(weighted_score, -1.0, 1.0)
                conf_ema[k] = np.clip(weighted_conf, 0.0, 1.0)
                dispersion[k] = max(0.0, disp)

            # 1-Hour and 24-Hour Volume & Bull/Bear Counts
            t_1h = t_sec - 3600.0
            t_24h = t_sec - 86400.0

            mask_1h = sub_ts >= t_1h
            vol_1h[k] = int(np.sum(mask_1h))

            mask_24h = sub_ts >= t_24h
            vol_24h[k] = int(np.sum(mask_24h))

            pos_count_24h = int(np.sum(sub_pos[mask_24h]))
            neg_count_24h = int(np.sum(sub_neg[mask_24h]))
            bull_bear[k] = (pos_count_24h + 1.0) / (neg_count_24h + 1.0)

            # Sentiment Momentum (Delta vs 3 bars prior or 1 hour prior)
            lookback_bars = min(3, k)
            if lookback_bars > 0:
                momentum[k] = score_ema[k] - score_ema[k - lookback_bars]

        df_res = pd.DataFrame(
            {
                "sentiment_score_ema": np.round(score_ema, 4),
                "sentiment_confidence_ema": np.round(conf_ema, 4),
                "news_volume_1h": vol_1h,
                "news_volume_24h": vol_24h,
                "sentiment_dispersion": np.round(dispersion, 4),
                "sentiment_momentum": np.round(momentum, 4),
                "bull_bear_ratio": np.round(bull_bear, 4),
            },
            index=dt_index,
        )

        return df_res

    def fuse_with_price_bars(
        self,
        ohlcv_df: pd.DataFrame,
        symbol: str,
        half_life_hours: float = 4.0,
    ) -> pd.DataFrame:
        """
        Fuse sentiment features directly with an OHLCV price DataFrame.
        Guarantees point-in-time alignment with no future lookahead leakage.
        """
        if ohlcv_df.empty:
            return ohlcv_df.copy()

        sentiment_df = self.get_sentiment_features(
            symbol=symbol,
            target_timestamps=ohlcv_df.index,
            half_life_hours=half_life_hours,
        )

        # Merge side-by-side on index
        fused = ohlcv_df.join(sentiment_df, how="left")
        # Fill any missing values with clean defaults
        fused["sentiment_score_ema"] = fused["sentiment_score_ema"].fillna(0.0)
        fused["sentiment_confidence_ema"] = fused["sentiment_confidence_ema"].fillna(0.0)
        fused["news_volume_1h"] = fused["news_volume_1h"].fillna(0).astype(int)
        fused["news_volume_24h"] = fused["news_volume_24h"].fillna(0).astype(int)
        fused["sentiment_dispersion"] = fused["sentiment_dispersion"].fillna(0.0)
        fused["sentiment_momentum"] = fused["sentiment_momentum"].fillna(0.0)
        fused["bull_bear_ratio"] = fused["bull_bear_ratio"].fillna(1.0)

        return fused
