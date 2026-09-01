"""
Data Subsystem for Dual-Resolution Ingestion, Columnar Storage, and Multi-Timeframe Alignment.
"""

from .loader import MarketDataLoader, UniverseConfig, AssetMarket
from .parquet_store import ParquetDataStore
from .synthetic_feed import SyntheticMicrostructureGenerator, MarketMicrostructureParams, ASSET_PRESETS
from .aligner import MultiTimeframeAligner, resample_ohlcv
from .historical_loader import HistoricalDataLoader

__all__ = [
    "MarketDataLoader",
    "UniverseConfig",
    "AssetMarket",
    "ParquetDataStore",
    "SyntheticMicrostructureGenerator",
    "MarketMicrostructureParams",
    "ASSET_PRESETS",
    "MultiTimeframeAligner",
    "resample_ohlcv",
    "HistoricalDataLoader"
]
