"""
Columnar Apache Parquet Store & SQLite Metadata Catalog Subsystem.
Provides high-performance ZSTD-compressed partitioned storage and relational
indexing for dual-resolution market data (daily and intraday bars).
"""

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger("DataPipeline.ParquetStore")


class ParquetDataStore:
    """
    Manages partitioned Parquet files and the SQLite metadata catalog.
    Partitioning layout: {base_dir}/parquet/{timeframe}/{safe_symbol}.parquet
    SQLite Catalog: {base_dir}/market_catalog.db
    """

    def __init__(
        self,
        base_dir: Union[str, Path] = "data_cache",
        db_name: str = "market_catalog.db"
    ):
        self.base_dir = Path(base_dir)
        self.parquet_dir = self.base_dir / "parquet"
        self.db_path = self.base_dir / db_name

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_dir.mkdir(parents=True, exist_ok=True)

        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite metadata catalog tables with indexing."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Asset Registry Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS asset_registry (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    market TEXT NOT NULL,
                    currency TEXT NOT NULL,
                    sector TEXT,
                    is_active INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Ingestion Manifest Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_manifest (
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    first_timestamp TEXT NOT NULL,
                    last_timestamp TEXT NOT NULL,
                    total_bars INTEGER NOT NULL,
                    file_path TEXT NOT NULL,
                    file_size_bytes INTEGER NOT NULL,
                    checksum_sha256 TEXT NOT NULL,
                    last_updated_utc TEXT NOT NULL,
                    PRIMARY KEY (symbol, timeframe)
                );
            """)

            # Indices for rapid querying
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_manifest_timeframe ON ingestion_manifest (timeframe);
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_asset_market ON asset_registry (market);
            """)
            conn.commit()

    @staticmethod
    def _sanitize_symbol(symbol: str) -> str:
        """Sanitize symbol string for filesystem compatibility."""
        return symbol.replace(".", "_").replace("/", "_").replace("\\", "_").upper()

    def _get_parquet_path(self, symbol: str, timeframe: str) -> Path:
        """Get absolute path to Parquet file for a given symbol and timeframe."""
        tf_dir = self.parquet_dir / timeframe.lower()
        tf_dir.mkdir(parents=True, exist_ok=True)
        safe_sym = self._sanitize_symbol(symbol)
        return tf_dir / f"{safe_sym}.parquet"

    @staticmethod
    def _compute_sha256(file_path: Path) -> str:
        """Compute SHA-256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def store_bars(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        market: str = "US",
        currency: str = "USD",
        name: Optional[str] = None,
        sector: Optional[str] = None
    ) -> str:
        """
        Store bar data to Parquet using PyArrow with ZSTD compression and update SQLite catalog.
        """
        if df.empty:
            raise ValueError(f"Cannot store empty DataFrame for {symbol} ({timeframe}).")

        # Standardize DataFrame
        data_df = df.copy()
        
        # Ensure timestamp index is DatetimeIndex with UTC
        if not isinstance(data_df.index, pd.DatetimeIndex):
            if "timestamp" in data_df.columns:
                data_df["timestamp"] = pd.to_datetime(data_df["timestamp"], utc=True)
                data_df.set_index("timestamp", inplace=True)
            else:
                data_df.index = pd.to_datetime(data_df.index, utc=True)

        if data_df.index.tz is None:
            data_df.index = data_df.index.tz_localize("UTC")
        else:
            data_df.index = data_df.index.tz_convert("UTC")

        data_df.index.name = "timestamp"
        data_df.sort_index(inplace=True)

        # Standardize column casing to lowercase
        data_df.columns = [c.lower() for c in data_df.columns]

        # Ensure required columns exist
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in data_df.columns:
                raise ValueError(f"Missing required price column '{col}' in DataFrame for {symbol}.")

        # Auto-compute VWAP if not present
        if "vwap" not in data_df.columns:
            data_df["vwap"] = (data_df["high"] + data_df["low"] + data_df["close"]) / 3.0

        target_file = self._get_parquet_path(symbol, timeframe)
        
        # Convert to PyArrow Table and write with ZSTD compression level 3
        table = pa.Table.from_pandas(data_df, preserve_index=True)
        pq.write_table(
            table,
            target_file,
            compression="ZSTD",
            compression_level=3,
            version="2.6"
        )

        file_size = target_file.stat().st_size
        checksum = self._compute_sha256(target_file)
        first_ts = data_df.index[0].isoformat()
        last_ts = data_df.index[-1].isoformat()
        total_bars = len(data_df)
        now_utc = datetime.now(timezone.utc).isoformat()

        # Update SQLite Manifest and Asset Registry atomically
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Upsert Asset Registry
            cursor.execute("""
                INSERT INTO asset_registry (symbol, name, market, currency, sector, is_active)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(symbol) DO UPDATE SET
                    market = excluded.market,
                    currency = excluded.currency,
                    name = COALESCE(excluded.name, asset_registry.name),
                    sector = COALESCE(excluded.sector, asset_registry.sector);
            """, (symbol.upper(), name, market.upper(), currency.upper(), sector))

            # Upsert Ingestion Manifest
            cursor.execute("""
                INSERT INTO ingestion_manifest (
                    symbol, timeframe, first_timestamp, last_timestamp,
                    total_bars, file_path, file_size_bytes, checksum_sha256, last_updated_utc
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, timeframe) DO UPDATE SET
                    first_timestamp = excluded.first_timestamp,
                    last_timestamp = excluded.last_timestamp,
                    total_bars = excluded.total_bars,
                    file_path = excluded.file_path,
                    file_size_bytes = excluded.file_size_bytes,
                    checksum_sha256 = excluded.checksum_sha256,
                    last_updated_utc = excluded.last_updated_utc;
            """, (
                symbol.upper(), timeframe.lower(), first_ts, last_ts,
                total_bars, str(target_file), file_size, checksum, now_utc
            ))
            conn.commit()

        logger.debug(f"Stored {total_bars} bars for {symbol} ({timeframe}) at {target_file}")
        return str(target_file)

    def load_bars(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[Union[str, pd.Timestamp]] = None,
        end: Optional[Union[str, pd.Timestamp]] = None,
        columns: Optional[List[str]] = None
    ) -> Optional[pd.DataFrame]:
        """
        Load bar data from Parquet file using column pruning and optional timestamp filtering.
        Returns None if file does not exist.
        """
        file_path = self._get_parquet_path(symbol, timeframe)
        if not file_path.exists():
            return None

        try:
            # Prepare column list for pruning if requested
            cols_to_read = None
            if columns is not None:
                # Ensure column names are lowercase
                cols_to_read = [c.lower() for c in columns]
                # If index column is explicitly stored, pyarrow handles it
                if "timestamp" not in cols_to_read and "timestamp" in pq.read_schema(file_path).names:
                    cols_to_read.append("timestamp")

            # Read Parquet using pandas/pyarrow engine
            df = pd.read_parquet(file_path, columns=cols_to_read)

            # Ensure UTC DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                if "timestamp" in df.columns:
                    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
                    df.set_index("timestamp", inplace=True)
                else:
                    df.index = pd.to_datetime(df.index, utc=True)
            elif df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            df.sort_index(inplace=True)

            # Apply date filters
            if start is not None:
                start_ts = pd.to_datetime(start, utc=True)
                df = df[df.index >= start_ts]
            if end is not None:
                end_ts = pd.to_datetime(end, utc=True)
                df = df[df.index <= end_ts]

            return df
        except Exception as e:
            logger.error(f"Error loading Parquet file {file_path}: {e}")
            return None

    def has_bars(self, symbol: str, timeframe: str) -> bool:
        """Check whether bar data exists in the Parquet store."""
        file_path = self._get_parquet_path(symbol, timeframe)
        return file_path.exists() and file_path.stat().st_size > 0

    def get_manifest_entry(self, symbol: str, timeframe: str) -> Optional[Dict[str, Any]]:
        """Retrieve manifest record for symbol and timeframe."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM ingestion_manifest
                WHERE symbol = ? AND timeframe = ?;
            """, (symbol.upper(), timeframe.lower()))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_catalog(self, market: Optional[str] = None) -> List[Dict[str, Any]]:
        """List registered assets in the catalog."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            if market:
                cursor.execute("SELECT * FROM asset_registry WHERE market = ? ORDER BY symbol;", (market.upper(),))
            else:
                cursor.execute("SELECT * FROM asset_registry ORDER BY symbol;")
            return [dict(row) for row in cursor.fetchall()]

    def list_manifest(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List ingestion manifest entries with optional filtering."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            query = "SELECT * FROM ingestion_manifest WHERE 1=1"
            params: List[Any] = []
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol.upper())
            if timeframe:
                query += " AND timeframe = ?"
                params.append(timeframe.lower())
            query += " ORDER BY symbol, timeframe;"
            cursor.execute(query, tuple(params))
            return [dict(row) for row in cursor.fetchall()]

    def list_stored_symbols(self, timeframe: Optional[str] = None) -> List[str]:
        """List distinct symbols available in the manifest."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if timeframe:
                cursor.execute(
                    "SELECT DISTINCT symbol FROM ingestion_manifest WHERE timeframe = ? ORDER BY symbol;",
                    (timeframe.lower(),)
                )
            else:
                cursor.execute("SELECT DISTINCT symbol FROM ingestion_manifest ORDER BY symbol;")
            return [r[0] for r in cursor.fetchall()]

    def delete_bars(self, symbol: str, timeframe: Optional[str] = None) -> bool:
        """Delete bar files and corresponding manifest entries."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if timeframe:
                file_path = self._get_parquet_path(symbol, timeframe)
                if file_path.exists():
                    file_path.unlink()
                cursor.execute(
                    "DELETE FROM ingestion_manifest WHERE symbol = ? AND timeframe = ?;",
                    (symbol.upper(), timeframe.lower())
                )
            else:
                # Delete all timeframes for symbol
                cursor.execute(
                    "SELECT timeframe, file_path FROM ingestion_manifest WHERE symbol = ?;",
                    (symbol.upper(),)
                )
                rows = cursor.fetchall()
                for tf, fpath in rows:
                    p = Path(fpath)
                    if p.exists():
                        p.unlink()
                cursor.execute("DELETE FROM ingestion_manifest WHERE symbol = ?;", (symbol.upper(),))
                cursor.execute("DELETE FROM asset_registry WHERE symbol = ?;", (symbol.upper(),))
            conn.commit()
        return True

    def verify_checksum(self, symbol: str, timeframe: str) -> bool:
        """Verify that the Parquet file on disk matches the SHA-256 hash in SQLite manifest."""
        entry = self.get_manifest_entry(symbol, timeframe)
        if not entry:
            return False
        file_path = Path(entry["file_path"])
        if not file_path.exists():
            return False
        current_hash = self._compute_sha256(file_path)
        return current_hash == entry["checksum_sha256"]
