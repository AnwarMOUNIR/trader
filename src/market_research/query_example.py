"""Demonstration query using DuckDB to join Point-in-Time S&P 500 membership with intraday candles without survivorship or look-ahead bias."""

import duckdb
import pandas as pd
from pathlib import Path

def run_point_in_time_query(query_date: str = "2023-05-15"):
    """
    Executes a point-in-time query using DuckDB.
    Filters the security master / membership intervals to only constituents active on `query_date`,
    and joins their 30-minute intraday candles on that date.
    """
    conn = duckdb.connect()

    meta_path = "data/metadata/membership_intervals.parquet"
    candles_30_glob = "data/candles/30min/*.parquet"

    print(f"=== POINT-IN-TIME S&P 500 QUERY FOR DATE: {query_date} ===")

    query = f"""
    WITH pit_members AS (
        SELECT 
            security_id,
            ticker,
            start_date,
            end_date
        FROM read_parquet('{meta_path}')
        WHERE start_date <= '{query_date}'
          AND (end_date IS NULL OR end_date >= '{query_date}')
    ),
    candles AS (
        SELECT 
            symbol,
            session_date,
            exchange_time_ny,
            duration_minutes,
            is_partial,
            open,
            high,
            low,
            close,
            volume,
            vwap
        FROM read_parquet('{candles_30_glob}')
        WHERE session_date = '{query_date}'
    )
    SELECT 
        m.security_id,
        m.ticker,
        c.exchange_time_ny,
        c.duration_minutes,
        c.is_partial,
        c.open,
        c.high,
        c.low,
        c.close,
        c.volume,
        c.vwap
    FROM pit_members m
    INNER JOIN candles c ON m.ticker = c.symbol
    ORDER BY m.ticker, c.exchange_time_ny;
    """

    try:
        res = conn.execute(query).df()
        print(f"Query returned {len(res)} candle rows for active constituents on {query_date}:")
        if not res.empty:
            print(res.head(15).to_string())
        else:
            print("No matching candles for this specific date in current pilot files.")
        return res
    except Exception as e:
        print(f"DuckDB Query execution note: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    run_point_in_time_query()
