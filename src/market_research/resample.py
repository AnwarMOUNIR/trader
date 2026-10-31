"""Ultra-fast vectorized candle resampling engine adhering strictly to regular US sessions, left-closed windows, and DST handling."""

import zoneinfo
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import exchange_calendars as xcals

NY_TZ = zoneinfo.ZoneInfo("America/New_York")
UTC = zoneinfo.ZoneInfo("UTC")

_CALENDAR = None
_SESSION_CACHE = None

def get_exchange_calendar():
    global _CALENDAR
    if _CALENDAR is None:
        _CALENDAR = xcals.get_calendar("XNYS", start="1950-01-01", end="2035-12-31")
    return _CALENDAR

def get_session_cache(start_date: str | None = None, end_date: str | None = None):
    """Return actual XNYS sessions, optionally bounded to an input date range.

    Exchange calendar bounds are authoritative: dates outside the calendar's
    supported range raise instead of being guessed as regular sessions.
    """
    global _SESSION_CACHE
    if _SESSION_CACHE is None:
        cal = get_exchange_calendar()
        sessions = cal.sessions_in_range(cal.first_session, cal.last_session)
        opens_ny = cal.opens.loc[sessions].dt.tz_convert("America/New_York")
        closes_ny = cal.closes.loc[sessions].dt.tz_convert("America/New_York")
        _SESSION_CACHE = {
            s.strftime("%Y-%m-%d"): (o, c)
            for s, o, c in zip(sessions, opens_ny, closes_ny)
        }
    if start_date is None and end_date is None:
        return _SESSION_CACHE
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date must be supplied together")
    cal = get_exchange_calendar()
    try:
        start = pd.Timestamp(start_date).normalize()
        end = pd.Timestamp(end_date).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("Calendar bounds must be valid dates") from exc
    first = pd.Timestamp(cal.first_session).tz_localize(None) if pd.Timestamp(cal.first_session).tzinfo else pd.Timestamp(cal.first_session)
    last = pd.Timestamp(cal.last_session).tz_localize(None) if pd.Timestamp(cal.last_session).tzinfo else pd.Timestamp(cal.last_session)
    if start > end:
        raise ValueError("start_date must be on or before end_date")
    if start < first or end > last:
        raise ValueError(f"Requested range must be within XNYS calendar bounds {first.date()}–{last.date()}")
    start_key, end_key = start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")
    return {day: bounds for day, bounds in _SESSION_CACHE.items() if start_key <= day <= end_key}

def get_exchange_sessions(start_date: str, end_date: str):
    cal = get_exchange_calendar()
    return cal.sessions_in_range(start_date, end_date)

def derive_candles_for_session(
    df_session: pd.DataFrame,
    session_date: str,
    interval_minutes: int = 30,
) -> list[dict]:
    """
    Vectorized derivation of left-closed, right-open [start, end) candles for a regular session.
    30min interval: 13 full candles for normal 390 min session.
    20min interval: 19 full candles + 1 final 10-min candle (is_partial=True).
    """
    if isinstance(interval_minutes, bool) or not isinstance(interval_minutes, (int, np.integer)) or interval_minutes <= 0:
        raise ValueError("interval_minutes must be a positive integer")
    if df_session.empty:
        return []

    cache = get_session_cache()
    try:
        open_time, close_time = cache[session_date]
    except KeyError as exc:
        raise ValueError(f"{session_date} is not a supported XNYS session") from exc

    df = df_session.copy()
    if "time_ny" not in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        df["time_ny"] = df["timestamp_utc"].dt.tz_convert("America/New_York")
    else:
        df["time_ny"] = pd.to_datetime(df["time_ny"], utc=True).dt.tz_convert("America/New_York")
    if df["time_ny"].duplicated().any():
        raise ValueError("Duplicate minute timestamps are not allowed within a session")
    df = df.sort_values("time_ny", kind="stable")

    # Filter strictly within session window [open_time, close_time)
    mask = (df["time_ny"] >= open_time) & (df["time_ny"] < close_time)
    sdf = df[mask]
    if sdf.empty:
        return []

    total_session_minutes = int((close_time - open_time).total_seconds() / 60)
    mins = ((sdf["time_ny"] - open_time).dt.total_seconds() // 60).astype(int)
    buckets = mins // interval_minutes

    has_trade_count = "trade_count" in sdf.columns and sdf["trade_count"].notna().any()
    has_vwap = "vwap" in sdf.columns and sdf["vwap"].notna().any()

    # Pre-calculate v_vwap if available
    if has_vwap:
        sdf = sdf.copy()
        sdf["v_vwap"] = sdf["vwap"] * sdf["volume"]

    candles = []
    # Group by bucket
    grouped = sdf.groupby(buckets, sort=True)

    for b_idx, group in grouped:
        b_start_min = int(b_idx) * interval_minutes
        b_end_min = min(b_start_min + interval_minutes, total_session_minutes)
        duration = b_end_min - b_start_min
        is_partial = (duration < interval_minutes)
        expected_bars = duration

        window_start_time = open_time + pd.Timedelta(minutes=b_start_min)
        observed_bars = len(group)

        c_open = float(group["open"].iloc[0])
        c_high = float(group["high"].max())
        c_low = float(group["low"].min())
        c_close = float(group["close"].iloc[-1])
        c_vol = int(group["volume"].sum())
        c_trade_count = int(group["trade_count"].sum()) if has_trade_count else None

        if has_vwap and c_vol > 0:
            c_vwap = float(group["v_vwap"].sum() / c_vol)
        else:
            c_vwap = float((c_high + c_low + c_close) / 3.0)

        candles.append({
            "timestamp_utc": window_start_time.tz_convert("UTC").to_pydatetime(),
            "exchange_time_ny": window_start_time.isoformat(),
            "session_date": session_date,
            "interval_minutes": interval_minutes,
            "duration_minutes": duration,
            "is_partial": is_partial,
            "open": c_open,
            "high": c_high,
            "low": c_low,
            "close": c_close,
            "volume": c_vol,
            "trade_count": c_trade_count,
            "vwap": round(c_vwap, 4),
            "expected_bars": expected_bars,
            "observed_bars": observed_bars,
            "is_complete": observed_bars == expected_bars,
        })

    return candles

def resample_symbol_bars(
    raw_bars_df: pd.DataFrame,
    symbol: str,
    output_dir_20: Path,
    output_dir_30: Path,
    output_dir_norm: Path | None = None,
) -> dict:
    """Resample raw 1-minute bars into 20-min and 30-min parquet files with high performance."""
    if raw_bars_df.empty:
        return {"bars_count_20": 0, "bars_count_30": 0}

    df = raw_bars_df.copy()
    if "t" in df.columns and "timestamp_utc" not in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["t"], utc=True)
    elif "timestamp" in df.columns and "timestamp_utc" not in df.columns:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp"], utc=True)
    else:
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)

    col_map = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "trade_count", "vw": "vwap"}
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    df["symbol"] = symbol

    df = df.drop_duplicates(subset=["timestamp_utc"]).sort_values("timestamp_utc")

    # Vectorized time conversion
    df["time_ny"] = df["timestamp_utc"].dt.tz_convert("America/New_York")
    df["session_date"] = df["time_ny"].dt.strftime("%Y-%m-%d")

    # Distinguish supported off-session dates (which are excluded from candles)
    # from input dates outside the calendar's supported range.
    cal = get_exchange_calendar()
    first_day = cal.first_session.strftime("%Y-%m-%d")
    last_day = cal.last_session.strftime("%Y-%m-%d")
    if df["session_date"].min() < first_day or df["session_date"].max() > last_day:
        raise ValueError(f"Input dates must be within supported XNYS calendar bounds {first_day}–{last_day}")
    sessions = get_session_cache(df["session_date"].min(), df["session_date"].max())
    is_session = df["session_date"].isin(sessions)
    session_df = df.loc[is_session].copy()

    # Save normalized 1Min bars atomically.
    if output_dir_norm:
        output_dir_norm.mkdir(parents=True, exist_ok=True)
        norm_path = output_dir_norm / f"{symbol}_1min_norm.parquet"
        _atomic_parquet(df, norm_path)

    candles_20_all = []
    candles_30_all = []

    for session_date, day_frame in session_df.groupby("session_date", sort=True):
        c20 = derive_candles_for_session(day_frame, session_date, interval_minutes=20)
        c30 = derive_candles_for_session(day_frame, session_date, interval_minutes=30)
        candles_20_all.extend(c20)
        candles_30_all.extend(c30)

    df_20 = pd.DataFrame(candles_20_all)
    df_30 = pd.DataFrame(candles_30_all)

    output_dir_20.mkdir(parents=True, exist_ok=True)
    output_dir_30.mkdir(parents=True, exist_ok=True)

    file_20 = output_dir_20 / f"{symbol}_20min.parquet"
    file_30 = output_dir_30 / f"{symbol}_30min.parquet"

    if not df_20.empty:
        df_20["symbol"] = symbol
        _atomic_parquet(df_20.sort_values("timestamp_utc"), file_20)
    if not df_30.empty:
        df_30["symbol"] = symbol
        _atomic_parquet(df_30.sort_values("timestamp_utc"), file_30)

    return {
        "bars_count_20": len(df_20),
        "bars_count_30": len(df_30),
        "file_20": str(file_20.as_posix()) if not df_20.empty else None,
        "file_30": str(file_30.as_posix()) if not df_30.empty else None,
    }


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a derived parquet beside its destination, then atomically replace it."""
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_parquet(tmp, index=False, compression="snappy")
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()
