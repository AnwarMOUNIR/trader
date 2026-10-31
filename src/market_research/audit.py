"""Audit and validation engine for raw, normalized, and derived candle datasets."""

from pathlib import Path
import pandas as pd
import numpy as np
from market_research.resample import get_session_cache

def audit_candle_file(file_path: Path, expected_interval: int) -> dict:
    if expected_interval not in (20, 30):
        raise ValueError("Expected interval must be 20 or 30 minutes")
    df = pd.read_parquet(file_path)
    total_bars = len(df)
    if total_bars == 0:
        return {"file": file_path.name, "total_bars": 0, "status": "EMPTY"}

    # 1. OHLC integrity and finite numeric fields. Comparisons alone miss NaN.
    numeric_cols = [c for c in ("open", "high", "low", "close", "volume", "vwap", "trade_count") if c in df]
    numeric = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype=float)).all(axis=1)
    prices = df[[c for c in ("open", "high", "low", "close") if c in df]]
    invalid_hl = (df["high"] < df["low"]).sum()
    invalid_ho = (df["high"] < df["open"]).sum()
    invalid_hc = (df["high"] < df["close"]).sum()
    invalid_lo = (df["low"] > df["open"]).sum()
    invalid_lc = (df["low"] > df["close"]).sum()
    invalid_vol = (df["volume"] < 0).sum()
    invalid_price = ((prices <= 0).any(axis=1)).sum()
    nonfinite_count = int((~finite).sum())
    ohlc_issues = invalid_hl + invalid_ho + invalid_hc + invalid_lo + invalid_lc + invalid_vol + invalid_price + nonfinite_count

    # 2. Monotonicity and duplicates
    df["dt"] = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
    is_monotonic = df["dt"].is_monotonic_increasing and not df["dt"].duplicated().any()
    duplicate_count = df.duplicated(subset=["timestamp_utc"]).sum()

    # 3. Session and partial candle validation
    if expected_interval == 20:
        full_candles = (df["duration_minutes"] == 20).sum()
        partial_candles = (df["duration_minutes"] == 10).sum()
        partial_flag_matches = ((df["duration_minutes"] == 10) == df["is_partial"]).fillna(False).all()
    elif expected_interval == 30:
        full_candles = (df["duration_minutes"] == 30).sum()
        partial_candles = df["is_partial"].eq(True).sum()
        partial_flag_matches = ((df["duration_minutes"] < 30) == df["is_partial"]).fillna(False).all()
    else:
        full_candles = total_bars
        partial_candles = 0
        partial_flag_matches = True
    completeness_violations = 0
    if "expected_bars" in df:
        completeness_violations += int((pd.to_numeric(df["expected_bars"], errors="coerce") != pd.to_numeric(df["duration_minutes"], errors="coerce")).fillna(True).sum())
    if {"observed_bars", "expected_bars", "is_complete"}.issubset(df.columns):
        observed = pd.to_numeric(df["observed_bars"], errors="coerce")
        expected = pd.to_numeric(df["expected_bars"], errors="coerce")
        expected_complete = observed == expected
        flag_valid = df["is_complete"].isin([True, False]) & (expected_complete == df["is_complete"])
        counts_valid = (np.isfinite(observed) & np.isfinite(expected)
                        & (observed > 0) & (observed <= expected)
                        & (observed % 1 == 0) & (expected % 1 == 0))
        completeness_violations += int((~flag_valid.fillna(False)).sum())
        completeness_violations += int((~counts_valid.fillna(False)).sum())

    # Session labels, opens, and candle durations must agree with XNYS.
    session_issue_count = total_bars
    if "session_date" in df and "duration_minutes" in df:
        dates = df["session_date"].astype("string")
        try:
            calendar_sessions = get_session_cache()
            sessions = {day: calendar_sessions[day] for day in dates.dropna().unique()
                        if day in calendar_sessions}
            # Map once per session, then validate every row in UTC. Offsets
            # are relative to the actual exchange open, not the clock hour;
            # this preserves 09:30 anchoring, DST and early closes.
            opens = pd.to_datetime(dates.map({day: op for day, (op, _) in sessions.items()}), utc=True)
            closes = pd.to_datetime(dates.map({day: cl for day, (_, cl) in sessions.items()}), utc=True)
            elapsed = df["dt"] - opens
            remaining = (closes - df["dt"]).dt.total_seconds() / 60
            duration = pd.to_numeric(df["duration_minutes"], errors="coerce")
            valid = (
                opens.notna() & closes.notna() & df["dt"].notna()
                & (df["dt"] >= opens) & (df["dt"] < closes)
                & (elapsed % pd.Timedelta(minutes=expected_interval) == pd.Timedelta(0))
                & (duration == remaining.clip(upper=expected_interval))
            )
            session_issue_count = int((~valid.fillna(False)).sum())
        except (ValueError, KeyError, TypeError):
            session_issue_count = total_bars
    session_valid = session_issue_count == 0

    # 4. Earliest and latest timestamp
    min_date = df["dt"].min().isoformat()
    max_date = df["dt"].max().isoformat()
    unique_sessions = df["session_date"].nunique() if "session_date" in df else None

    status = "PASS" if ohlc_issues == 0 and is_monotonic and duplicate_count == 0 and partial_flag_matches and session_valid and completeness_violations == 0 else "WARN"

    return {
        "file": file_path.name,
        "symbol": df["symbol"].iloc[0] if "symbol" in df else file_path.name.split("_")[0],
        "interval_minutes": expected_interval,
        "total_candles": total_bars,
        "unique_sessions": unique_sessions,
        "earliest_timestamp": min_date,
        "latest_timestamp": max_date,
        "ohlc_violations": int(ohlc_issues),
        "nonfinite_values": nonfinite_count,
        "session_violations": int(session_issue_count),
        "completeness_violations": completeness_violations,
        "duplicate_timestamps": int(duplicate_count),
        "is_strictly_monotonic": bool(is_monotonic),
        "full_candles": int(full_candles),
        "partial_candles": int(partial_candles),
        "status": status,
    }

def run_all_audits() -> pd.DataFrame:
    records = []
    dir_20 = Path("data/candles/20min")
    dir_30 = Path("data/candles/30min")

    if dir_20.exists():
        for p in sorted(dir_20.glob("*.parquet")):
            records.append(audit_candle_file(p, expected_interval=20))

    if dir_30.exists():
        for p in sorted(dir_30.glob("*.parquet")):
            records.append(audit_candle_file(p, expected_interval=30))

    audit_df = pd.DataFrame(records)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(reports_dir / "candle_audit_summary.csv", index=False)
    return audit_df

if __name__ == "__main__":
    df = run_all_audits()
    print("=== CANDLE AUDIT REPORT ===")
    print(df.to_string(index=False) if not df.empty else "No candles found yet.")
