"""Incremental daily updater for recent, fully closed trading sessions."""

from datetime import datetime, timezone, timedelta
from pathlib import Path
import shutil
import tempfile

import exchange_calendars as xcals
import pandas as pd

from market_research.acquire import AlpacaAcquisitionClient
from market_research.resample import resample_symbol_bars

PROVIDER_DELAY = timedelta(minutes=15)
OVERLAP_SESSIONS = 5


def derived_paths(feed: str = "sip", adjustment: str = "raw", data_root: Path = Path("data")) -> dict[str, Path]:
    """Use established paths for canonical SIP/raw data; isolate all other variants."""
    if feed == "sip" and adjustment == "raw":
        return {
            "20": data_root / "candles/20min",
            "30": data_root / "candles/30min",
            "norm": data_root / "normalized",
        }
    root = data_root / "candles" / f"feed={feed}" / f"adjustment={adjustment}"
    return {
        "20": root / "20min",
        "30": root / "30min",
        "norm": data_root / "normalized" / f"feed={feed}" / f"adjustment={adjustment}",
    }


def last_completed_session(now_utc: datetime, provider_delay: timedelta = PROVIDER_DELAY) -> str | None:
    """Return latest XNYS session whose scheduled close plus provider delay has passed."""
    now_utc = now_utc.astimezone(timezone.utc)
    cal = xcals.get_calendar("XNYS")
    start = (now_utc - timedelta(days=45)).date().isoformat()
    sessions = cal.sessions_in_range(start, now_utc.date().isoformat())
    for session in reversed(sessions):
        close = cal.session_close(session).to_pydatetime().astimezone(timezone.utc)
        if now_utc >= close + provider_delay:
            return session.strftime("%Y-%m-%d")
    return None


def _cleanup_stage(stage: Path, allowed_parent: Path) -> None:
    """Remove only a resolved temporary directory contained by its known output root."""
    stage_resolved = stage.resolve()
    parent_resolved = allowed_parent.resolve()
    if stage_resolved == parent_resolved or parent_resolved not in stage_resolved.parents:
        raise RuntimeError(f"Refusing to recursively remove staging directory outside output root: {stage_resolved}")
    shutil.rmtree(stage_resolved)


def _atomic_resample(combined: pd.DataFrame, symbol: str, dirs: dict[str, Path]) -> dict:
    """Stage resample products beside their destinations, then atomically replace each file."""
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    stage_root = Path(tempfile.mkdtemp(prefix=f".{symbol}_resample_", dir=dirs["20"].parent))
    stage_resolved = stage_root.resolve()
    allowed_parent = dirs["20"].parent.resolve()
    if stage_resolved == allowed_parent or allowed_parent not in stage_resolved.parents:
        raise RuntimeError(f"Refusing to clean staging directory outside output root: {stage_resolved}")
    try:
        staged = {key: stage_root / key for key in dirs}
        result = resample_symbol_bars(
            combined,
            symbol,
            output_dir_20=staged["20"],
            output_dir_30=staged["30"],
            output_dir_norm=staged["norm"],
        )
        for key, filename in (("20", f"{symbol}_20min.parquet"), ("30", f"{symbol}_30min.parquet"), ("norm", f"{symbol}_1min_norm.parquet")):
            source = staged[key] / filename
            if source.exists():
                source.replace(dirs[key] / filename)
        result["file_20"] = str((dirs["20"] / f"{symbol}_20min.parquet").as_posix()) if (dirs["20"] / f"{symbol}_20min.parquet").exists() else None
        result["file_30"] = str((dirs["30"] / f"{symbol}_30min.parquet").as_posix()) if (dirs["30"] / f"{symbol}_30min.parquet").exists() else None
        result["file_norm"] = str((dirs["norm"] / f"{symbol}_1min_norm.parquet").as_posix()) if (dirs["norm"] / f"{symbol}_1min_norm.parquet").exists() else None
        return result
    finally:
        _cleanup_stage(stage_root, dirs["20"].parent)


def run_daily_update(
    symbols: list[str] | None = None,
    feed: str = "sip",
    adjustment: str = "raw",
    now: datetime | None = None,
    provider_delay: timedelta = PROVIDER_DELAY,
):
    print("=== STARTING INCREMENTAL DAILY UPDATE ===")
    now_utc = now or datetime.now(timezone.utc)
    last_session = last_completed_session(now_utc, provider_delay)
    if last_session is None:
        print("No recent completed sessions found.")
        return
    print(f"Targeting updates through completed session: {last_session}")

    dirs = derived_paths(feed, adjustment)
    if not symbols:
        symbols = [p.name.split("_")[0] for p in dirs["30"].glob("*_30min.parquet")]
    if not symbols:
        symbols = ["AAPL", "META", "MMM"]

    cal = xcals.get_calendar("XNYS")
    with AlpacaAcquisitionClient(feed=feed) as client:
        for sym in symbols:
            c30_file = dirs["30"] / f"{sym}_30min.parquet"
            norm_file = dirs["norm"] / f"{sym}_1min_norm.parquet"
            if c30_file.exists() and not norm_file.exists():
                raise FileNotFoundError(
                    f"Cannot safely update {sym}: candles exist at {c30_file}, but normalized history is missing at {norm_file}"
                )

            old_norm = pd.read_parquet(norm_file) if norm_file.exists() else None
            if old_norm is not None and not old_norm.empty:
                old_norm = old_norm.copy()
                old_norm["timestamp_utc"] = pd.to_datetime(old_norm["timestamp_utc"], utc=True)
                observed = old_norm["timestamp_utc"]
                latest_date = observed.max().tz_convert("America/New_York").date()
                sessions = cal.sessions_in_range((latest_date - timedelta(days=20)).isoformat(), latest_date.isoformat())
                recent = [s.strftime("%Y-%m-%d") for s in sessions[-OVERLAP_SESSIONS:]]
                start_date = recent[0] if recent else latest_date.isoformat()
            else:
                start_date = "2024-01-01"
            if start_date > last_session:
                print(f"  {sym} is already up to date.")
                continue

            print(f"  Refetching {sym} from {start_date} through {last_session} (overlap included)...")
            new_bars = client.fetch_bars_chunk(
                sym, f"{start_date}T00:00:00Z", f"{last_session}T23:59:59Z",
                timeframe="1Min", adjustment=adjustment,
            )
            if not new_bars:
                print(f"  No bars returned for {sym} in range.")
                continue
            new_df = pd.DataFrame(new_bars).rename(columns={"t": "timestamp_utc", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "trade_count", "vw": "vwap"})
            new_df["timestamp_utc"] = pd.to_datetime(new_df["timestamp_utc"], utc=True)
            new_df["symbol"] = sym
            new_df["feed"] = feed
            new_df["adjustment"] = adjustment
            combined = pd.concat([old_norm, new_df], ignore_index=True) if old_norm is not None else new_df
            combined = combined.drop_duplicates(subset=["timestamp_utc"], keep="last").sort_values("timestamp_utc")
            result = _atomic_resample(combined, sym, dirs)
            print(f"  Updated {sym}: {result['bars_count_20']:,} 20min, {result['bars_count_30']:,} 30min candles.")

    print("\n=== INCREMENTAL UPDATE FINISHED ===")


if __name__ == "__main__":
    run_daily_update()
