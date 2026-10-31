"""Shared calendar splits and frozen-input checks for new research runs."""
from pathlib import Path
import hashlib
from datetime import datetime, timezone
from uuid import uuid4

import pandas as pd


def new_run_dir(base: Path) -> Path:
    """Keep corrected experiments separate from all historical frozen outputs."""
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid4().hex[:8]
    path = base / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def chronological_split(data, fractions=(0.6, 0.8), time_col="timestamp_utc"):
    """Shared session cuts across issuers; purge labels touching the next block."""
    times = pd.to_datetime(data[time_col], utc=True)
    dates = times.dt.tz_convert("America/New_York").dt.normalize()
    unique = dates.drop_duplicates().sort_values().tolist()
    indices = [int(len(unique) * f) for f in fractions]
    if not unique or any(i <= 0 or i >= len(unique) for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("Insufficient sessions for chronological split")
    cuts = [unique[i] for i in indices]
    # Legacy intraday frames without an explicit label endpoint use a whole
    # session embargo, sufficient only for their documented same-session target.
    ends = pd.to_datetime(data.label_end_utc, utc=True) if "label_end_utc" in data else times + pd.Timedelta(days=1)
    parts = []
    for i in range(len(cuts) + 1):
        mask = pd.Series(True, index=data.index)
        if i:
            mask &= dates >= cuts[i - 1]
        if i < len(cuts):
            mask &= (dates < cuts[i]) & (ends < cuts[i])
        part = data.loc[mask].copy()
        if part.empty:
            raise ValueError("Empty chronological block after label purging")
        parts.append(part)
    return tuple(parts)


def verify_manifest(root: Path, manifest: dict, required_symbols, candle_dir: Path | None = None):
    """Reject missing, modified or out-of-project candle inputs before scoring."""
    records = {r["symbol"]: r for r in manifest.get("resampled", [])}
    for symbol in set(required_symbols) | {"SPY"}:
        if symbol not in records:
            raise RuntimeError(f"Missing manifest entry: {symbol}")
        record = records[symbol]
        path = (root / record["path"]).resolve()
        if candle_dir is not None and path != (candle_dir / f"{symbol}_30min.parquet").resolve():
            raise RuntimeError(f"Manifest does not describe scoring input: {symbol}")
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise RuntimeError(f"Invalid manifest file: {symbol}")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != record.get("sha256"):
            raise RuntimeError(f"Candle checksum mismatch: {symbol}")


def require_unscored(protocol: dict):
    status = str(protocol.get("status", "")).lower()
    if status not in {"registered", "registered_unscored", "pending", "lockbox_registered"} or protocol.get("scored_utc"):
        raise RuntimeError(f"Lockbox is not awaiting first scoring: {status}")
