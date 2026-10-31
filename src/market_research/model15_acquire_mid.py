"""Acquire missing 2019-2021 bars for Model 15's existing development symbols."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from market_research.acquire import AlpacaAcquisitionClient, acquire_symbol_year
from market_research.model14_experiment import REPORT as MODEL14_REPORT
from market_research.resample import get_session_cache

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "data/challenges/model15/train/mid"
MANIFEST = ROOT / "reports/model15/mid_data_manifest.json"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _resample_30min(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Vectorized equivalent of the project session-anchored 30-minute bars."""
    df = raw.copy()
    df["timestamp_utc"] = pd.to_datetime(df.timestamp_utc, utc=True)
    df = df.drop_duplicates("timestamp_utc").sort_values("timestamp_utc")
    df["time_ny"] = df.timestamp_utc.dt.tz_convert("America/New_York")
    df["session_date"] = df.time_ny.dt.strftime("%Y-%m-%d")
    sessions = get_session_cache(df.session_date.min(), df.session_date.max())
    unmapped = ~df.session_date.isin(sessions)
    if unmapped.any():
        invalid = sorted(df.loc[unmapped, "session_date"].unique())
        raise ValueError(f"Input contains non-session or unsupported dates: {invalid[:5]}")
    opens = {key: value[0] for key, value in sessions.items()}
    closes = {key: value[1] for key, value in sessions.items()}
    df["open_time"] = df.session_date.map(opens)
    df["close_time"] = df.session_date.map(closes)
    df = df.loc[(df.time_ny >= df.open_time) & (df.time_ny < df.close_time)].copy()
    elapsed = ((df.time_ny - df.open_time).dt.total_seconds() // 60).astype(int)
    df["bucket"] = elapsed // 30
    df["dollar_vwap"] = df.vwap * df.volume
    agg = df.groupby(["session_date", "bucket"], sort=True, observed=True).agg(
        open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), trade_count=("trade_count", "sum"), observed_bars=("timestamp_utc", "size"),
        dollar_vwap=("dollar_vwap", "sum")).reset_index()
    agg["open_time"] = agg.session_date.map(opens)
    agg["close_time"] = agg.session_date.map(closes)
    bucket_start = agg.open_time + pd.to_timedelta(agg.bucket * 30, unit="m")
    duration = np.minimum(30, ((agg.close_time - bucket_start).dt.total_seconds() // 60).astype(int))
    agg["timestamp_utc"] = bucket_start.dt.tz_convert("UTC")
    agg["exchange_time_ny"] = bucket_start.map(lambda x: x.isoformat())
    agg["interval_minutes"] = 30
    agg["duration_minutes"] = duration.astype(int)
    agg["is_partial"] = agg.duration_minutes < 30
    agg["expected_bars"] = agg.duration_minutes
    agg["is_complete"] = agg.observed_bars == agg.expected_bars
    weighted_vwap = agg.dollar_vwap / agg.volume.replace(0, np.nan)
    agg["vwap"] = weighted_vwap.fillna((agg.high + agg.low + agg.close) / 3.0).round(4)
    agg["symbol"] = symbol
    agg["observed_bars"] = agg.observed_bars.astype(int)
    agg["volume"] = agg.volume.astype("int64")
    agg["trade_count"] = agg.trade_count.fillna(0).astype("int64")
    return agg[["timestamp_utc", "exchange_time_ny", "session_date", "interval_minutes", "duration_minutes",
                "is_partial", "open", "high", "low", "close", "volume", "trade_count", "vwap",
                "expected_bars", "observed_bars", "is_complete", "symbol"]].sort_values("timestamp_utc")


def _verify_resampler() -> None:
    paths = [ROOT / f"data/raw/feed=sip/timeframe=1Min/symbol=AAPL/AAPL_{year}_1min.parquet" for year in (2016, 2017, 2018)]
    if not all(path.exists() for path in paths):
        return
    raw = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    fresh = _resample_30min(raw, "AAPL")
    old = pd.read_parquet(ROOT / "data/challenges/model14/train/pre/candles/30min/AAPL_30min.parquet").sort_values("timestamp_utc")
    cols = ["timestamp_utc", "open", "high", "low", "close", "volume", "trade_count", "vwap", "expected_bars", "observed_bars", "is_complete"]
    fresh, old = fresh[cols].reset_index(drop=True), old[cols].reset_index(drop=True)
    fresh_times = pd.to_datetime(fresh.timestamp_utc, utc=True).reset_index(drop=True)
    old_times = pd.to_datetime(old.timestamp_utc, utc=True).reset_index(drop=True)
    if len(fresh) != len(old) or not fresh_times.astype(str).equals(old_times.astype(str)):
        raise RuntimeError("Vectorized resampler timestamps disagree with audited AAPL bars")
    numeric = ["open", "high", "low", "close", "volume", "trade_count", "vwap", "expected_bars", "observed_bars"]
    if not np.allclose(fresh[numeric].to_numpy(float), old[numeric].to_numpy(float), rtol=0, atol=1e-7, equal_nan=True):
        raise RuntimeError("Vectorized resampler values disagree with audited AAPL bars")
    if not fresh.is_complete.equals(old.is_complete):
        raise RuntimeError("Vectorized resampler completeness flags disagree with audited AAPL bars")
    print("Verified vectorized 30-minute resampling against audited AAPL 2016-2018 bars.", flush=True)


def run() -> dict:
    protocol = json.loads(MODEL14_REPORT.joinpath("protocol.json").read_text(encoding="utf-8"))
    symbols = sorted(set(protocol["training_symbols"]) | {"SPY"})
    years = (2019, 2020, 2021)
    results = []
    with AlpacaAcquisitionClient(feed="sip") as client:
        jobs = [(symbol, year) for symbol in symbols for year in years]
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(acquire_symbol_year, client, symbol, year, output_base=ROOT / "data/raw"):
                       (symbol, year) for symbol, year in jobs}
            for future in as_completed(futures):
                symbol, year = futures[future]
                result = future.result()
                results.append({"symbol": symbol, "year": year, **result})
                print(f"{symbol} {year}: {result['status']} ({result['bars_count']:,} bars)", flush=True)
    _verify_resampler()
    result_by_key = {(item["symbol"], item["year"]): item for item in results}
    file_records = []
    for symbol in symbols:
        parts = []
        for year in years:
            result = result_by_key.get((symbol, year), {})
            path = Path(result["file_path"]) if result.get("file_path") else None
            if path and path.exists():
                parts.append(pd.read_parquet(path))
        if not parts:
            continue
        raw = pd.concat(parts, ignore_index=True)
        bars = _resample_30min(raw, symbol)
        path = OUTPUT / "candles/30min" / f"{symbol}_30min.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        bars.to_parquet(path, index=False, compression="snappy")
        file_records.append({"symbol": symbol, "rows_1min": int(len(raw)), "rows_30min": int(len(bars)),
                            "path_30min": str(path.relative_to(ROOT)), "sha256_30min": _sha(path)})
        print(f"Resampled {symbol}: {len(raw):,} one-minute to {len(bars):,} 30-minute bars", flush=True)
        del raw, bars, parts
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": "Alpaca SIP raw one-minute bars",
                "years": list(years), "symbols": symbols, "acquisition_results": results, "resampled_files": file_records}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"symbols_requested": len(symbols), "files_resampled": len(file_records), "manifest": str(MANIFEST.relative_to(ROOT))}, indent=2))
    return manifest


if __name__ == "__main__":
    run()
