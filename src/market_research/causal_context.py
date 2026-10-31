"""Acquire small, separate daily context caches and inspect local event returns."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import argparse
import pandas as pd
import numpy as np
import exchange_calendars as xcals

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/causal"
CACHE = ROOT / "data/context/causal"


def build_panel():
    cal = xcals.get_calendar("XNYS", start="2015-01-01", end="2025-01-01")
    dates = cal.sessions.strftime("%Y-%m-%d")
    schedule = pd.DataFrame({"session_date": dates, "calendar_index": np.arange(len(dates)),
                            "session_open": cal.opens.to_numpy(), "session_close": cal.closes.to_numpy()})
    schedule.session_open = pd.to_datetime(schedule.session_open, utc=True)
    schedule.session_close = pd.to_datetime(schedule.session_close, utc=True)
    parts = []
    for p in sorted((ROOT / "data/normalized").glob("*.parquet")):
        df = pd.read_parquet(p, columns=["symbol", "timestamp_utc", "open", "high", "low", "close", "volume"])
        df.timestamp_utc = pd.to_datetime(df.timestamp_utc, utc=True)
        df["session_date"] = df.timestamp_utc.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        df = df.merge(schedule, on="session_date", how="left", validate="many_to_one")
        df = df.loc[df.timestamp_utc.ge(df.session_open) & df.timestamp_utc.lt(df.session_close)].sort_values("timestamp_utc")
        grouped = df.groupby("session_date", sort=True).agg(symbol=("symbol", "first"), open=("open", "first"),
            high=("high", "max"), low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
            first_bar=("timestamp_utc", "min"), last_bar=("timestamp_utc", "max"), observed_minutes=("timestamp_utc", "nunique"))
        parts.append(grouped.reset_index())
    panel = pd.concat(parts, ignore_index=True).rename(columns={"symbol": "source_symbol"})
    # Explicitly verified same Class A CUSIP; retain original ticker and do not change source files.
    panel["symbol"] = panel.source_symbol.replace({"FB": "META"})
    panel = panel.merge(schedule, on="session_date", validate="many_to_one").sort_values(["symbol", "calendar_index"])
    assert not panel.duplicated(["symbol", "session_date"]).any()
    panel["endpoint_complete"] = panel.first_bar.eq(panel.session_open) & panel.last_bar.eq(panel.session_close-pd.Timedelta(minutes=1))
    panel["minute_coverage_pct"] = panel.observed_minutes / ((panel.session_close-panel.session_open).dt.total_seconds()/60)*100
    grouped = panel.groupby("symbol", sort=False)
    panel["previous_close"] = grouped.close.shift()
    valid = panel.calendar_index.sub(grouped.calendar_index.shift()).eq(1)
    valid &= panel.last_bar.eq(panel.session_close-pd.Timedelta(minutes=1))
    valid &= grouped.last_bar.shift().eq(grouped.session_close.shift()-pd.Timedelta(minutes=1))
    panel["close_return_pct"] = ((panel.close/panel.previous_close-1)*100).where(valid)
    panel["overnight_return_pct"] = ((panel.open/panel.previous_close-1)*100).where(valid & panel.first_bar.eq(panel.session_open))
    panel["intraday_return_pct"] = (panel.close/panel.open-1)*100
    return panel


def fetch_context():
    import yfinance as yf
    CACHE.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(CACHE / "yfinance_cache"))
    register = []
    for symbol in ["SPY", "QQQ", "KRE", "XLI", "SOLV"]:
        path = CACHE / f"{symbol}_daily.parquet"
        item = {"symbol": symbol, "provider": "Yahoo Finance via yfinance", "interval": "1d",
                "request_start": "2020-01-01", "request_end_exclusive": "2025-01-01",
                "url": f"https://finance.yahoo.com/quote/{symbol}/history/",
                "retrieved_utc": datetime.now(timezone.utc).isoformat(), "adjustment": "auto_adjust=False; raw OHLC and Adj Close retained"}
        try:
            if not path.exists():
                df = yf.Ticker(symbol).history(start="2020-01-01", end="2025-01-01", auto_adjust=False, actions=True, raise_errors=True)
                if df.empty:
                    raise ValueError("Provider returned no data")
                df = df.reset_index()
                df["session_date"] = df.Date.dt.strftime("%Y-%m-%d")
                df["symbol"] = symbol
                df.to_parquet(path, index=False)
            df = pd.read_parquet(path)
            item.update(status="ok", rows=len(df), first_date=df.session_date.min(), last_date=df.session_date.max(),
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(), path=str(path.relative_to(ROOT)),
                        cache_mtime_utc=datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat())
        except Exception as exc:
            item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        register.append(item)
        print(symbol, item["status"], item.get("rows", item.get("error")))
    manifest_path = CACHE / "provenance.json"
    if manifest_path.exists():
        old = {x["symbol"]: x for x in json.loads(manifest_path.read_text())}
        for item in register:
            previous = old.get(item["symbol"], {})
            if item.get("sha256") == previous.get("sha256"):
                item["retrieved_utc"] = previous["retrieved_utc"]
    manifest_path.write_text(json.dumps(register, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.fetch:
        fetch_context()
    panel = build_panel()
    panel.to_parquet(OUT / "daily_panel.parquet", index=False)
    print(panel.loc[panel.close_return_pct.abs().groupby(panel.symbol).rank(ascending=False).le(4),
                    ["symbol", "session_date", "close_return_pct", "overnight_return_pct", "intraday_return_pct", "minute_coverage_pct"]].to_string(index=False))
    print("Panel rows:", len(panel))


if __name__ == "__main__":
    main()
