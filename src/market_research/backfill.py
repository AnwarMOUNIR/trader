"""Resumable bulk backfill CLI for historical S&P 500 members."""

import argparse
from pathlib import Path
import tempfile
import pandas as pd

from market_research.acquire import AlpacaAcquisitionClient, acquire_symbol_year
from market_research.daily_update import _cleanup_stage, derived_paths
from market_research.resample import resample_symbol_bars
from market_research.db import get_db, init_db

def run_backfill(
    symbols: list[str] | None = None,
    start_year: int = 2016,
    end_year: int = 2024,
    feed: str = "sip",
    limit: int | None = None,
):
    init_db()
    sec_master_path = Path("data/metadata/security_master.parquet")
    if not sec_master_path.exists():
        raise FileNotFoundError("Security master parquet not found. Run build_security_master.py first.")

    master_df = pd.read_parquet(sec_master_path)

    if symbols:
        target_symbols = [s.strip().upper() for s in symbols]
    else:
        # Default to current members ordered by ticker
        target_symbols = list(master_df[master_df["is_current_member"]]["primary_ticker"].unique())

    if limit:
        target_symbols = target_symbols[:limit]

    print(f"=== STARTING BULK BACKFILL ({len(target_symbols)} symbols, {start_year}-{end_year}, feed={feed}) ===")

    out_base_raw = Path("data/raw")
    derived = derived_paths(feed=feed, adjustment="raw")
    out_dir_20, out_dir_30, out_dir_norm = derived["20"], derived["30"], derived["norm"]

    with AlpacaAcquisitionClient(feed=feed) as client:
        for idx, sym in enumerate(target_symbols, 1):
            print(f"\n[{idx}/{len(target_symbols)}] Backfilling {sym}...")
            sym_raw_dfs = []

            for y in range(start_year, end_year + 1):
                try:
                    res = acquire_symbol_year(client, sym, y, output_base=out_base_raw)
                    print(f"  Year {y}: {res['status']} ({res['bars_count']:,} bars)")
                    if res["bars_count"] > 0 and res["file_path"]:
                        df_chunk = pd.read_parquet(res["file_path"])
                        sym_raw_dfs.append(df_chunk)
                except Exception as e:
                    print(f"  Error acquiring {sym} for {y}: {e}")

            if sym_raw_dfs:
                combined_raw = pd.concat(sym_raw_dfs, ignore_index=True)
                norm_path = out_dir_norm / f"{sym}_1min_norm.parquet"
                candle_path = out_dir_30 / f"{sym}_30min.parquet"
                if candle_path.exists() and not norm_path.exists():
                    raise FileNotFoundError(
                        f"Cannot safely rebuild {sym}: candles exist at {candle_path}, but normalized history is missing at {norm_path}"
                    )
                if norm_path.exists():
                    existing = pd.read_parquet(norm_path)
                    existing["timestamp_utc"] = pd.to_datetime(existing["timestamp_utc"], utc=True)
                    combined_raw = pd.concat([existing, combined_raw], ignore_index=True)
                combined_raw["timestamp_utc"] = pd.to_datetime(combined_raw["timestamp_utc"], utc=True)
                combined_raw = combined_raw.drop_duplicates(subset=["timestamp_utc"], keep="last").sort_values("timestamp_utc")
                print(f"  Resampling {sym} ({len(combined_raw):,} raw bars)...")
                for output_dir in (out_dir_20, out_dir_30, out_dir_norm):
                    output_dir.mkdir(parents=True, exist_ok=True)
                stage = Path(tempfile.mkdtemp(prefix=f".{sym}_backfill_", dir=out_dir_20.parent))
                try:
                    stage_20, stage_30, stage_norm = stage / "20", stage / "30", stage / "norm"
                    resample_res = resample_symbol_bars(combined_raw, sym, stage_20, stage_30, stage_norm)
                    for source, target in (
                        (stage_20 / f"{sym}_20min.parquet", out_dir_20 / f"{sym}_20min.parquet"),
                        (stage_30 / f"{sym}_30min.parquet", out_dir_30 / f"{sym}_30min.parquet"),
                        (stage_norm / f"{sym}_1min_norm.parquet", out_dir_norm / f"{sym}_1min_norm.parquet"),
                    ):
                        if source.exists():
                            source.replace(target)
                    resample_res["file_20"] = str((out_dir_20 / f"{sym}_20min.parquet").as_posix()) if (out_dir_20 / f"{sym}_20min.parquet").exists() else None
                    resample_res["file_30"] = str((out_dir_30 / f"{sym}_30min.parquet").as_posix()) if (out_dir_30 / f"{sym}_30min.parquet").exists() else None
                    resample_res["file_norm"] = str((out_dir_norm / f"{sym}_1min_norm.parquet").as_posix()) if (out_dir_norm / f"{sym}_1min_norm.parquet").exists() else None
                finally:
                    _cleanup_stage(stage, out_dir_20.parent)
                print(f"  Derived: {resample_res['bars_count_20']:,} 20-min candles, {resample_res['bars_count_30']:,} 30-min candles.")

    print("\n=== BULK BACKFILL BATCH FINISHED ===")

def main():
    parser = argparse.ArgumentParser(description="Market Research S&P 500 Intraday Bulk Backfill")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of symbols (e.g. AAPL,MSFT,NVDA)")
    parser.add_argument("--start-year", type=int, default=2016, help="Earliest year (default 2016)")
    parser.add_argument("--end-year", type=int, default=2024, help="Latest year (default 2024)")
    parser.add_argument("--feed", type=str, default="sip", help="Feed name (default sip)")
    parser.add_argument("--limit", type=int, help="Limit number of symbols to process")
    args = parser.parse_args()

    sym_list = args.symbols.split(",") if args.symbols else None
    run_backfill(symbols=sym_list, start_year=args.start_year, end_year=args.end_year, feed=args.feed, limit=args.limit)

if __name__ == "__main__":
    main()
