"""Run pilot acquisition and candle derivation for representative S&P 500 securities."""

import sys
from pathlib import Path
import pandas as pd
from datetime import datetime

from market_research.acquire import AlpacaAcquisitionClient, acquire_symbol_year
from market_research.resample import resample_symbol_bars

PILOT_SYMBOLS = [
    # (symbol, [years], category)
    ("AAPL", [2016, 2023, 2024], "Current Member / Earliest Alpaca Depth"),
    ("META", [2022, 2023, 2024], "Current Member / Post-rename Meta"),
    ("FB", [2021], "Renamed Ticker (Historical Facebook)"),
    ("TWTR", [2021, 2022], "Acquired / Delisted (Twitter)"),
    ("ATVI", [2022, 2023], "Acquired / Delisted (Activision Blizzard)"),
    ("MMM", [2016, 2023, 2024], "Original 1957 Member (3M)"),
    ("FRC", [2022, 2023], "Bankrupt / Failed (First Republic Bank)"),
    ("SIVB", [2022, 2023], "Bankrupt / Failed (Silicon Valley Bank)"),
]

def run_pilot():
    print("=== STARTING MARKET RESEARCH PILOT ACQUISITION ===")
    out_base_raw = Path("data/raw")
    out_dir_20 = Path("data/candles/20min")
    out_dir_30 = Path("data/candles/30min")
    out_dir_norm = Path("data/normalized")

    pilot_results = []

    with AlpacaAcquisitionClient(feed="sip") as client:
        for sym, years, cat in PILOT_SYMBOLS:
            print(f"\n--- Processing {sym} ({cat}) ---")
            sym_raw_dfs = []
            total_raw_bars = 0

            for y in years:
                print(f"  Acquiring {sym} year {y}...")
                res = acquire_symbol_year(client, sym, y, output_base=out_base_raw)
                print(f"    Status: {res['status']}, Bars: {res['bars_count']:,}")
                total_raw_bars += res['bars_count']
                if res['bars_count'] > 0 and res['file_path']:
                    df_chunk = pd.read_parquet(res['file_path'])
                    sym_raw_dfs.append(df_chunk)

            if sym_raw_dfs:
                combined_raw = pd.concat(sym_raw_dfs, ignore_index=True)
                print(f"  Deriving 20-min and 30-min candles for {sym} ({len(combined_raw):,} total raw 1Min bars)...")
                resample_res = resample_symbol_bars(
                    combined_raw,
                    sym,
                    output_dir_20=out_dir_20,
                    output_dir_30=out_dir_30,
                    output_dir_norm=out_dir_norm,
                )
                print(f"    Derived 20-min candles: {resample_res['bars_count_20']:,}")
                print(f"    Derived 30-min candles: {resample_res['bars_count_30']:,}")

                pilot_results.append({
                    "symbol": sym,
                    "category": cat,
                    "years": years,
                    "raw_1min_bars": total_raw_bars,
                    "candles_20min": resample_res["bars_count_20"],
                    "candles_30min": resample_res["bars_count_30"],
                    "status": "ACQUIRED_AND_DERIVED",
                })
            else:
                pilot_results.append({
                    "symbol": sym,
                    "category": cat,
                    "years": years,
                    "raw_1min_bars": 0,
                    "candles_20min": 0,
                    "candles_30min": 0,
                    "status": "NO_BARS_FOR_RANGE",
                })

    print("\n=== PILOT SUMMARY ===")
    summary_df = pd.DataFrame(pilot_results)
    print(summary_df.to_string(index=False))

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(reports_dir / "pilot_summary.csv", index=False)
    print(f"\nSaved pilot summary to {reports_dir / 'pilot_summary.csv'}")

if __name__ == "__main__":
    run_pilot()
