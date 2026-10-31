"""Compare Alpaca consolidated SIP intraday bars against an independent comparable feed (Yahoo Finance / yfinance)."""

import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path
from datetime import date, timedelta

def compare_alpaca_vs_yfinance(symbol: str = "AAPL", test_date: str = "2024-01-05"):
    print(f"=== INDEPENDENT FEED AUDIT: ALPACA SIP vs YAHOO FINANCE ({symbol} on {test_date}) ===")

    # 1. Load Alpaca 30-min derived candles
    candles_path = Path(f"data/candles/30min/{symbol}_30min.parquet")
    if not candles_path.exists():
        print(f"Alpaca candles file {candles_path} not found.")
        return None

    alpaca_df = pd.read_parquet(candles_path)
    alpaca_day = alpaca_df[alpaca_df["session_date"] == test_date].sort_values("timestamp_utc").copy()
    if alpaca_day.empty:
        print(f"No Alpaca candles found for date {test_date}")
        return None

    print(f"Alpaca SIP candles for {test_date}: {len(alpaca_day)} candles")
    print("Alpaca Session Open:", alpaca_day["open"].iloc[0], "Close:", alpaca_day["close"].iloc[-1], "High:", alpaca_day["high"].max(), "Low:", alpaca_day["low"].min(), "Volume:", alpaca_day["volume"].sum())

    # 2. Fetch Yahoo Finance daily bar (both unadjusted and adjusted)
    ticker = yf.Ticker(symbol)
    end_date = (date.fromisoformat(test_date) + timedelta(days=1)).isoformat()
    hist_raw = ticker.history(start=test_date, end=end_date, auto_adjust=False)
    hist_adj = ticker.history(start=test_date, end=end_date, auto_adjust=True)

    diff_summary = {}
    if not hist_raw.empty:
        yf_raw_open = float(hist_raw["Open"].iloc[0])
        yf_raw_high = float(hist_raw["High"].iloc[0])
        yf_raw_low = float(hist_raw["Low"].iloc[0])
        yf_raw_close = float(hist_raw["Close"].iloc[0])
        yf_vol = int(hist_raw["Volume"].iloc[0])
        yf_adj_close = float(hist_adj["Close"].iloc[0]) if not hist_adj.empty else yf_raw_close

        alpaca_open = float(alpaca_day["open"].iloc[0])
        alpaca_high = float(alpaca_day["high"].max())
        alpaca_low = float(alpaca_day["low"].min())
        alpaca_close = float(alpaca_day["close"].iloc[-1])
        alpaca_vol = int(alpaca_day["volume"].sum())

        print("\nYahoo Finance Daily Reference:")
        print(f"YF Raw Close: {yf_raw_close:.4f}, Alpaca Close: {alpaca_close:.4f} (Diff: {abs(alpaca_close - yf_raw_close):.4f})")
        print(f"YF Adj Close: {yf_adj_close:.4f} (Split/Dividend Adjusted)")

        open_diff_pct = abs(alpaca_open - yf_raw_open) / abs(yf_raw_open) * 100 if yf_raw_open else np.nan
        close_diff_pct = abs(alpaca_close - yf_raw_close) / abs(yf_raw_close) * 100 if yf_raw_close else np.nan
        volume_diff_pct = abs(alpaca_vol - yf_vol) / abs(yf_vol) * 100 if yf_vol else np.nan
        explanation = (
            f"Observed Alpaca versus Yahoo raw close difference: {abs(alpaca_close - yf_raw_close):.4f} "
            f"({close_diff_pct:.4f}% of Yahoo raw close). "
            f"Yahoo adjusted close differs from its raw close by {abs(yf_adj_close - yf_raw_close):.4f}. "
            f"Observed volume difference: {abs(alpaca_vol - yf_vol):,} shares ({volume_diff_pct:.2f}% of Yahoo volume); "
            "this comparison does not identify the cause of any volume difference."
        )
        diff_summary = {
            "symbol": symbol,
            "test_date": test_date,
            "alpaca_open": round(alpaca_open, 4),
            "yf_raw_open": round(yf_raw_open, 4),
            "open_diff_pct": round(open_diff_pct, 4),
            "alpaca_close": round(alpaca_close, 4),
            "yf_raw_close": round(yf_raw_close, 4),
            "close_diff_pct": round(close_diff_pct, 4),
            "yf_adj_close": round(yf_adj_close, 4),
            "alpaca_volume": alpaca_vol,
            "yf_volume": yf_vol,
            "volume_diff_pct": round(volume_diff_pct, 2),
            "explanation": explanation,
        }

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    comp_df = pd.DataFrame([diff_summary])
    comp_df.to_csv(reports_dir / "feed_comparison_audit.csv", index=False)
    print(f"\nSaved independent feed comparison audit to {reports_dir / 'feed_comparison_audit.csv'}")
    return diff_summary

if __name__ == "__main__":
    compare_alpaca_vs_yfinance()
