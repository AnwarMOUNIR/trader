"""
Live Market Whale Momentum Scanner.
Scans current market data across US & BVC universes and outputs exact Buy/Hold/Sell
recommendations with Entry Prices, Target Allocations, and Stop-Losses for verification.
"""

import os
import json
from datetime import datetime
import pandas as pd
import yfinance as yf

from src.signals.whale_detector import WhaleDetector
from src.portfolio.allocator import PortfolioAllocator
from src.data.loader import UniverseConfig, AssetMarket


def main():
    print("=" * 80)
    print("      REAL-TIME WHALE-TRACKING MARKET SCANNER & SIGNAL REPORT")
    print(f"      Execution Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    symbols = [
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA",
        "AVGO", "AMD", "PLTR", "SMCI", "NFLX", "CRM", "ORCL",
        "QCOM", "PANW", "NOW", "SNOW", "CRWD", "UBER"
    ]

    detector = WhaleDetector(
        volume_window=30,
        z_score_threshold=1.5,
        rvol_threshold=1.5,
        min_momentum=0.08
    )

    print(f"\n[1/3] Ingesting latest market candles for {len(symbols)} US Momentum Leaders...")
    data_dict = {}
    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(period="1y")
            if not hist.empty and len(hist) > 40:
                hist.index = pd.to_datetime(hist.index).date
                data_dict[sym] = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
        except Exception as e:
            print(f"Warning: could not fetch {sym}: {e}")

    # Fetch SPY benchmark
    spy_df = yf.Ticker("SPY").history(period="1y")[["Open", "High", "Low", "Close", "Volume"]]
    spy_df.index = pd.to_datetime(spy_df.index).date
    spy_close = spy_df["Close"].iloc[-1]
    spy_sma200 = spy_df["Close"].rolling(200, min_periods=40).mean().iloc[-1]
    is_macro_bull = spy_close >= spy_sma200 * 0.98

    print(f"[2/3] Macro Regime Check: SPY=${spy_close:.2f} vs SMA200=${spy_sma200:.2f} -> "
          f"{'BULLISH REGIME (Active Longs Allowed)' if is_macro_bull else 'BEARISH REGIME (Cash Defense)'}")

    signals_dict = {}
    asset_summaries = []

    for sym, df in data_dict.items():
        sig_df = detector.process_asset(df, market="US")
        signals_dict[sym] = sig_df
        last_row = sig_df.iloc[-1]

        asset_summaries.append({
            "Symbol": sym,
            "Price": round(float(last_row["Close"]), 2),
            "Vol_ZScore": round(float(last_row.get("Vol_ZScore", 0.0)), 2),
            "RVOL": round(float(last_row.get("RVOL", 1.0)), 2),
            "Mom_12M": f"{float(last_row.get('Mom_12M', 0.0))*100:+.1f}%",
            "52W_High_Ratio": f"{float(last_row.get('Ratio_52W', 1.0))*100:.1f}%",
            "Whale_Absorption": bool(last_row.get("Wyckoff_Absorption", 0)),
            "In_Whale_Regime": bool(last_row.get("In_Whale_Position", 0)),
            "Alpha_Score": round(float(last_row.get("Composite_Alpha_Score", 0.0)), 2)
        })

    allocator = PortfolioAllocator(
        target_annual_vol=0.32,
        max_single_weight=0.34,
        max_total_exposure=1.00
    )
    alloc_res = allocator.compute_weights(signals_dict, benchmark_df=spy_df)
    latest_date = alloc_res.target_weights.index[-1]
    target_weights = alloc_res.target_weights.loc[latest_date]

    # Rank and filter top recommendations
    ranked_assets = sorted(asset_summaries, key=lambda x: x["Alpha_Score"], reverse=True)

    print("\n" + "=" * 80)
    print("                     MARKET SCAN: ASSET ALPHA SCORES")
    print("=" * 80)
    print(f"{'Symbol':<7} | {'Price':<8} | {'Vol Z':<6} | {'RVOL':<5} | {'12M Mom':<8} | {'52W High':<8} | {'Whale?':<7} | {'Alpha Score':<11}")
    print("-" * 80)
    for a in ranked_assets:
        print(f"{a['Symbol']:<7} | ${a['Price']:<7.2f} | {a['Vol_ZScore']:<6.2f} | {a['RVOL']:<5.2f} | {a['Mom_12M']:<8} | {a['52W_High_Ratio']:<8} | {str(a['In_Whale_Regime']):<7} | {a['Alpha_Score']:<11.2f}")

    # Top Recommended Buys on $100k account
    active_buys = target_weights[target_weights > 0.05].to_dict()

    print("\n" + "=" * 80)
    print("             TARGET PORTFOLIO ALLOCATION ($100,000 ACCOUNT)")
    print("=" * 80)

    recommendations = []
    if active_buys and is_macro_bull:
        for sym, weight in active_buys.items():
            price = float(data_dict[sym]["Close"].iloc[-1])
            dollar_alloc = 100_000.0 * weight
            shares = int(dollar_alloc / price)
            stop_loss = round(price * 0.952, 2)
            tp1 = round(price * 1.15, 2)
            tp2 = round(price * 1.30, 2)

            rec = {
                "symbol": sym,
                "weight": f"{weight*100:.1f}%",
                "current_price": price,
                "shares_to_buy": shares,
                "dollar_value": round(dollar_alloc, 2),
                "stop_loss_price": stop_loss,
                "take_profit_target_1": tp1,
                "take_profit_target_2": tp2
            }
            recommendations.append(rec)

            print(f"\n[ACTION: BUY] {sym}")
            print(f"   * Target Weight:     {weight*100:.1f}% (${dollar_alloc:,.2f})")
            print(f"   * Entry Price:       ${price:.2f}")
            print(f"   * Shares to Buy:     {shares} shares")
            print(f"   * Hard Stop Loss:    ${stop_loss:.2f} (-4.8% max risk)")
            print(f"   * Target 1 (+15%):   ${tp1:.2f}")
            print(f"   * Target 2 (+30%):   ${tp2:.2f}")
    else:
        # If no breakout triggered or cash defense
        top3 = [a for a in ranked_assets if a["In_Whale_Regime"]][:3]
        if not top3:
            top3 = ranked_assets[:3]
        print("\n[STATUS] Market is currently in optimal cash/watch regime (No active breakout on Friday close).")
        print("Top 3 Whale Watchlist Candidates for Monday morning open:")
        for idx, item in enumerate(top3, 1):
            sym = item["Symbol"]
            price = item["Price"]
            stop_loss = round(price * 0.952, 2)
            print(f"   {idx}. {sym:<6} (Price: ${price:.2f}, 12M Mom: {item['Mom_12M']}, 52W High: {item['52W_High_Ratio']}, Stop: ${stop_loss:.2f})")

    # Save verification snapshot
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "latest_market_date": str(latest_date),
        "macro_bullish": bool(is_macro_bull),
        "spy_price": float(spy_close),
        "spy_sma200": float(spy_sma200),
        "recommendations": recommendations,
        "full_rankings": ranked_assets
    }

    os.makedirs("output", exist_ok=True)
    with open("output/current_signals_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2)

    print("\n" + "=" * 80)
    print("  Snapshot saved to output/current_signals_snapshot.json for Monday verification!")
    print("=" * 80)


if __name__ == "__main__":
    main()
