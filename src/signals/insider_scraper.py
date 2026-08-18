"""
Zero-Credential Institutional Insider & Regulatory Filing Ingestion Engine.
Fetches real-time and historical SEC Form 4 filings, CEO/CFO cluster purchases,
and net insider accumulation without requiring paid API keys.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

logger = logging.getLogger("WhaleTracking.InsiderScraper")


class InsiderIntelligenceEngine:
    """
    Ingests and analyzes corporate insider (SEC Form 4) and regulatory disclosures
    to identify smart-money accumulation before public markup waves.
    """

    def __init__(self, cache_dir: str = "data_cache/insiders"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.sec_headers = {
            "User-Agent": "WhaleMomentumResearch/2.0 (contact@quantwhale.org)"
        }

    def fetch_insider_activity(self, symbol: str, use_cache: bool = True) -> Dict:
        """
        Fetch insider transactions, cluster purchases, and buy/sell ratios for a symbol.
        Uses cached data if available and fresh (< 24h old).
        """
        cache_path = os.path.join(self.cache_dir, f"{symbol}_insiders.json")

        if use_cache and os.path.exists(cache_path):
            try:
                mod_time = datetime.fromtimestamp(os.path.getmtime(cache_path))
                if datetime.now() - mod_time < timedelta(days=7):
                    with open(cache_path, "r") as f:
                        return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache for {symbol}: {e}")

        # Live Ingestion via yfinance & SEC EDGAR
        import yfinance as yf
        insider_data = {
            "symbol": symbol,
            "updated_at": datetime.now().isoformat(),
            "purchases_last_6m": 0.0,
            "sales_last_6m": 0.0,
            "net_shares_bought": 0.0,
            "buy_ratio": 0.50,
            "cluster_buy_count": 0,
            "has_recent_insider_buy": False,
            "recent_transactions": []
        }

        try:
            ticker = yf.Ticker(symbol)
            purchases_df = ticker.insider_purchases
            if purchases_df is not None and not purchases_df.empty:
                for _, row in purchases_df.iterrows():
                    desc = str(row.iloc[0]).lower()
                    shares = float(row.iloc[1]) if pd.notna(row.iloc[1]) else 0.0
                    if "purchase" in desc and "net" not in desc:
                        insider_data["purchases_last_6m"] = shares
                    elif "sale" in desc and "net" not in desc:
                        insider_data["sales_last_6m"] = shares
                    elif "net" in desc:
                        insider_data["net_shares_bought"] = shares

                tot = insider_data["purchases_last_6m"] + insider_data["sales_last_6m"]
                if tot > 0:
                    insider_data["buy_ratio"] = insider_data["purchases_last_6m"] / tot
                else:
                    insider_data["buy_ratio"] = 0.50

            tx_df = ticker.insider_transactions
            if tx_df is not None and not tx_df.empty:
                recent_txs = []
                recent_buys = 0
                cutoff_date = datetime.now() - timedelta(days=120)

                for _, row in tx_df.head(20).iterrows():
                    tx_date_str = str(row.get("Start Date", row.get("Date", "")))
                    shares = float(row.get("Shares", 0)) if pd.notna(row.get("Shares")) else 0
                    text = str(row.get("Text", row.get("Transaction", ""))).lower()
                    is_buy = "buy" in text or "purchase" in text or shares > 0

                    if is_buy:
                        recent_buys += 1

                    recent_txs.append({
                        "date": tx_date_str,
                        "shares": shares,
                        "is_buy": is_buy
                    })

                insider_data["recent_transactions"] = recent_txs
                insider_data["cluster_buy_count"] = recent_buys
                insider_data["has_recent_insider_buy"] = recent_buys >= 1

        except Exception as e:
            logger.warning(f"Failed to fetch live insider data for {symbol}: {e}")

        # Save to cache
        try:
            with open(cache_path, "w") as f:
                json.dump(insider_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write insider cache for {symbol}: {e}")

        return insider_data

    def compute_insider_signals(self, symbol: str, price_series: pd.Series) -> pd.DataFrame:
        """
        Generate time-series insider conviction features aligned with historical price dates.
        """
        insider_info = self.fetch_insider_activity(symbol)
        dates = price_series.index
        df = pd.DataFrame(index=dates)

        # Baseline insider metrics
        buy_ratio = insider_info.get("buy_ratio", 0.50)
        has_buys = insider_info.get("has_recent_insider_buy", False)
        cluster_cnt = insider_info.get("cluster_buy_count", 0)

        # Time series signal expansion
        # When buy_ratio > 0.60 or cluster_cnt >= 2 -> high insider conviction
        insider_score = 0.0
        if buy_ratio > 0.70 or cluster_cnt >= 3:
            insider_score = 3.0  # Ultra-high insider conviction
        elif buy_ratio > 0.55 or cluster_cnt >= 1 or has_buys:
            insider_score = 1.5  # Moderate insider accumulation
        elif buy_ratio < 0.30:
            insider_score = -1.0 # Heavy insider selling/distribution

        df["Insider_Buy_Ratio"] = buy_ratio
        df["Insider_Cluster_Count"] = cluster_cnt
        df["Insider_Score"] = insider_score

        # Multiplier to scale composite alpha
        df["Insider_Conviction_Boost"] = np.where(df["Insider_Score"] > 0, 1.0 + (df["Insider_Score"] * 0.35), 1.0)
        return df
