"""Probe Alpaca historical depth across multiple years."""

import os
import json
import httpx
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://data.alpaca.markets")

headers = {
    "APCA-API-KEY-ID": ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Accept": "application/json",
}

years = [2010, 2014, 2015, 2015, 2016, 2017, 2020, 2024]
test_sym = "AAPL"

with httpx.Client(base_url=BASE_URL, headers=headers, timeout=15.0) as client:
    for feed in ["sip", "iex"]:
        print(f"\n--- Testing Feed: {feed} for {test_sym} ---")
        for y in [2010, 2014, 2015, 2016, 2017, 2020]:
            # First Tuesday of October for consistency
            start = f"{y}-10-05T14:30:00Z"
            end = f"{y}-10-05T15:30:00Z"
            url = f"/v2/stocks/{test_sym}/bars"
            params = {
                "timeframe": "1Min",
                "start": start,
                "end": end,
                "feed": feed,
                "limit": 5,
            }
            resp = client.get(url, params=params)
            if resp.status_code == 200:
                bars = resp.json().get("bars") or []
                print(f"Year {y}: 200 OK, bars returned: {len(bars)}")
                if bars:
                    print(f"   First bar: {bars[0]['t']} close={bars[0]['c']}")
            else:
                print(f"Year {y}: HTTP {resp.status_code} - {resp.text[:80]}")
