"""Check Alpaca API credentials and test feed entitlements empirically without leaking secrets."""

import os
import json
from pathlib import Path
import httpx
from dotenv import load_dotenv

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = os.getenv("ALPACA_BASE_URL", "https://data.alpaca.markets")

def mask_key(k: str | None) -> str:
    if not k:
        return "MISSING"
    if len(k) <= 6:
        return f"PRESENT (len {len(k)})"
    return f"{k[:3]}...{k[-3:]} (len {len(k)})"

def test_credentials() -> dict:
    results = {
        "api_key_status": mask_key(ALPACA_API_KEY),
        "secret_key_status": mask_key(ALPACA_SECRET_KEY),
        "has_credentials": bool(ALPACA_API_KEY and ALPACA_SECRET_KEY),
        "feeds_tested": {},
    }

    if not results["has_credentials"]:
        print("[AUTH AUDIT] Alpaca credentials missing from environment / .env")
        return results

    headers = {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Accept": "application/json",
    }

    test_symbols = ["AAPL", "META", "FB", "TWTR", "ATVI"]
    feeds = ["sip", "iex"]

    with httpx.Client(base_url=BASE_URL, headers=headers, timeout=15.0) as client:
        for feed in feeds:
            feed_res = {}
            for sym in test_symbols:
                # Test recent bar (2024-01-05)
                url = f"/v2/stocks/{sym}/bars"
                params = {
                    "timeframe": "1Min",
                    "start": "2024-01-05T14:30:00Z",
                    "end": "2024-01-05T15:00:00Z",
                    "feed": feed,
                    "limit": 10,
                }
                try:
                    resp = client.get(url, params=params)
                    status = resp.status_code
                    bar_count = 0
                    msg = ""
                    if status == 200:
                        data = resp.json()
                        bars = data.get("bars", [])
                        bar_count = len(bars)
                        msg = f"OK, received {bar_count} bars"
                    elif status in (401, 403):
                        msg = f"Subscription/Auth forbidden: {resp.text.strip()}"
                    elif status == 422:
                        msg = f"Unprocessable: {resp.text.strip()}"
                    else:
                        msg = f"HTTP {status}: {resp.text.strip()}"
                    
                    feed_res[f"{sym}_recent"] = {
                        "status_code": status,
                        "bar_count": bar_count,
                        "detail": msg[:150]
                    }
                except Exception as e:
                    feed_res[f"{sym}_recent"] = {"error": str(e)}

                # Also test historical 2016 date for AAPL
                if sym == "AAPL":
                    params_old = {
                        "timeframe": "1Min",
                        "start": "2016-01-05T14:30:00Z",
                        "end": "2016-01-05T15:00:00Z",
                        "feed": feed,
                        "limit": 10,
                    }
                    try:
                        resp_old = client.get(url, params=params_old)
                        b_cnt = 0
                        if resp_old.status_code == 200:
                            b_cnt = len(resp_old.json().get("bars", []))
                        feed_res["AAPL_2016"] = {
                            "status_code": resp_old.status_code,
                            "bar_count": b_cnt,
                            "detail": resp_old.text[:150] if resp_old.status_code != 200 else f"OK, {b_cnt} bars"
                        }
                    except Exception as e:
                        feed_res["AAPL_2016"] = {"error": str(e)}

                # For delisted TWTR, test historical date 2021-06-01
                if sym == "TWTR":
                    params_delisted = {
                        "timeframe": "1Min",
                        "start": "2021-06-01T14:30:00Z",
                        "end": "2021-06-01T15:00:00Z",
                        "feed": feed,
                        "limit": 10,
                    }
                    try:
                        resp_delisted = client.get(url, params=params_delisted)
                        b_cnt = 0
                        if resp_delisted.status_code == 200:
                            b_cnt = len(resp_delisted.json().get("bars", []))
                        feed_res["TWTR_2021_active"] = {
                            "status_code": resp_delisted.status_code,
                            "bar_count": b_cnt,
                            "detail": resp_delisted.text[:150] if resp_delisted.status_code != 200 else f"OK, {b_cnt} bars"
                        }
                    except Exception as e:
                        feed_res["TWTR_2021_active"] = {"error": str(e)}

            results["feeds_tested"][feed] = feed_res

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "credentials_audit.json", "w") as f:
        json.dump(results, f, indent=2)

    return results

if __name__ == "__main__":
    res = test_credentials()
    print("--- Credentials Audit Summary ---")
    print(f"API Key: {res['api_key_status']}")
    print(f"Secret Key: {res['secret_key_status']}")
    for feed, checks in res.get("feeds_tested", {}).items():
        print(f"\nFeed '{feed}':")
        for test_name, detail in checks.items():
            print(f"  {test_name}: status={detail.get('status_code', 'ERR')}, bars={detail.get('bar_count', 0)}, info={detail.get('detail', detail.get('error'))}")
