"""Alpaca intraday historical data acquisition client with robust pagination, rate-limiting, and atomic parquet persistence."""

import os
import time
import random
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import httpx
import pandas as pd
from dotenv import load_dotenv

from market_research.db import get_db, init_db

load_dotenv()

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://data.alpaca.markets")

class AlpacaAcquisitionClient:
    def __init__(self, feed: str = "sip", base_url: str = ALPACA_BASE_URL):
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise ValueError("Alpaca API credentials missing. Set ALPACA_API_KEY and ALPACA_SECRET_KEY.")
        self.feed = feed
        self.base_url = base_url
        self.client = httpx.Client(
            base_url=self.base_url,
            headers={
                "APCA-API-KEY-ID": ALPACA_API_KEY,
                "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
                "Accept": "application/json",
            },
            timeout=30.0,
        )

    def close(self):
        self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def fetch_bars_chunk(
        self,
        symbol: str,
        start_iso: str,
        end_iso: str,
        timeframe: str = "1Min",
        adjustment: str = "raw",
        max_retries: int = 5,
    ) -> list[dict]:
        """
        Fetches all bars for a symbol and date range, exhaustively paginating.
        Returns list of dicts.
        """
        url = f"/v2/stocks/{symbol}/bars"
        all_bars = []
        page_token = None
        seen_tokens = set()

        while True:
            params = {
                "timeframe": timeframe,
                "start": start_iso,
                "end": end_iso,
                "feed": self.feed,
                "adjustment": adjustment,
                "limit": 10000,
            }
            if page_token:
                params["page_token"] = page_token

            attempt = 0
            while attempt < max_retries:
                attempt += 1
                try:
                    resp = self.client.get(url, params=params)
                    
                    # Handle rate limiting
                    rem = resp.headers.get("x-ratelimit-remaining")
                    if rem is not None and rem.isdigit() and int(rem) < 5:
                        time.sleep(1.0 + random.uniform(0.1, 0.5))

                    if resp.status_code == 200:
                        data = resp.json()
                        bars = data.get("bars") or []
                        all_bars.extend(bars)
                        page_token = data.get("next_page_token")
                        break
                    elif resp.status_code == 429:
                        reset_header = resp.headers.get("Retry-After", "5")
                        try:
                            sleep_time = max(0.0, float(reset_header))
                        except ValueError:
                            sleep_time = 2.0 ** attempt + random.uniform(0.5, 2.0)
                        print(f"Rate limited (429) for {symbol}. Sleeping {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                    elif resp.status_code in (500, 502, 503, 504):
                        sleep_time = 2.0 ** attempt + random.uniform(0.5, 2.0)
                        print(f"Server error {resp.status_code} for {symbol}. Retry {attempt}/{max_retries} in {sleep_time:.2f}s...")
                        time.sleep(sleep_time)
                    elif resp.status_code in (401, 403):
                        # Authentication/authorization failures are not transient.
                        raise PermissionError(f"Alpaca authentication/authorization failed ({resp.status_code})")
                    elif resp.status_code in (404, 422):
                        raise ValueError(
                            f"Alpaca rejected {symbol} request ({resp.status_code}): {resp.text}"
                        )
                    else:
                        resp.raise_for_status()
                except (httpx.RequestError, httpx.HTTPStatusError) as e:
                    if attempt >= max_retries:
                        raise RuntimeError(
                            f"Alpaca request failed after {max_retries} attempts for {symbol} "
                            f"page_token={page_token!r}"
                        ) from e
                    time.sleep(2.0 ** attempt + random.uniform(0.5, 1.5))
                except ValueError:
                    # Includes explicit unsupported-request errors above.
                    raise

            else:
                # 429/5xx retries are handled without raising inside the loop.
                raise RuntimeError(
                    f"Alpaca request exhausted {max_retries} attempts for {symbol} "
                    f"page_token={page_token!r}"
                )

            if not page_token:
                break
            if page_token in seen_tokens:
                raise RuntimeError(f"Alpaca returned a repeated pagination token for {symbol}: {page_token!r}")
            seen_tokens.add(page_token)

        return all_bars

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def acquire_symbol_year(
    client: AlpacaAcquisitionClient,
    symbol: str,
    year: int,
    security_id: str | None = None,
    output_base: Path = Path("data/raw"),
) -> dict:
    """Acquires 1 year of 1-minute bars for a symbol, saves atomically to Parquet, and logs state."""
    init_db()
    conn = get_db()
    destination = str(output_base.resolve())
    destination_key = hashlib.sha256(destination.encode("utf-8")).hexdigest()[:12]
    task_id = f"RAW_{symbol}_{year}_{client.feed}_1Min_{destination_key}"
    sec_id = security_id or f"SEC_{symbol.replace('.', '_')}"

    # Check if already completed in DB
    row = conn.execute("SELECT status, bars_count, file_path, file_sha256 FROM acquisition_tasks WHERE task_id = ?", (task_id,)).fetchone()
    if row and row["status"] == "COMPLETED":
        cached_path = Path(row["file_path"]) if row["file_path"] else None
        if cached_path and not cached_path.is_absolute():
            cached_path = Path.cwd() / cached_path
        if cached_path and cached_path.is_file() and row["file_sha256"]:
            if sha256_file(cached_path) == row["file_sha256"]:
                conn.close()
                return {"status": "COMPLETED", "bars_count": row["bars_count"], "file_path": row["file_path"], "sha256": row["file_sha256"]}
    # EMPTY entries are retried: without a response manifest, a historical EMPTY
    # may have been produced by an exhausted request and cannot be verified.

    now_iso = datetime.now(timezone.utc).isoformat()
    conn.execute("""
        INSERT INTO acquisition_tasks (task_id, security_id, symbol, source, feed, timeframe, start_date, end_date, status, created_at, updated_at)
        VALUES (?, ?, ?, 'alpaca', ?, '1Min', ?, ?, 'IN_PROGRESS', ?, ?)
        ON CONFLICT(task_id) DO UPDATE SET status='IN_PROGRESS', updated_at=?
    """, (task_id, sec_id, symbol, client.feed, f"{year}-01-01", f"{year}-12-31", now_iso, now_iso, now_iso))
    conn.commit()

    start_iso = f"{year}-01-01T00:00:00Z"
    end_iso = f"{year}-12-31T23:59:59Z"

    try:
        bars = client.fetch_bars_chunk(symbol, start_iso, end_iso, timeframe="1Min", adjustment="raw")
        bar_count = len(bars)
        status = "COMPLETED" if bar_count > 0 else "EMPTY"

        file_path_str = None
        file_hash = None

        if bar_count > 0:
            df = pd.DataFrame(bars)
            # Map columns to clear names
            col_map = {"t": "timestamp_utc", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume", "n": "trade_count", "vw": "vwap"}
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
            df["symbol"] = symbol
            df["security_id"] = sec_id
            df["feed"] = client.feed
            df["source"] = "alpaca"
            df["adjustment"] = "raw"

            sym_dir = output_base / f"feed={client.feed}" / "timeframe=1Min" / f"symbol={symbol}"
            sym_dir.mkdir(parents=True, exist_ok=True)
            target_path = sym_dir / f"{symbol}_{year}_1min.parquet"
            tmp_path = sym_dir / f"{symbol}_{year}_1min.parquet.tmp"

            # Write atomically
            df.to_parquet(tmp_path, index=False, compression="snappy")
            tmp_path.replace(target_path)

            file_path_str = str(target_path.as_posix())
            file_hash = sha256_file(target_path)

        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            UPDATE acquisition_tasks
            SET status = ?, bars_count = ?, file_path = ?, file_sha256 = ?, updated_at = ?
            WHERE task_id = ?
        """, (status, bar_count, file_path_str, file_hash, now_iso, task_id))
        conn.commit()
        conn.close()

        return {
            "status": status,
            "bars_count": bar_count,
            "file_path": file_path_str,
            "sha256": file_hash,
        }
    except Exception as e:
        now_iso = datetime.now(timezone.utc).isoformat()
        conn.execute("""
            UPDATE acquisition_tasks
            SET status = 'FAILED', error_message = ?, updated_at = ?, attempts = attempts + 1
            WHERE task_id = ?
        """, (str(e), now_iso, task_id))
        conn.commit()
        conn.close()
        raise e
