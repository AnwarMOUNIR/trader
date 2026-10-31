"""Alpaca Live Paper Trading Engine with Continuous Autonomous Background Execution.

Implements the validated quantitative architecture:
- 30-Stock Liquid S&P 500 Universe
- Macro SPY 200-Day SMA Trend Gate (100% cash in bear regimes)
- CPPI Dynamic Drawdown Dial ($6k soft floor, $8.5k hard cash lock)
- 60-Day Momentum + 200-Day Stock Trend Selection
- Single-stock position cap at 15% ($15,000 max on $100k account)
- Autonomous market calendar synchronization (sleeps outside market hours)
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from market_research.acquire import AlpacaAcquisitionClient
from market_research.daily_update import last_completed_session

load_dotenv()

# Logger setup
logger = logging.getLogger("paper_trader")
logger.setLevel(logging.INFO)
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "UNH", "XOM",
    "V", "PG", "COST", "HD", "MA", "ABBV", "CVX", "BAC", "PEP", "KO",
    "AMD", "ADBE", "CRM", "NFLX", "WMT", "CSCO", "TMO", "CAT", "BA", "DIS"
]


class AlpacaPaperTrader:
    """Production Paper Trading Engine interfacing with Alpaca Paper API."""

    def __init__(
        self,
        universe: list[str] | None = None,
        soft_floor_dollars: float = 6_000.0,
        hard_floor_dollars: float = 8_500.0,
        max_stocks: int = 8,
        max_single_stock_frac: float = 0.15,
        state_path: Path | str = "data/paper_trading/state.json",
        audit_path: Path | str = "reports/paper_trading/audit_trail.jsonl",
    ) -> None:
        self.api_key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
        self.api_secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
        self.paper_url = os.getenv("APCA_API_BASE_URL", "https://paper-api.alpaca.markets")
        self.data_url = os.getenv("ALPACA_BASE_URL", "https://data.alpaca.markets")

        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API credentials missing. Check .env for ALPACA_API_KEY and ALPACA_SECRET_KEY.")

        self.universe = universe or DEFAULT_UNIVERSE
        self.soft_floor = float(soft_floor_dollars)
        self.hard_floor = float(hard_floor_dollars)
        self.max_stocks = int(max_stocks)
        self.max_stock_frac = float(max_single_stock_frac)
        self.state_path = Path(state_path)
        self.audit_path = Path(audit_path)

        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

        self.client = httpx.Client(
            base_url=self.paper_url,
            headers={
                "APCA-API-KEY-ID": self.api_key,
                "APCA-API-SECRET-KEY": self.api_secret,
                "Accept": "application/json",
            },
            timeout=25.0,
        )

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> AlpacaPaperTrader:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    # --- Alpaca REST Endpoints ---

    def get_account(self) -> dict:
        """Fetch current account equity, cash, and status."""
        resp = self.client.get("/v2/account")
        resp.raise_for_status()
        data = resp.json()
        return {
            "status": data.get("status"),
            "equity": float(data.get("equity", 0.0)),
            "cash": float(data.get("cash", 0.0)),
            "buying_power": float(data.get("buying_power", 0.0)),
            "currency": data.get("currency", "USD"),
        }

    def get_clock(self) -> dict:
        """Fetch exchange session state and next session timestamps."""
        resp = self.client.get("/v2/clock")
        resp.raise_for_status()
        data = resp.json()
        return {
            "is_open": bool(data.get("is_open", False)),
            "next_open": str(data.get("next_open")),
            "next_close": str(data.get("next_close")),
            "timestamp": str(data.get("timestamp")),
        }

    def get_positions(self) -> dict[str, dict]:
        """Fetch current open positions mapped by symbol."""
        resp = self.client.get("/v2/positions")
        resp.raise_for_status()
        positions = {}
        for p in resp.json():
            sym = p.get("symbol")
            positions[sym] = {
                "symbol": sym,
                "qty": float(p.get("qty", 0.0)),
                "market_value": float(p.get("market_value", 0.0)),
                "cost_basis": float(p.get("cost_basis", 0.0)),
                "current_price": float(p.get("current_price", 0.0)),
                "unrealized_pl": float(p.get("unrealized_pl", 0.0)),
            }
        return positions

    def cancel_open_orders(self) -> int:
        """Cancel pending unfilled orders prior to rebalancing."""
        resp = self.client.delete("/v2/orders")
        if resp.status_code == 200:
            return len(resp.json())
        return 0

    def submit_order(self, symbol: str, side: str, notional: float | None = None, qty: float | None = None) -> dict:
        """Submit a day market order via Alpaca Paper API."""
        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side.lower(),
            "type": "market",
            "time_in_force": "day",
        }
        if qty is not None and qty > 0:
            payload["qty"] = str(round(qty, 4))
        elif notional is not None and notional > 0:
            payload["notional"] = str(round(notional, 2))
        else:
            raise ValueError(f"Order for {symbol} must specify positive qty or notional")

        resp = self.client.post("/v2/orders", json=payload)
        resp.raise_for_status()
        return resp.json()

    # --- High-Water Mark & State Management ---

    def load_state(self, current_equity: float) -> dict:
        """Load or initialize high-water mark state."""
        if self.state_path.exists():
            try:
                state = json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                state = {}
        else:
            state = {}

        high_water = max(float(state.get("high_water_mark", current_equity)), current_equity)
        state["high_water_mark"] = high_water
        state["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return state

    def update_high_water(self, current_equity: float) -> float:
        """Update and persist high-water mark."""
        state = self.load_state(current_equity)
        if current_equity > state["high_water_mark"]:
            state["high_water_mark"] = current_equity
            state["last_updated_utc"] = datetime.now(timezone.utc).isoformat()
            self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        return float(state["high_water_mark"])

    # --- Risk Engine & Signal Generation ---

    def compute_drawdown_dial(self, current_equity: float, high_water: float) -> tuple[float, str]:
        """Compute the CPPI Drawdown Dial based on distance from peak."""
        drawdown = max(0.0, high_water - current_equity)
        if drawdown >= self.hard_floor:
            return 0.0, f"HARD_FLOOR_CASH_LOCK (DD: ${drawdown:,.2f} >= ${self.hard_floor:,.2f})"
        if drawdown >= self.soft_floor:
            scale = 1.0 - 0.80 * ((drawdown - self.soft_floor) / (self.hard_floor - self.soft_floor))
            scale = float(np.clip(scale, 0.20, 1.0))
            return scale, f"SOFT_FLOOR_DELEVER (DD: ${drawdown:,.2f}, Dial: {scale:.1%})"
        return 1.0, f"NORMAL_REGIME (DD: ${drawdown:,.2f})"

    def evaluate_macro_trend(self) -> tuple[bool, float, float]:
        """Fetch SPY daily history through latest closed session and verify if SPY > 200 SMA."""
        end_date = last_completed_session(datetime.now(timezone.utc))
        if end_date is None:
            end_date = (datetime.now(timezone.utc) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

        with AlpacaAcquisitionClient(feed="sip", base_url=self.data_url) as client:
            bars = client.fetch_bars_chunk("SPY", f"{start_date}T00:00:00Z", f"{end_date}T23:59:59Z", timeframe="1Day")

        if len(bars) < 200:
            logger.warning(f"SPY has only {len(bars)} daily bars. Defaulting to defensive.")
            return False, 0.0, 0.0

        closes = pd.Series([b["c"] for b in bars])
        current_spy = float(closes.iloc[-1])
        sma200 = float(closes.rolling(200).mean().iloc[-1])
        is_bull = current_spy > sma200
        return is_bull, current_spy, sma200

    def rank_momentum_universe(self) -> list[tuple[str, float, float]]:
        """Rank universe stocks by 60-day momentum that trade above their 200-day SMA."""
        candidates = []
        end_date = last_completed_session(datetime.now(timezone.utc))
        if end_date is None:
            end_date = (datetime.now(timezone.utc) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")

        with AlpacaAcquisitionClient(feed="sip", base_url=self.data_url) as client:
            for sym in self.universe:
                try:
                    bars = client.fetch_bars_chunk(sym, f"{start_date}T00:00:00Z", f"{end_date}T23:59:59Z", timeframe="1Day")
                    if len(bars) < 200:
                        continue
                    closes = pd.Series([b["c"] for b in bars])
                    cur_price = float(closes.iloc[-1])
                    sma200 = float(closes.rolling(200).mean().iloc[-1])
                    if cur_price > sma200:
                        mom60 = float(cur_price / closes.iloc[-60] - 1.0)
                        candidates.append((sym, mom60, cur_price))
                except Exception as exc:
                    logger.debug(f"Failed bar fetch for {sym}: {exc}")
                    continue

        # Sort descending by 60-day momentum
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:self.max_stocks]

    # --- Core Execution Cycle ---

    def evaluate_and_rebalance(self, dry_run: bool = False) -> dict:
        """Run a complete evaluation and rebalancing cycle."""
        timestamp = datetime.now(timezone.utc).isoformat()
        acc = self.get_account()
        clock = self.get_clock()
        equity = acc["equity"]
        positions = self.get_positions()

        high_water = self.update_high_water(equity)
        dial_fraction, dial_reason = self.compute_drawdown_dial(equity, high_water)
        macro_bull, spy_px, spy_sma = self.evaluate_macro_trend()

        logger.info(f"Account Equity: ${equity:,.2f} | Peak: ${high_water:,.2f} | Clock Open: {clock['is_open']}")
        logger.info(f"SPY: ${spy_px:.2f} (200 SMA: ${spy_sma:.2f}) -> Regime: {'BULL' if macro_bull else 'BEAR'}")
        logger.info(f"Drawdown Dial: {dial_fraction:.1%} ({dial_reason})")

        # Determine target gross exposure
        if not macro_bull:
            target_gross_frac = 0.0
            strategy_state = "BEAR_REGIME_CASH_LOCK"
        elif dial_fraction == 0.0:
            target_gross_frac = 0.0
            strategy_state = "HARD_FLOOR_CASH_LOCK"
        else:
            target_gross_frac = dial_fraction
            strategy_state = "ACTIVE_MOMENTUM"

        target_total_equity = equity * target_gross_frac
        top_picks = self.rank_momentum_universe() if target_gross_frac > 0 else []
        target_symbols = [p[0] for p in top_picks]

        logger.info(f"Strategy State: {strategy_state} | Target Capital: ${target_total_equity:,.2f}")
        if target_symbols:
            logger.info(f"Selected Top Momentum Picks ({len(target_symbols)}): {', '.join(target_symbols)}")

        orders_placed = []
        if clock["is_open"] and not dry_run:
            self.cancel_open_orders()

            # 1. Liquidate positions not in target picks
            for sym, pos in positions.items():
                if sym not in target_symbols:
                    logger.info(f"Exiting position: {sym} (Qty: {pos['qty']})")
                    try:
                        order = self.submit_order(symbol=sym, side="sell", qty=pos["qty"])
                        orders_placed.append(order)
                    except Exception as exc:
                        logger.error(f"Failed to submit sell order for {sym}: {exc}")

            # 2. Rebalance / enter target positions
            if target_symbols:
                alloc_per_stock = min(target_total_equity / len(target_symbols), equity * self.max_stock_frac)
                for sym, mom, px in top_picks:
                    cur_mv = positions.get(sym, {}).get("market_value", 0.0)
                    needed_dollar = alloc_per_stock - cur_mv
                    if needed_dollar > 100.0:  # Minimum trade threshold to avoid churn
                        logger.info(f"Entering/Scaling {sym}: +${needed_dollar:,.2f} (Target: ${alloc_per_stock:,.2f})")
                        try:
                            order = self.submit_order(symbol=sym, side="buy", notional=needed_dollar)
                            orders_placed.append(order)
                        except Exception as exc:
                            logger.error(f"Failed to submit buy order for {sym}: {exc}")

        result = {
            "timestamp_utc": timestamp,
            "market_open": clock["is_open"],
            "account_equity": equity,
            "high_water_mark": high_water,
            "current_drawdown": high_water - equity,
            "dial_factor": dial_fraction,
            "macro_bull": macro_bull,
            "strategy_state": strategy_state,
            "target_symbols": target_symbols,
            "orders_placed_count": len(orders_placed),
            "dry_run": dry_run,
        }

        # Append to audit trail
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        return result

    def run_daemon(self, poll_interval_minutes: int = 15) -> None:
        """Continuous background execution loop."""
        logger.info(f"=== Starting Alpaca Paper Trading Daemon (Interval: {poll_interval_minutes}m) ===")
        while True:
            try:
                clock = self.get_clock()
                if not clock["is_open"]:
                    next_open = clock["next_open"]
                    logger.info(f"Market is currently CLOSED. Next open: {next_open}. Sleeping...")
                    # Compute seconds until next open
                    now = pd.Timestamp.now(tz="UTC")
                    target = pd.Timestamp(next_open)
                    secs = max(60, int((target - now).total_seconds()) - 60)
                    time.sleep(min(secs, 1800))  # Sleep at most 30 mins before re-checking clock
                    continue

                # Market is open: run rebalancing cycle
                self.evaluate_and_rebalance(dry_run=False)
                time.sleep(poll_interval_minutes * 60)

            except Exception as exc:
                logger.error(f"Daemon cycle encountered error: {exc}", exc_info=True)
                time.sleep(60)
