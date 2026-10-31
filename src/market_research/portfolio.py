"""Cash-and-shares accounting for the daily strategy books.

Entries are dollar notionals at the supplied open price.  Costs are quoted as
round-trip basis points and charged half on each fill.  Missing active-position
marks and positions left open at the end fail closed; callers must provide a
complete mark history through each scheduled exit.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


def _value_at(frame: pd.DataFrame, date: str, column: str) -> float:
    if date not in frame.index:
        raise ValueError(f"missing {column} mark for active position on {date}")
    value = frame.loc[date, column]
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"invalid {column} mark for active position on {date}: {value}")
    return value


def simulate_ledger(entries: Iterable[dict], daily: dict[str, pd.DataFrame], *,
                    capital: float, cost_bps: float,
                    max_entry_gross_fraction: float = 1.0) -> dict:
    """Simulate signed dollar entries, charging fills and enforcing a NAV cap.

    Each entry has symbol, entry_date, exit_date, entry_price, and signed
    notional. Concurrent entries are scaled pro rata when their aggregate
    requested notional would exceed available NAV after the entry fee.
    """
    try:
        capital, cost_bps, max_entry_gross_fraction = map(
            float, (capital, cost_bps, max_entry_gross_fraction))
    except (TypeError, ValueError) as exc:
        raise ValueError("capital, costs, and gross fraction must be finite numbers") from exc
    if not np.isfinite([capital, cost_bps, max_entry_gross_fraction]).all():
        raise ValueError("capital, costs, and gross fraction must be finite numbers")
    if capital <= 0 or cost_bps < 0 or max_entry_gross_fraction <= 0:
        raise ValueError("capital and gross fraction must be positive; costs cannot be negative")
    by_date: dict[str, list[dict]] = defaultdict(list)
    all_dates = set()
    for symbol, frame in daily.items():
        if frame.index.has_duplicates:
            raise ValueError(f"duplicate mark dates for {symbol}")
        date_keys = [str(x) for x in frame.index]
        if len(set(date_keys)) != len(date_keys):
            raise ValueError(f"duplicate normalized mark dates for {symbol}")
        all_dates.update(date_keys)
    for raw in entries:
        e = dict(raw)
        symbol = str(e["symbol"])
        if symbol not in daily:
            raise ValueError(f"missing price history for {symbol}")
        e["entry_date"], e["exit_date"] = str(e["entry_date"]), str(e["exit_date"])
        px, notional = float(e["entry_price"]), float(e["notional"])
        if not np.isfinite(px) or px <= 0 or not np.isfinite(notional) or notional == 0:
            raise ValueError(f"invalid entry for {symbol} on {e['entry_date']}")
        if e["exit_date"] < e["entry_date"]:
            raise ValueError(f"exit precedes entry for {symbol}")
        e["entry_price"], e["notional"] = px, notional
        by_date[e["entry_date"]].append(e)
        all_dates.add(e["entry_date"])
    dates = sorted(all_dates)
    cash = float(capital)
    active: list[dict] = []
    fills: list[dict] = []
    trades: list[dict] = []
    daily_ledger: list[dict] = []
    costs_paid = 0.0
    peak, max_drawdown, max_gross = cash, 0.0, 0.0
    fee_rate = float(cost_bps) / 20_000.0
    prior_marks: dict[str, float] = {}

    for date in dates:
        todays = by_date.get(date, [])
        current_mv = 0.0
        nav_before = cash
        for pos in active:
            mark = _value_at(daily[pos["symbol"]], date, "open")
            current_mv += abs(pos["shares"] * mark)
            nav_before += pos["shares"] * mark
        gross_at_open_before_entries = current_mv
        gross_at_open_after_entries = current_mv
        if todays:
            # Use today's actual open marks for existing holdings so overnight
            # gaps are reflected before sizing today's new fills. Reserve the
            # fee with its effect on the permitted gross/NAV ratio included.
            fraction = max_entry_gross_fraction
            cap = max(0.0, nav_before * fraction - current_mv) / (1 + fraction * fee_rate)
            requested = sum(abs(e["notional"]) for e in todays)
            scale = min(1.0, cap / requested) if requested else 0.0
            gross_at_open_after_entries = current_mv + requested * scale
            for e in todays:
                notional = e["notional"] * scale
                if abs(notional) < 1e-12:
                    continue
                shares = notional / e["entry_price"]
                fee = abs(notional) * fee_rate
                cash -= notional + fee
                costs_paid += fee
                pos = {"symbol": e["symbol"], "shares": shares,
                       "entry_price": e["entry_price"], "entry_date": date,
                       "exit_date": e["exit_date"], "entry_fee": fee,
                       "entry_notional": notional}
                active.append(pos)
                fills.append({"date": date, "symbol": e["symbol"], "side": "buy" if shares > 0 else "short",
                              "shares": shares, "price": e["entry_price"], "notional": notional, "fee": fee})

        gross_before_exits = 0.0
        keep = []
        for pos in active:
            frame = daily[pos["symbol"]]
            close = _value_at(frame, date, "close")
            prior_marks[pos["symbol"]] = close
            market_value = pos["shares"] * close
            gross_before_exits += abs(market_value)
            if pos["exit_date"] == date:
                exit_notional = pos["shares"] * close
                fee = abs(exit_notional) * fee_rate
                cash += exit_notional - fee
                costs_paid += fee
                pnl_before_costs = pos["shares"] * (close - pos["entry_price"])
                trades.append({"symbol": pos["symbol"], "entry_date": pos["entry_date"],
                               "exit_date": date, "shares": pos["shares"],
                               "entry_price": pos["entry_price"], "exit_price": close,
                               "entry_notional": pos["entry_notional"],
                               "pnl_before_costs": pnl_before_costs,
                               "costs_paid": pos["entry_fee"] + fee,
                               "net_pnl": pnl_before_costs - pos["entry_fee"] - fee})
                fills.append({"date": date, "symbol": pos["symbol"], "side": "sell" if pos["shares"] > 0 else "cover",
                              "shares": -pos["shares"], "price": close, "notional": -exit_notional, "fee": fee})
            else:
                keep.append(pos)
        active = keep
        closing_gross = sum(abs(pos["shares"] * prior_marks[pos["symbol"]]) for pos in active)
        equity = cash + sum(pos["shares"] * prior_marks[pos["symbol"]] for pos in active)
        max_gross = max(max_gross, gross_at_open_after_entries, gross_before_exits)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
        daily_ledger.append({"date": date, "cash": cash, "equity": equity,
                             "gross_before_exits": gross_before_exits,
                             "gross_at_open_before_entries": gross_at_open_before_entries,
                             "gross_at_open_after_entries": gross_at_open_after_entries,
                             "gross_after_exits": closing_gross,
                             "gross_to_equity": gross_before_exits / equity if equity > 0 else None,
                             "open_positions": len(active)})

    if active:
        open_summary = [{"symbol": p["symbol"], "shares": p["shares"], "entry_date": p["entry_date"],
                         "scheduled_exit_date": p["exit_date"]} for p in active]
        raise ValueError(f"terminal holdings remain without closure: {open_summary}")
    ending_equity = cash
    net_pnl = ending_equity - capital
    trade_net = sum(t["net_pnl"] for t in trades)
    if not np.isclose(net_pnl, trade_net, rtol=1e-9, atol=1e-7):
        raise AssertionError(f"ledger does not reconcile: equity P&L {net_pnl}, trade P&L {trade_net}")
    return {"net_pnl": float(net_pnl), "ending_equity": float(ending_equity),
            "max_drawdown_dollars": float(max_drawdown), "costs_paid": float(costs_paid),
            "max_gross_dollars": float(max_gross), "trades": int(len(trades)),
            "fills": fills, "trade_ledger": trades, "daily_ledger": daily_ledger,
            "terminal_holdings": [],
            "gross_cap_note": "Entry notional is capped against pre-entry NAV including entry fees; subsequent price gaps can move market-value gross above NAV until the next fill/exit."}
