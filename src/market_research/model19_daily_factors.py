"""Lower-turnover daily factor rules on previously studied development cohorts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_research.model17_daily import M15_SYMBOLS, TRAINING_SYMBOLS, _make_rows
from market_research.portfolio import simulate_ledger
from market_research.trend_experiment import ROOT

OUT = ROOT / "reports/model19"
CAPITAL = 100_000.0
TRANCHE = 20_000.0  # five overlapping five-session tranches, at most 1x gross
COSTS = (10, 25)
FACTORS = ("resid_mom20", "mom20_volnorm", "mom5_volnorm", "cs_resid20_rank", "cs_mom20_rank", "cs_mom5_rank")


def _simulate(events: list[dict], daily: dict[str, pd.DataFrame], cost_bps: int, side: str) -> dict:
    entries = []
    for e in events:
        if side == "spy":
            weights = [("SPY", TRANCHE)]
        elif side == "long_only":
            longs = e["picks"][0]
            weights = [(symbol, TRANCHE / len(longs)) for symbol in longs] if longs else []
        elif side == "long_short":
            longs, shorts = e["picks"]
            weights = ([(symbol, TRANCHE / 2 / len(longs)) for symbol in longs]
                       + [(symbol, -TRANCHE / 2 / len(shorts)) for symbol in shorts])
        else:
            weights = e["positions"]
        for symbol, dollars in weights:
            row = e["rows"].get(symbol)
            if side == "spy":
                row = e["spy_row"]
            if row is None:
                continue
            entry_price = float(row["entry_open"])
            exit_date = str(row["exit_date"])
            if not np.isfinite(entry_price) or entry_price <= 0:
                raise ValueError(f"invalid entry price for {symbol} on {row.get('entry_date')}")
            entries.append({
                "symbol": symbol, "notional": float(dollars), "entry_price": entry_price,
                "entry_date": str(row["entry_date"]), "exit_date": exit_date,
            })
    result = simulate_ledger(entries, daily, capital=CAPITAL, cost_bps=cost_bps,
                              max_entry_gross_fraction=1.0)
    result["entry_tranches"] = len({e["entry_date"] for e in entries})
    return result


def _make_events(d: pd.DataFrame, daily: dict[str, pd.DataFrame], factor: str,
                 start: str, end: str, top_n: int, regime_gate: bool = False) -> list[dict]:
    sub = d.loc[(d.date >= start) & (d.date < end)].copy()
    events = []
    spy_daily = daily["SPY"].reset_index(drop=True).set_index("date", drop=False)
    for date, group in sub.groupby("date", sort=True):
        group = group.dropna(subset=[factor, "target_return", "entry_open", "exit_date"])
        if len(group) < top_n * 2:
            continue
        if regime_gate:
            spy_row = spy_daily.loc[date] if date in spy_daily.index else None
            if spy_row is None or not np.isfinite(spy_row.get("spy_mom20", np.nan)) or spy_row.spy_mom20 <= 0:
                continue
        ordered = group.sort_values(factor, ascending=False)
        longs = ordered.head(top_n)
        shorts = ordered.tail(top_n)
        rows = {r.symbol: r._asdict() for r in group.itertuples(index=False)}
        sr = spy_daily.loc[date]
        events.append({"date": str(date),
                       "picks": (longs.symbol.tolist(), shorts.symbol.tolist()), "rows": rows,
                       "spy_row": sr.to_dict()})
    return events


def _evaluate(d: pd.DataFrame, daily: dict[str, pd.DataFrame], start: str, end: str,
              validation: bool = False) -> dict:
    out = {"window": [start, end], "strategies": {}}
    # top 10% (at least 3) basket, long-only and dollar-neutral. Score is a
    # point-in-time, date-wise rank of information known at the signal close.
    for factor in FACTORS:
        for top_n in (3, max(3, round(len(d.symbol.unique()) * .10))):
            key = f"{factor}_top{top_n}"
            events = _make_events(d, daily, factor, start, end, top_n)
            # Both sleeves use the same signal dates; the book uses actual next
            # open and fifth-session close, not the stored target for its P&L.
            result = {"selected_dates": len(events), "top_n": top_n, "costs": {}}
            for cost in COSTS:
                result["costs"][str(cost)] = {
                    side: _simulate(events, daily, cost, side)
                    for side in ("long_only", "long_short", "spy")
                }
            out["strategies"][key] = result
    return out


def run() -> dict:
    from market_research.validation import new_run_dir
    out_dir = new_run_dir(OUT)
    symbols = sorted(set(TRAINING_SYMBOLS + M15_SYMBOLS) - {"SPY"})
    d, daily = _make_rows(symbols, "development")
    for factor in FACTORS:
        # Rank inputs cross-sectionally per date; ranks are robust to changing
        # nominal price/volatility scales. The source factors are all trailing.
        d[f"rank_{factor}"] = d.groupby("date")[factor].rank(pct=True) - .5
    # Factor-only score is fixed before evaluating the test windows. Model 18
    # data and all external sector series are excluded by construction.
    results = {
        "status": "development_only_previous_and_consumed_cohorts",
        "data_scope": "Model 17 development symbols from Model 14 training + Model 15 consumed cohort; no Model 16 test symbols; no Model 18 files",
        "target": "next session open to fifth session close; five-session holding; daily staggered $20k tranches",
        "cost_assumption": "fixed round-trip bps on each dollar of position notional; short borrow, impact and dividends excluded",
        "symbols": symbols,
        "fit_or_validation_2025H1": _evaluate(d, daily, "2025-01-15", "2025-06-20", True),
        "development_test_2025H2": _evaluate(d, daily, "2025-07-10", "2025-12-20"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "daily_factor_rules.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    result = run()
    print(f"Wrote daily factor development results for {len(result['symbols'])} existing symbols under {OUT / 'runs'}")
