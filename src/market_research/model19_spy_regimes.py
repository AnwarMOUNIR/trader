"""Test simple lagged market-regime controllers against SPY buy-and-hold."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_research.model17_daily import M15_LOCK_DATA, M16_LOCK_DATA, _daily, _source_paths
from market_research.trend_experiment import ROOT

OUT = ROOT / "reports/model19"
CAPITAL = 100_000.0
COSTS = (10, 25)
WINDOWS = {
    "2016_2018": ("2016-01-01", "2019-01-01"),
    "2019_2021": ("2019-01-01", "2022-01-01"),
    "2022_2024": ("2022-01-01", "2025-01-01"),
    "2025_2026_oct": ("2025-01-01", "2026-10-29"),
}


def _spydaily() -> pd.DataFrame:
    paths = _source_paths("SPY") + [M15_LOCK_DATA / "SPY_30min.parquet", M16_LOCK_DATA / "SPY_30min.parquet"]
    frames = [_daily(p) for p in paths if p.exists()]
    d = pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    d["date"] = d.date.astype(str)
    for n in (10, 20, 50, 100, 200):
        d[f"sma{n}"] = d.close.rolling(n, min_periods=n).mean()
    d["rv20"] = d.close.pct_change().rolling(20, min_periods=15).std() * np.sqrt(252)
    # Decision at prior close, position held from today's open to next open.
    d["open_to_next_open"] = d.open.shift(-1) / d.open - 1
    d["decision_date"] = d.date.shift(1)
    d["position_sma10_50"] = (d.sma10 > d.sma50).shift(1).fillna(False).astype(float)
    d["position_sma20_100"] = (d.sma20 > d.sma100).shift(1).fillna(False).astype(float)
    d["position_sma50_200"] = (d.sma50 > d.sma200).shift(1).fillna(False).astype(float)
    d["position_mom20"] = (d.close.shift(1) > d.close.shift(21)).fillna(False).astype(float)
    base = d.position_sma50_200
    # Volatility targeting only scales an already-positive trend state; it is
    # capped at 1x and uses realized volatility available at the prior close.
    d["position_trend_vol10"] = (base * (0.10 / d.rv20.shift(1)).clip(upper=1.0)).fillna(0.0)
    experts = ["buy_hold", "position_sma10_50", "position_sma20_100", "position_sma50_200",
               "position_mom20", "position_trend_vol10"]
    expert_positions = pd.DataFrame({"buy_hold": np.ones(len(d)), **{
        col: d[col].to_numpy(float) for col in experts if col != "buy_hold"
    }}, index=d.index)
    expert_returns = expert_positions.mul(d.open_to_next_open, axis=0)
    # At open t, mixture weights use realized expert returns only through the
    # prior open-to-open interval. Softmax temperature is fixed at 2% cumulative
    # 60-session return; this is a research candidate, not tuned on the windows.
    trailing = expert_returns.rolling(60, min_periods=20).sum().shift(1)
    z = trailing.to_numpy(float) / 0.02
    z = np.where(np.isfinite(z), z, 0.0)
    z -= z.max(axis=1, keepdims=True)
    ew = np.exp(np.clip(z, -50, 0))
    ew /= ew.sum(axis=1, keepdims=True)
    d["position_online_mix60"] = np.sum(ew * expert_positions.to_numpy(float), axis=1)
    return d


def _run_window(d: pd.DataFrame, start: str, end: str, cost_bps: int) -> dict:
    x = d.loc[(d.date >= start) & (d.date < end)].copy()
    x = x.loc[x.open_to_next_open.notna()]
    strategies = {"buy_hold": np.ones(len(x))}
    for col in ("position_sma10_50", "position_sma20_100", "position_sma50_200", "position_mom20", "position_trend_vol10", "position_online_mix60"):
        strategies[col.removeprefix("position_")] = x[col].to_numpy(float)
    strategies["buy_hold_dd_dial"] = np.ones(len(x))
    strategies["sma50_200_dd_dial"] = x.position_sma50_200.to_numpy(float)
    out = {}
    for name, weight in strategies.items():
        nav = CAPITAL
        holding_value = 0.0
        total_cost = 0.0
        turnover_total = 0.0
        daily_pnl = []
        nav_marks = []
        exposure_path = []
        high_water = CAPITAL
        intraday_dd_max = 0.0
        for i, (target_weight, ret, opn, hi, lo) in enumerate(zip(
                weight, x.open_to_next_open.to_numpy(float), x.open.to_numpy(float),
                x.high.to_numpy(float), x.low.to_numpy(float))):
            actual_weight = float(target_weight)
            if name.endswith("_dd_dial"):
                high_water = max(high_water, nav)
                drawdown_before_trade = max(0.0, high_water - nav)
                # Begin cutting at $10k drawdown; scale to flat at $25k,
                # leaving a $5k buffer against the project's $30k risk limit.
                dial = float(np.clip((25_000 - drawdown_before_trade) / 15_000, 0.0, 1.0))
                actual_weight *= dial
            exposure_path.append(actual_weight)
            if name == "buy_hold":
                target_value = holding_value if i > 0 else CAPITAL
            else:
                target_value = actual_weight * nav
            trade = target_value - holding_value
            fee = abs(trade) * cost_bps / 20_000
            nav -= fee
            total_cost += fee
            turnover_total += abs(trade)
            holding_value = target_value
            if opn > 0 and np.isfinite(hi) and np.isfinite(lo):
                high_equity = nav + holding_value * (hi / opn - 1.0)
                low_equity = nav + holding_value * (lo / opn - 1.0)
                high_water = max(high_water, high_equity)
                intraday_dd_max = max(intraday_dd_max, high_water - low_equity)
            pnl = holding_value * float(ret)
            nav += pnl
            high_water = max(high_water, nav)
            daily_pnl.append(pnl - fee)
            holding_value *= 1.0 + float(ret)
            nav_marks.append(nav)
        # Close the position at the end of the window and include liquidation
        # friction in final equity and drawdown.
        exit_fee = abs(holding_value) * cost_bps / 20_000
        nav -= exit_fee
        total_cost += exit_fee
        turnover_total += abs(holding_value)
        if nav_marks:
            nav_marks[-1] = nav
        equity = np.asarray(nav_marks, float)
        peak = np.maximum.accumulate(np.r_[CAPITAL, equity])[1:]
        dd = peak - equity
        nav_before = np.r_[CAPITAL, equity[:-1]] if len(equity) else np.asarray([], float)
        daily_ret = np.asarray(daily_pnl, float) / np.maximum(nav_before, 1.0)
        vol = float(np.std(daily_ret, ddof=1) * np.sqrt(252)) if len(daily_ret) > 1 else 0.0
        sharpe = float(np.mean(daily_ret) / np.std(daily_ret, ddof=1) * np.sqrt(252)) if len(daily_ret) > 1 and np.std(daily_ret, ddof=1) else 0.0
        out[name] = {"net_pnl": float(nav - CAPITAL), "return_pct_start": float((nav - CAPITAL) / CAPITAL * 100),
                     "max_drawdown_dollars": float(dd.max(initial=0.0)),
                     "max_drawdown_dollars_conservative_ohlc": float(max(dd.max(initial=0.0), intraday_dd_max)),
                     "annualized_daily_vol": vol,
                     "sharpe_daily_simple": sharpe, "average_exposure_pct": float(np.mean(exposure_path) * 100),
                     "turnover_notional": float(turnover_total), "costs_paid": float(total_cost),
                     "observations": int(len(x))}
    return out


def run() -> dict:
    d = _spydaily()
    results = {name: {str(cost): _run_window(d, *window, cost) for cost in COSTS}
               for name, window in WINDOWS.items()}
    full_period = {str(cost): _run_window(d, "2016-01-01", "2026-10-29", cost) for cost in COSTS}
    payload = {"status": "development_only_previously_studied_SPY_context",
               "source": "existing Model 14 training SPY archives plus already-consumed Model 15/16 SPY context",
               "data_scope": "SPY only; no Model 18 data",
               "execution": "prior-close signal; position applied to next open-to-open return; 1x max; round-trip costs charged on turnover and terminal liquidation",
               "strategies": ["buy_hold", "SMA10/50", "SMA20/100", "SMA50/200", "20-day momentum", "SMA50/200 plus 10% annualized volatility target", "60-session online softmax mixture of passive and trend experts (2% temperature)", "buy-and-hold with drawdown dial ($10k start reducing, flat at $25k)", "SMA50/200 with same drawdown dial"],
               "windows": results, "full_period_2016_2026": full_period}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spy_regime_rules.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if __name__ == "__main__":
    result = run()
    print(f"Wrote SPY regime rule comparisons to {OUT / 'spy_regime_rules.json'}")
