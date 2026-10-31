"""Recompute Model 15 development portfolios and concentration diagnostics."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_research.model15_research import (
    GROSS_LIMIT, PROTOCOL_PATH, REPORT_DIR, RT_COST_BPS, STARTING_CAPITAL,
    _portfolio, _rows, _split,
)


def run() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    symbols = [s for s in protocol["training_symbols"] if s != "SPY"]
    rows, bars = _rows("post", symbols)
    _, _, test = _split(rows, "post")
    pred = pd.read_parquet(REPORT_DIR / "oos_predictions.parquet")
    pred = pred.loc[pred.era.eq("post")].copy()
    key = ["symbol", "timestamp_utc"]
    results, diagnostics = [], {}
    base = test.copy().reset_index(drop=True)
    for name, signals in pred.groupby("model", sort=True):
        merged = base.merge(signals[key + ["expected_return"]], on=key, how="left", validate="one_to_one")
        if merged.expected_return.isna().any():
            raise RuntimeError(f"Missing predictions for {name}")
        mu = merged.expected_return.to_numpy(float)
        neutral = name.endswith("residual")
        portfolio = _portfolio(merged, mu, bars, name, market_neutral=neutral)
        beta = merged.beta_spy.to_numpy(float) if neutral else np.zeros(len(merged))
        vol_col = "rv_resid_30m" if neutral else "rv_30m"
        sigma3 = (merged[vol_col].to_numpy(float) * np.sqrt(3)).clip(0.0002, 0.05)
        cost_per_unit = RT_COST_BPS / 10_000 * (1.0 + np.abs(beta))
        edge = np.sign(mu) * np.maximum(np.abs(mu) - cost_per_unit, 0.0)
        notional = np.clip(0.10 * edge / sigma3**2, -0.0167, 0.0167) * STARTING_CAPITAL
        notional[np.abs(mu) <= cost_per_unit] = 0.0
        hedge_return = merged.spy_exec_return.to_numpy(float) if neutral else np.zeros(len(merged))
        gross_pnl = notional * (merged.exec_return.to_numpy(float) - beta * hedge_return)
        abs_turnover = np.abs(notional) * (1.0 + np.abs(beta))
        exits = pd.to_datetime(merged.exit_bar_time, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        trades = pd.DataFrame({"symbol": merged.symbol, "exit_day": exits, "notional": notional,
                               "gross_pnl": gross_pnl, "abs_turnover": abs_turnover,
                               "mu_bps": mu * 10_000, "realized_bps": (merged.resid_return if neutral else merged.exec_return).to_numpy(float) * 10_000})
        trades = trades.loc[trades.notional.ne(0)].copy()
        all_days = sorted(exits.unique())
        daily_gross = trades.groupby("exit_day").gross_pnl.sum().reindex(all_days, fill_value=0.0)
        sensitivities = {}
        for bps in (10, 15, 25, 35, 50):
            daily_net = trades.gross_pnl - trades.abs_turnover * bps / 10_000
            sensitivities[str(bps)] = float(daily_net.sum())
        bootstrap_net = trades.gross_pnl - trades.abs_turnover * RT_COST_BPS / 10_000
        daily_net = pd.Series(0.0, index=all_days)
        if len(trades):
            daily_net.loc[trades.exit_day.unique()] = trades.groupby("exit_day").apply(
                lambda g: float((g.gross_pnl - g.abs_turnover * RT_COST_BPS / 10_000).sum())).reindex(
                    trades.exit_day.unique()).to_numpy()
        arr = daily_net.to_numpy(float)
        monthly = trades.assign(month=trades.exit_day.str[:7], net=trades.gross_pnl-trades.abs_turnover*RT_COST_BPS/10_000).groupby("month").net.sum()
        by_symbol = trades.assign(net=trades.gross_pnl-trades.abs_turnover*RT_COST_BPS/10_000).groupby("symbol").net.sum().sort_values()
        rng = np.random.default_rng(1501)
        n, block, reps = len(arr), 5, 20000
        samples = np.empty(reps)
        if n:
            for j in range(reps):
                idx = []
                while len(idx) < n:
                    start = rng.integers(0, n)
                    idx.extend((start + k) % n for k in range(block))
                samples[j] = arr[np.asarray(idx[:n])].sum()
        diagnostics[name] = {
            "trades": int(len(trades)), "gross_pnl": float(trades.gross_pnl.sum()),
            "net_pnl_10bps": float(bootstrap_net.sum()), "cost_sensitivity_dollars": sensitivities,
            "gross_turnover": float(trades.abs_turnover.sum()), "max_single_trade_dollars": float(trades.notional.abs().max()) if len(trades) else 0.0,
            "block_bootstrap_5day_95pct_pnl": [float(x) for x in np.quantile(samples, [0.025, 0.975])] if n else [0.0, 0.0],
            "bootstrap_fraction_positive": float((samples > 0).mean()) if n else 0.0,
            "monthly_net_pnl_10bps": {str(k): float(v) for k, v in monthly.items()},
            "symbol_net_pnl_10bps": {str(k): float(v) for k, v in by_symbol.items()},
        }
        results.append({"model": name, "market_neutral": neutral, **portfolio})
    payload = {"status": "development_diagnostics_only", "replay_parameters": {
        "starting_capital": STARTING_CAPITAL, "gross_limit_multiple": GROSS_LIMIT,
        "round_trip_cost_bps_per_leg": RT_COST_BPS, "position_volatility": "stock volatility for raw forecasts; SPY-residual volatility for hedged forecasts",
        "fractional_kelly": 0.10, "per_tranche_cap_pct_nav": 1.67,
        "bootstrap": "20,000 five-trading-day circular block resamples of daily net P&L"},
        "corrected_portfolio_results": results, "candidate_diagnostics": diagnostics}
    path = REPORT_DIR / "replay_diagnostics.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
