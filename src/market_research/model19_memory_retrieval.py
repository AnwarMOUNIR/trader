"""Causal nearest-episode memory baseline for five-session return forecasts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler

from market_research.model17_daily import M15_SYMBOLS, M16_SYMBOLS, TRAINING_SYMBOLS, _make_rows
from market_research.model19_daily_factors import _simulate
from market_research.trend_experiment import ROOT

OUT = ROOT / "reports/model19"
FEATURES = ["mom5_volnorm", "mom20_volnorm", "resid_mom5", "resid_mom20", "spy_mom5",
            "spy_mom20", "spy_rv20", "rv20", "overnight_gap", "intraday_ret"]
K_VALUES = (25, 100, 400)


def _features(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURES].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)


def _predict(train: pd.DataFrame, query: pd.DataFrame, k: int) -> np.ndarray:
    scaler = StandardScaler().fit(_features(train))
    xt = np.clip(scaler.transform(_features(train)), -6, 6)
    xq = np.clip(scaler.transform(_features(query)), -6, 6)
    unique, idx = np.unique(xt, axis=0, return_index=True)
    y = train.iloc[idx].target_scaled.to_numpy(float)
    tree = cKDTree(unique, compact_nodes=True, balanced_tree=True)
    distance, indices = tree.query(xq, k=min(k, len(y)), workers=-1)
    if distance.ndim == 1:
        distance, indices = distance[:, None], indices[:, None]
    weights = 1.0 / np.maximum(distance, 0.05)
    value = (weights * y[indices]).sum(axis=1) / weights.sum(axis=1)
    horizon_vol = (query.rv20.to_numpy(float) * np.sqrt(5)).clip(0.003, 0.20)
    return value * horizon_vol


def _score_metrics(d: pd.DataFrame) -> dict:
    y = d.target_return.to_numpy(float)
    p = d.expected_return.to_numpy(float)
    return {"rows": int(len(d)), "auc_direction": float(roc_auc_score(y > 0, p)) if len(np.unique(y > 0)) == 2 else None,
            "spearman_ic": float(spearmanr(p, y).statistic),
            "rmse": float(np.sqrt(mean_squared_error(y, p))),
            "zero_rmse": float(np.sqrt(np.mean(y ** 2))),
            "mean_forecast_bps": float(np.mean(p) * 10000),
            "mean_actual_bps": float(np.mean(y) * 10000)}


def _calibrate(cal: pd.DataFrame, raw: np.ndarray) -> tuple[float, float]:
    x, y = np.asarray(raw, float), cal.target_return.to_numpy(float)
    slope = float(np.cov(x, y, ddof=0)[0, 1] / np.var(x)) if np.var(x) > 1e-16 else 0.0
    intercept = float(y.mean() - slope * x.mean())
    return intercept, slope


def _events(d: pd.DataFrame, daily: dict[str, pd.DataFrame], k: int) -> list[dict]:
    spy_daily = daily["SPY"].reset_index(drop=True).set_index("date", drop=False)
    events = []
    for date, g in d.groupby("date", sort=True):
        if len(g) < 2 * k:
            continue
        ordered = g.sort_values("expected_return", ascending=False)
        long_rows, short_rows = ordered.head(k), ordered.tail(k)
        rows = {r.symbol: r._asdict() for r in g.itertuples(index=False)}
        if date not in spy_daily.index:
            continue
        events.append({"date": str(date), "entry_date": str(long_rows.iloc[0].entry_date),
                       "picks": (long_rows.symbol.tolist(), short_rows.symbol.tolist()),
                       "rows": rows, "spy_row": spy_daily.loc[date].to_dict()})
    return events


def _strategy_result(d: pd.DataFrame, daily: dict[str, pd.DataFrame], k: int) -> dict:
    events = _events(d, daily, k)
    out = {"signals": len(events), "costs": {}}
    for cost in (10, 25):
        out["costs"][str(cost)] = {
            side: _simulate(events, daily, cost, side) for side in ("long_only", "long_short", "spy")
        }
    return out


def _top3_attribution(d: pd.DataFrame, daily: dict[str, pd.DataFrame], cost_bps: int = 10) -> dict:
    spy = daily["SPY"].reset_index(drop=True).set_index("date", drop=False)
    daily_rows = []
    ticker_pnl: dict[str, float] = {}
    monthly: dict[str, float] = {}
    for date, g in d.groupby("date", sort=True):
        picks = g.nlargest(3, "expected_return")
        if len(picks) < 3 or date not in spy.index:
            continue
        model_pnl = 0.0
        for r in picks.itertuples(index=False):
            pnl = (20_000 / 3) * (float(r.target_return) - cost_bps / 10_000)
            ticker_pnl[r.symbol] = ticker_pnl.get(r.symbol, 0.0) + pnl
            model_pnl += pnl
        spy_row = spy.loc[date]
        if isinstance(spy_row, pd.DataFrame):
            spy_row = spy_row.iloc[-1]
        spy_pnl = 20_000 * (float(spy_row.target_return) - cost_bps / 10_000)
        daily_rows.append((str(date), model_pnl, spy_pnl))
        month = str(date)[:7]
        monthly[month] = monthly.get(month, 0.0) + model_pnl
    a = np.asarray([[r[1], r[2]] for r in daily_rows], float)
    diff = a[:, 0] - a[:, 1] if len(a) else np.asarray([], float)
    rng = np.random.default_rng(19019)
    samples = []
    if len(diff):
        for _ in range(10_000):
            ix = []
            while len(ix) < len(diff):
                start = int(rng.integers(0, len(diff)))
                ix.extend((start + j) % len(diff) for j in range(5))
            samples.append(float(diff[np.asarray(ix[:len(diff)])].sum()))
    return {"signals": int(len(daily_rows)), "model_net_at_cost": float(a[:, 0].sum()) if len(a) else 0.0,
            "spy_net_same_dates": float(a[:, 1].sum()) if len(a) else 0.0,
            "active_difference": float(diff.sum()) if len(diff) else 0.0,
            "active_5day_block_bootstrap_95pct": [float(x) for x in np.quantile(samples, [0.025, 0.975])] if samples else [0.0, 0.0],
            "positive_signal_fraction": float((a[:, 0] > 0).mean()) if len(a) else 0.0,
            "monthly_model_net": monthly,
            "ticker_net_pnl_top10": dict(sorted(ticker_pnl.items(), key=lambda x: x[1], reverse=True)[:10]),
            "top_ticker_pnl_share": float(max(ticker_pnl.values()) / a[:, 0].sum()) if ticker_pnl and a[:, 0].sum() else None,
            "signal_event_pnl": daily_rows}


def run() -> dict:
    from market_research.validation import new_run_dir
    out_dir = new_run_dir(OUT)
    development_symbols = sorted(set(TRAINING_SYMBOLS + M15_SYMBOLS) - {"SPY"})
    dev, dev_daily = _make_rows(development_symbols, "development")
    dev = dev.copy()
    dev["date"] = dev.date.astype(str)
    dev["exit_date"] = dev.exit_date.astype(str)
    dev["target_scaled"] = dev.target_return / (dev.rv20 * np.sqrt(5)).clip(lower=0.003)
    train = dev.loc[(dev.date < "2025-01-01") & (dev.exit_date < "2025-01-01")].copy()
    cal = dev.loc[(dev.date >= "2025-01-15") & (dev.date < "2025-06-20")].copy()
    test = dev.loc[(dev.date >= "2025-07-10") & (dev.date < "2025-12-20")].copy()
    output = {"status": "development_only_previous_or_consumed_cohorts",
              "data_scope": "Model 17 development symbols; no Model 16 stock rows and no Model 18 files",
              "memory_key_features": FEATURES,
              "memory_values": "five-session outcome scaled by trailing 20-session daily volatility; memory labels excluded until exit_date < 2025-01-01",
              "fit_rows": int(len(train)), "calibration_rows": int(len(cal)), "test_rows": int(len(test)),
              "test_window": ["2025-07-10", "2025-12-20"], "results": {}}
    prediction_parts = []
    for k in K_VALUES:
        raw_cal = _predict(train, cal, k)
        raw_test = _predict(train, test, k)
        intercept, slope = _calibrate(cal, raw_cal)
        scored = test.copy()
        scored["expected_return"] = intercept + slope * raw_test
        prediction_parts.append(scored[["symbol", "date", "entry_date", "exit_date", "target_return", "expected_return"]].assign(cohort="model15_consumed_2025h2", k=k))
        output["results"][str(k)] = {
            "calibration_intercept": intercept, "calibration_slope": slope,
            "prediction_metrics": _score_metrics(scored),
            "portfolio_top3": _strategy_result(scored, dev_daily, 3),
            "top3_longonly_attribution_10bps": _top3_attribution(scored, dev_daily, 10),
            "portfolio_top10pct": _strategy_result(scored, dev_daily, max(3, round(len(development_symbols) * 0.10))),
        }
    # A transfer diagnostic on Model 16's previously consumed stocks. Calibrate
    # on Q1 2026, then evaluate April-October; these are not an unseen cohort.
    transfer, transfer_daily = _make_rows(M16_SYMBOLS, "model16", "2026")
    transfer["date"] = transfer.date.astype(str)
    transfer["exit_date"] = transfer.exit_date.astype(str)
    transfer["target_scaled"] = transfer.target_return / (transfer.rv20 * np.sqrt(5)).clip(lower=0.003)
    transfer_cal = transfer.loc[(transfer.date >= "2026-01-10") & (transfer.date < "2026-04-01")].copy()
    transfer_test = transfer.loc[(transfer.date >= "2026-04-15") & (transfer.date < "2026-10-20")].copy()
    output["transfer_model16_consumed_cohort"] = {"calibration_rows": int(len(transfer_cal)), "test_rows": int(len(transfer_test)), "results": {}}
    for k in K_VALUES:
        raw_cal = _predict(train, transfer_cal, k)
        raw_test = _predict(train, transfer_test, k)
        intercept, slope = _calibrate(transfer_cal, raw_cal)
        scored = transfer_test.copy()
        scored["expected_return"] = intercept + slope * raw_test
        prediction_parts.append(scored[["symbol", "date", "entry_date", "exit_date", "target_return", "expected_return"]].assign(cohort="model16_consumed_2026", k=k))
        output["transfer_model16_consumed_cohort"]["results"][str(k)] = {
            "calibration_intercept": intercept, "calibration_slope": slope,
            "prediction_metrics": _score_metrics(scored),
            "portfolio_top3": _strategy_result(scored, transfer_daily, 3),
            "top3_longonly_attribution_10bps": _top3_attribution(scored, transfer_daily, 10),
        }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "memory_retrieval.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    pd.concat(prediction_parts, ignore_index=True).to_parquet(out_dir / "memory_retrieval_predictions.parquet", index=False)
    return output


if __name__ == "__main__":
    result = run()
    print(f"Wrote nearest-episode memory results under {OUT / 'runs'}")
