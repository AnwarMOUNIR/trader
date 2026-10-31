"""Development experiment: five-session trend model with a daily staggered book.

Only previously consumed development cohorts are used here. The Model 16
unseen-stock cohort is a development diagnostic for this new model, so a
separate lockbox would still be required before promotion.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error
from market_research.daily_data import prepare_daily
from market_research.resample import get_session_cache
from market_research.portfolio import simulate_ledger

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/model17"
M15_LOCK_DATA = ROOT / "data/challenges/model15/lockbox/candles/30min"
M16_LOCK_DATA = ROOT / "data/challenges/model16/lockbox/candles/30min"
TRAINING_SYMBOLS = json.loads((ROOT / "reports/model15/protocol.json").read_text())["training_symbols"]
M15_SYMBOLS = [x["ticker"] for x in json.loads((ROOT / "reports/model15/lockbox_protocol.json").read_text())["symbols"]]
M16_SYMBOLS = [x["ticker"] for x in json.loads((ROOT / "reports/model16/lockbox_protocol.json").read_text())["symbols"]]

FEATURES = [
    "mom1", "mom5", "mom10", "mom20", "mom60", "overnight_gap", "intraday_ret",
    "range1", "close_location", "rv5", "rv20", "rv60", "vol_rel20",
    "mom5_volnorm", "mom20_volnorm", "beta60", "resid_mom5", "resid_mom20",
    "spy_mom1", "spy_mom5", "spy_mom20", "spy_rv20",
    "cs_mom5_rank", "cs_mom20_rank", "cs_resid20_rank",
]


def _daily(path: Path) -> pd.DataFrame:
    bars = pd.read_parquet(path).sort_values("timestamp_utc").copy()
    bars["timestamp_utc"] = pd.to_datetime(bars.timestamp_utc, utc=True)
    if "session_date" not in bars:
        bars["session_date"] = bars.timestamp_utc.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
    grouped = bars.groupby("session_date", sort=True)
    daily = grouped.agg(
        date=("session_date", "first"), timestamp_utc=("timestamp_utc", "last"),
        open=("open", "first"), high=("high", "max"), low=("low", "min"),
        close=("close", "last"), volume=("volume", "sum"),
    ).reset_index(drop=True)
    sessions = get_session_cache()
    completeness = {}
    for date, g in grouped:
        bounds = sessions.get(str(date))
        if bounds is None:
            raise ValueError(f"Non-session daily input: {date}")
        opn, end = bounds
        expected = pd.date_range(opn, end, freq="30min", inclusive="left").tz_convert("UTC")
        observed = pd.DatetimeIndex(g.timestamp_utc)
        completeness[date] = (observed.equals(expected) and "is_complete" in g
                              and bool(g.is_complete.fillna(False).all()))
    daily["session_complete"] = daily.date.map(completeness).fillna(False)
    return daily


def _combine(paths: list[Path]) -> pd.DataFrame:
    frames = [_daily(path) for path in paths if path.exists()]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def _source_paths(symbol: str) -> list[Path]:
    paths = []
    for period in ("pre", "mid", "post"):
        if (symbol == "META" and period in {"pre", "mid"}) or (symbol == "FB" and period == "post"):
            continue
        paths.append(ROOT / f"data/challenges/model14/train/{period}/candles/30min/{symbol}_30min.parquet")
        if period == "mid":
            paths[-1] = ROOT / f"data/challenges/model15/train/mid/candles/30min/{symbol}_30min.parquet"
    return paths


def _make_rows(symbols: list[str], source: str, score_period: str | None = None,
               *, actions_by_symbol: dict | None = None, require_labels: bool = True):
    """Feature eligibility is past-only; label filtering is an explicit research step.

    Optional actions map ticker to dated event records consumed by prepare_daily.
    This is conservative quarantine, not certified total-return reconstruction.
    """
    actions_by_symbol = actions_by_symbol or {}
    if source == "development":
        files = {s: _source_paths(s) for s in TRAINING_SYMBOLS if s != "SPY"}
        files.update({s: [M15_LOCK_DATA / f"{s}_30min.parquet"] for s in M15_SYMBOLS})
    elif source == "model16":
        files = {s: [M16_LOCK_DATA / f"{s}_30min.parquet"] for s in symbols}
    else:
        raise ValueError(source)

    spy_paths = [p for p in _source_paths("SPY") if p.exists()]
    if source == "development":
        spy_paths.append(M15_LOCK_DATA / "SPY_30min.parquet")
    else:
        spy_paths = [M16_LOCK_DATA / "SPY_30min.parquet"]
    spy = prepare_daily(_combine(spy_paths), "SPY", actions_by_symbol.get("SPY"))
    spy["spy_ret1"] = spy.close.pct_change(1, fill_method=None)
    for lag in (5, 20):
        spy[f"spy_mom{lag}"] = spy.close / spy.close.shift(lag) - 1
    spy["spy_rv20"] = spy.spy_ret1.rolling(20, min_periods=12).std()
    spy_cols = ["date", "spy_ret1", "spy_mom1", "spy_mom5", "spy_mom20", "spy_rv20"]
    spy["spy_mom1"] = spy.spy_ret1
    market = spy[spy_cols].drop_duplicates("date", keep="last")

    rows = []
    daily_by_symbol = {}
    daily_by_symbol["SPY"] = spy.set_index("date", drop=False)
    for symbol, paths in files.items():
        if symbol not in symbols:
            continue
        d = _combine(paths)
        if d.empty:
            continue
        d = prepare_daily(d, symbol, actions_by_symbol.get(symbol))
        d["symbol"] = symbol
        d["ret1"] = d.close.pct_change(1, fill_method=None)
        for lag, name in ((1, "mom1"), (5, "mom5"), (10, "mom10"), (20, "mom20"), (60, "mom60")):
            d[name] = d.close / d.close.shift(lag) - 1
        d["overnight_gap"] = d.open / d.close.shift(1) - 1
        d["intraday_ret"] = d.close / d.open - 1
        d["range1"] = (d.high - d.low) / d.open
        d["close_location"] = (d.close - d.low) / (d.high - d.low).replace(0, np.nan)
        for lag in (5, 20, 60):
            d[f"rv{lag}"] = d.ret1.rolling(lag, min_periods=max(5, lag // 2)).std()
        d["vol_rel20"] = np.log1p(d.volume) - np.log1p(d.volume.shift(1).rolling(20, min_periods=10).median())
        d = d.merge(market, on="date", how="left", validate="many_to_one")
        cov = d.ret1.rolling(60, min_periods=25).cov(d.spy_ret1)
        var = d.spy_ret1.rolling(60, min_periods=25).var().replace(0, np.nan)
        d["beta60"] = (cov / var).clip(-3, 3).fillna(1.0)
        d["resid_mom5"] = d.mom5 - d.beta60 * d.spy_mom5
        d["resid_mom20"] = d.mom20 - d.beta60 * d.spy_mom20
        d["mom5_volnorm"] = d.mom5 / (d.rv20 * np.sqrt(5)).clip(lower=0.001)
        d["mom20_volnorm"] = d.mom20 / (d.rv20 * np.sqrt(20)).clip(lower=0.002)
        daily_by_symbol[symbol] = d.set_index("date", drop=False)
        rows.append(d)
    all_rows = pd.concat(rows, ignore_index=True)
    # Rank only currently eligible stocks, before consulting any future labels.
    all_rows = all_rows.loc[all_rows.feature_eligible & all_rows.date.isin(
        set(spy.loc[spy.feature_eligible, "date"]))].copy()
    for source_col, dest in (("mom5_volnorm", "cs_mom5_rank"), ("mom20_volnorm", "cs_mom20_rank"),
                             ("resid_mom20", "cs_resid20_rank")):
        all_rows[dest] = all_rows.groupby("date", sort=False)[source_col].rank(pct=True) - 0.5
    valid = all_rows[FEATURES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    if require_labels:
        valid &= all_rows.label_eligible
    all_rows = all_rows.loc[valid].copy()
    if score_period == "2025h2":
        all_rows = all_rows.loc[(all_rows.date >= "2025-07-10") & (all_rows.date < "2025-12-20")]
    elif score_period == "2026":
        all_rows = all_rows.loc[(all_rows.date >= "2026-01-10") & (all_rows.date < "2026-10-20")]
    return all_rows, daily_by_symbol


def _metrics(d: pd.DataFrame, label: str) -> dict:
    y = d.target_return.to_numpy(float)
    mu = d.expected_return.to_numpy(float)
    by_date = d.assign(mu=mu, y=y).groupby("date", sort=True)
    daily_ic = [spearmanr(g.mu, g.y).statistic for _, g in by_date if len(g) >= 10]
    return {
        "rows": int(len(d)), "positive_rate": float((y > 0).mean()),
        "auc": float(roc_auc_score(y > 0, mu)) if len(np.unique(y > 0)) == 2 else None,
        "row_rank_ic": float(spearmanr(mu, y).statistic),
        "mean_daily_cross_sectional_rank_ic": float(np.nanmean(daily_ic)),
        "rmse": float(np.sqrt(mean_squared_error(y, mu))),
        "zero_forecast_rmse": float(np.sqrt(np.mean(y ** 2))),
        "mean_expected_return_bps": float(mu.mean() * 10000),
        "mean_actual_return_bps": float(y.mean() * 10000),
        "top_forecast_decile_actual_bps": float(d.loc[d.groupby("date").expected_return.rank(pct=True) >= .90, "target_return"].mean() * 10000),
    }


def _book(d: pd.DataFrame, daily: dict[str, pd.DataFrame], threshold_bps: float, cost_bps: float = 10.0) -> dict:
    """One $20k long tranche per day, held for five sessions; at most 1x gross."""
    selections = []
    for date, g in d.groupby("date", sort=True):
        top = g.sort_values("expected_return", ascending=False).iloc[0]
        if top.expected_return * 10000 >= threshold_bps:
            selections.append(top)
    entries = [{"symbol": str(r.symbol), "entry_date": str(r.entry_date),
                "exit_date": str(r.exit_date), "entry_price": float(r.entry_open),
                "notional": 20000.0} for r in selections]
    result = simulate_ledger(entries, daily, capital=100000.0, cost_bps=cost_bps,
                             max_entry_gross_fraction=1.0)
    result["round_trip_cost_bps"] = cost_bps
    result["threshold_bps"] = threshold_bps
    return result


def run() -> dict:
    from market_research.validation import new_run_dir
    out_dir = new_run_dir(OUT)
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_symbols = sorted(set(TRAINING_SYMBOLS + M15_SYMBOLS) - {"SPY"})
    dev, _ = _make_rows(dev_symbols, "development")
    # Strict chronological split with five sessions removed at each boundary.
    fit = dev.loc[(dev.date < "2024-12-20") & (dev.exit_date < "2025-01-15")].copy()
    cal = dev.loc[(dev.date >= "2025-01-15") & (dev.date < "2025-06-20") & (dev.exit_date < "2025-07-10")].copy()
    test = dev.loc[(dev.date >= "2025-07-10") & (dev.date < "2025-12-20")].copy()
    params = dict(loss="squared_error", max_iter=120, learning_rate=0.05, max_leaf_nodes=15,
                  min_samples_leaf=300, l2_regularization=20.0, early_stopping=False, random_state=17017)
    model = HistGradientBoostingRegressor(**params)
    model.fit(fit[FEATURES], fit.target_return.to_numpy(float))
    raw_cal = model.predict(cal[FEATURES]); ycal = cal.target_return.to_numpy(float)
    den = float(np.sum((raw_cal - raw_cal.mean()) ** 2))
    slope = float(np.sum((raw_cal - raw_cal.mean()) * (ycal - ycal.mean())) / den) if den > 1e-16 else 0.0
    intercept = float(ycal.mean() - slope * raw_cal.mean())
    test = test.copy(); test["expected_return"] = intercept + slope * model.predict(test[FEATURES])
    test_metrics = _metrics(test, "2025h2")
    test_daily = _make_rows(M15_SYMBOLS, "development", "2025h2")[1]
    dev_books = {str(th): {str(cost): _book(test, test_daily, th, cost) for cost in (10, 25)}
                 for th in (10, 25, 50)}
    spy_test = test_daily["SPY"].reset_index(drop=True)
    spy_test = spy_test.loc[spy_test.date.isin(set(test.date))].copy()
    spy_test["symbol"] = "SPY"
    spy_test["expected_return"] = 1.0
    dev_books["SPY_reference"] = {str(cost): _book(spy_test, test_daily, 0.0, cost) for cost in (10, 25)}

    # A second development diagnostic transfers the 2025-calibrated candidate to
    # the previously consumed Model 16 cohort, which is not a lockbox for Model 17.
    m16, m16_daily = _make_rows(M16_SYMBOLS, "model16", "2026")
    m16["expected_return"] = intercept + slope * model.predict(m16[FEATURES])
    m16_metrics = _metrics(m16, "2026") if len(m16) else {"rows": 0}
    m16_books = {str(th): {str(cost): _book(m16, m16_daily, th, cost) for cost in (10, 25)}
                 for th in (10, 25, 50)}
    spy_m16 = m16_daily["SPY"].reset_index(drop=True)
    spy_m16 = spy_m16.loc[spy_m16.date.isin(set(m16.date))].copy()
    spy_m16["symbol"] = "SPY"
    spy_m16["expected_return"] = 1.0
    m16_books["SPY_reference"] = {str(cost): _book(spy_m16, m16_daily, 0.0, cost) for cost in (10, 25)}

    result = {
        "status": "development_candidate_only_not_frozen",
        "return_basis": "raw prices; optional known-action quarantine; action coverage not certified",
        "target": "five-session return from next session open to fifth session close",
        "fit_window_max_feature_date": str(fit.date.max()), "fit_rows": int(len(fit)),
        "calibration_window": [str(cal.date.min()), str(cal.date.max())], "calibration_rows": int(len(cal)),
        "calibration_intercept": intercept, "calibration_slope": slope,
        "test_window": [str(test.date.min()), str(test.date.max())], "test_metrics": test_metrics,
        "test_books": dev_books,
        "Model16_consumed_cohort_2026_diagnostics": {"metrics": m16_metrics, "books": m16_books},
        "features": FEATURES, "model_parameters": params,
        "training_stocks": dev_symbols, "training_stock_count": len(dev_symbols),
        "lockbox_stock_cohorts_excluded_from_fit": M16_SYMBOLS,
    }
    (out_dir / "development_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    from joblib import dump
    dump({"estimator": model, "features": FEATURES, "calibration_intercept": intercept,
          "calibration_slope": slope, "model_parameters": params}, out_dir / "model_17_dev.joblib")
    test[["symbol", "date", "entry_date", "exit_date", "target_return", "expected_return"]].to_parquet(
        out_dir / "development_predictions.parquet", index=False)
    m16[["symbol", "date", "entry_date", "exit_date", "target_return", "expected_return"]].to_parquet(
        out_dir / "model16_development_predictions.parquet", index=False)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
