"""Leakage-audited development experiments for Model 15.

This module uses only the existing Model 14 training cohort.  The consumed
Model 14 evaluation cohort is deliberately excluded.  Signals are formed after
a completed 30-minute bar, entered at the next bar open, and closed after a
90-minute holding interval.  Results remain research estimates, not a live
trading claim.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from market_research.trend_experiment import FEATURES, make_rows

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports/model15"
PROTOCOL_PATH = ROOT / "reports/model14/protocol.json"
STARTING_CAPITAL = 100_000.0
GROSS_LIMIT = 1.0
RT_COST_BPS = 10.0
HORIZON = 3
FEATURES_NEW = FEATURES + ["regime_mid", "regime_post", "rv_30m", "r1_volnorm", "r3_volnorm", "r6_volnorm",
                           "range_volnorm", "body_volnorm", "vwap_volnorm", "beta_spy", "resid_r1",
                           "resid_r3", "cs_r1_rank", "cs_r3_rank", "cs_vwap_rank"]


def _period_root(period: str) -> Path:
    if period == "mid":
        return ROOT / "data/challenges/model15/train/mid"
    return ROOT / f"data/challenges/model14/train/{period}"


def _market(period: str) -> pd.DataFrame:
    path = _period_root(period) / "candles/30min/SPY_30min.parquet"
    spy = pd.read_parquet(path).sort_values("timestamp_utc").copy()
    g = spy.groupby("session_date", sort=False)
    spy["spy_r1"] = g.close.pct_change(1, fill_method=None)
    spy["spy_r3"] = g.close.pct_change(3, fill_method=None)
    spy["spy_range1"] = (spy.high - spy.low) / spy.open
    spy["spy_entry_open"] = g.open.shift(-1)
    spy["spy_exit_close"] = g.close.shift(-HORIZON)
    spy["spy_exec_return"] = spy.spy_exit_close / spy.spy_entry_open - 1.0
    spy["timestamp_utc"] = pd.to_datetime(spy.timestamp_utc, utc=True)
    return spy[["timestamp_utc", "spy_r1", "spy_r3", "spy_range1", "spy_entry_open", "spy_exec_return"]]


def _rows(period: str, symbols: list[str]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    frames, bars_by_symbol = [], {}
    root = _period_root(period) / "candles/30min"
    market = _market(period)
    for symbol in symbols:
        # FB/META are one issuer, not two independent samples. Use the valid
        # point-in-time ticker for each window to avoid doubling that issuer.
        if (symbol == "META" and period in {"pre", "mid"}) or (symbol == "FB" and period == "post"):
            continue
        path = root / f"{symbol}_30min.parquet"
        if not path.exists():
            continue
        bars = pd.read_parquet(path).sort_values("timestamp_utc").copy()
        bars["timestamp_utc"] = pd.to_datetime(bars.timestamp_utc, utc=True)
        g = bars.groupby("session_date", sort=False)
        bars["entry_open"] = g.open.shift(-1)
        bars["exit_close"] = g.close.shift(-HORIZON)
        bars["entry_time"] = g.timestamp_utc.shift(-1)
        bars["exit_bar_time"] = g.timestamp_utc.shift(-HORIZON)
        bars["exec_return"] = bars.exit_close / bars.entry_open - 1.0
        # This remains a close-to-close conditional feature.  Rolling volatility
        # includes only completed bars and never crosses an overnight return.
        r1 = g.close.pct_change(1, fill_method=None)
        rv = r1.rolling(65, min_periods=20).std()
        rv.index = bars.index
        bars["rv_30m"] = rv
        spy_by_time = market.set_index("timestamp_utc")
        spy_r1 = bars.timestamp_utc.map(spy_by_time.spy_r1)
        spy_r3 = bars.timestamp_utc.map(spy_by_time.spy_r3)
        spy_future = bars.timestamp_utc.map(spy_by_time.spy_exec_return)
        stock_r1 = g.close.pct_change(1, fill_method=None)
        cov = stock_r1.rolling(65, min_periods=30).cov(spy_r1)
        var = spy_r1.rolling(65, min_periods=30).var()
        beta = (cov / var.replace(0, np.nan)).clip(-3.0, 3.0).fillna(1.0)
        bars["beta_spy"] = beta
        resid_r1 = stock_r1 - beta * spy_r1
        resid_vol = resid_r1.rolling(65, min_periods=30).std()
        bars["rv_resid_30m"] = resid_vol.fillna(rv)
        bars_by_symbol[symbol] = bars.set_index("timestamp_utc", drop=False)
        feats = make_rows(path, market=market, deadzone=0.0)
        if feats.empty:
            continue
        extra = bars[["timestamp_utc", "entry_open", "exit_close", "entry_time", "exit_bar_time", "exec_return", "rv_30m", "rv_resid_30m", "beta_spy"]].copy()
        extra["spy_exec_return"] = spy_future.to_numpy()
        extra["spy_entry_open"] = bars.timestamp_utc.map(spy_by_time.spy_entry_open).to_numpy()
        extra["resid_r1"] = resid_r1.to_numpy()
        extra["resid_r3"] = feats.set_index("timestamp_utc").r3.reindex(extra.timestamp_utc).to_numpy() - beta.to_numpy() * spy_r3.to_numpy()
        feats = feats.merge(extra, on="timestamp_utc", how="inner", validate="one_to_one")
        # Scale price movement features by trailing, completed-bar volatility.
        den = feats.rv_30m.clip(lower=1e-5)
        feats["r1_volnorm"] = feats.r1 / den
        feats["r3_volnorm"] = feats.r3 / (den * np.sqrt(3.0))
        feats["r6_volnorm"] = feats.r6 / (den * np.sqrt(6.0))
        feats["range_volnorm"] = feats.range1 / den
        feats["body_volnorm"] = feats.body1 / den
        feats["vwap_volnorm"] = feats.vwap_gap / den
        feats["regime_post"] = float(period == "post")
        feats["regime_mid"] = float(period == "mid")
        good = feats[[f for f in FEATURES_NEW if not f.startswith("cs_")]].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
        good &= feats.exec_return.notna() & feats.entry_open.gt(0) & feats.exit_close.gt(0)
        feats = feats.loc[good].copy()
        feats["period"] = period
        feats["resid_return"] = feats.exec_return - feats.beta_spy * feats.spy_exec_return
        feats["label_raw"] = (feats.exec_return > 0).astype(int)
        feats["label_exec"] = (feats.resid_return > 0).astype(int)
        frames.append(feats)
    if not frames:
        raise RuntimeError(f"No development rows for {period}")
    combined = pd.concat(frames, ignore_index=True)
    for source, dest in (("r1_volnorm", "cs_r1_rank"), ("r3_volnorm", "cs_r3_rank"), ("vwap_volnorm", "cs_vwap_rank")):
        combined[dest] = combined.groupby("timestamp_utc", sort=False)[source].rank(pct=True) - 0.5
    combined = combined.loc[combined[FEATURES_NEW].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)].copy()
    spy_path = root / "SPY_30min.parquet"
    spy_bars = pd.read_parquet(spy_path).sort_values("timestamp_utc").copy()
    spy_bars["timestamp_utc"] = pd.to_datetime(spy_bars.timestamp_utc, utc=True)
    bars_by_symbol["SPY"] = spy_bars.set_index("timestamp_utc", drop=False)
    return combined, bars_by_symbol


def _split(data: pd.DataFrame, era: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(data.timestamp_utc, utc=True)
    if era == "pre":
        train = data.loc[ts < "2018-01-01"].copy()
        cal = data.loc[(ts >= "2018-01-01") & (ts < "2018-07-01")].copy()
        test = data.loc[(ts >= "2018-07-01") & (ts < "2019-01-01")].copy()
    else:
        train = data.loc[ts < "2024-01-01"].copy()
        cal = data.loc[(ts >= "2024-01-01") & (ts < "2024-07-01")].copy()
        test = data.loc[(ts >= "2024-07-01") & (ts < "2025-01-01")].copy()
    # Labels overlap for three bars. Drop a one-session boundary band from both
    # sides so no event window can straddle a fit/calibration/test cut.
    for frame in (train, cal, test):
        frame["timestamp_utc"] = pd.to_datetime(frame.timestamp_utc, utc=True)
    if era == "pre":
        train = train.loc[train.timestamp_utc < "2017-12-29"]
        cal = cal.loc[cal.timestamp_utc < "2018-06-29"]
    else:
        train = train.loc[train.timestamp_utc < "2023-12-29"]
        cal = cal.loc[cal.timestamp_utc < "2024-06-28"]
    return train, cal, test


def _calibrate_prob(model, cal: pd.DataFrame, test: pd.DataFrame, label_col: str):
    raw_cal = np.clip(model.predict_proba(cal[FEATURES_NEW])[:, 1], 1e-6, 1 - 1e-6)
    raw_test = np.clip(model.predict_proba(test[FEATURES_NEW])[:, 1], 1e-6, 1 - 1e-6)
    logit_cal = np.log(raw_cal / (1 - raw_cal)).reshape(-1, 1)
    logit_test = np.log(raw_test / (1 - raw_test)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=500).fit(logit_cal, cal[label_col].astype(int))
    return calibrator.predict_proba(logit_test)[:, 1], calibrator


def _metrics(y, p, realized, baseline_rate: float):
    y, p, realized = np.asarray(y, int), np.clip(np.asarray(p, float), 1e-8, 1 - 1e-8), np.asarray(realized, float)
    prior = np.full(len(y), float(np.clip(baseline_rate, 1e-8, 1 - 1e-8)))
    gross_long_short = np.mean(np.sign(p - 0.5) * realized)
    return {
        "n": int(len(y)), "positive_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "constant_prior_log_loss": float(log_loss(y, prior, labels=[0, 1])),
        "gross_signed_return_bps_per_event": float(gross_long_short * 10_000),
        "mean_abs_realized_return_bps": float(np.mean(np.abs(realized)) * 10_000),
    }


def _portfolio(test: pd.DataFrame, mu: np.ndarray, bars: dict[str, pd.DataFrame],
               label: str, market_neutral: bool = True, kelly: float = 0.10, asset_cap: float = 0.0167) -> dict:
    """Conservative 90-minute tranche simulation with a 1x gross cap.

    Expected return must exceed the explicit round-trip cost. The position is
    fractional-Kelly sized from trailing volatility, capped per tranche, then
    uniformly scaled to remaining portfolio gross capacity.
    """
    d = test.copy().reset_index(drop=True)
    d["mu"] = np.asarray(mu, float)
    vol_col = "rv_resid_30m" if market_neutral else "rv_30m"
    sigma3 = (d[vol_col].to_numpy(float) * np.sqrt(3)).clip(0.0002, 0.05)
    cost = RT_COST_BPS / 10_000
    beta = d.beta_spy.to_numpy(float) if market_neutral else np.zeros(len(d))
    d["position_beta"] = beta
    per_unit_cost = cost * (1.0 + np.abs(beta))
    edge = np.sign(d.mu.to_numpy()) * np.maximum(np.abs(d.mu.to_numpy()) - per_unit_cost, 0.0)
    d["desired"] = np.clip(kelly * edge / (sigma3 ** 2), -asset_cap, asset_cap) * STARTING_CAPITAL
    d.loc[np.abs(d.mu.to_numpy()) <= cost, "desired"] = 0.0
    d["entry_time"] = pd.to_datetime(d.entry_time, utc=True)
    d["exit_bar_time"] = pd.to_datetime(d.exit_bar_time, utc=True)
    d = d.sort_values(["entry_time", "symbol"]).reset_index(drop=True)
    # Mark prices are indexed by candle start. A trade is marked at each
    # completed candle and exits at its third completed bar's close.
    tmin = pd.to_datetime(d.timestamp_utc, utc=True).min() - pd.Timedelta(minutes=30)
    tmax = pd.to_datetime(d.timestamp_utc, utc=True).max() + pd.Timedelta(hours=2)
    times = sorted({t for df in bars.values() for t in df.index if tmin <= t <= tmax})
    entrants_by_time = {t: group for t, group in d.groupby("entry_time", sort=False)}
    bars_idx = {s: {t: float(px) for t, px in df.close.items()} for s, df in bars.items()}
    active = []
    equity, peak, mdd, max_gross = STARTING_CAPITAL, STARTING_CAPITAL, 0.0, 0.0
    costs_paid = 0.0
    trade_pnls = []
    for t in times:
        entrants = entrants_by_time.get(t)
        if entrants is None:
            entrants = d.iloc[0:0]
        current_gross = sum(abs(x["notional"]) + abs(x["hedge_notional"]) for x in active)
        desired_total = float((entrants.desired.abs() * (1.0 + entrants.position_beta.abs())).sum())
        scale = min(1.0, max(0.0, STARTING_CAPITAL * GROSS_LIMIT - current_gross) / desired_total) if desired_total else 0.0
        for row in entrants.itertuples(index=False):
            notional = float(row.desired) * scale
            if notional == 0 or row.symbol not in bars_idx:
                continue
            b = bars_idx[row.symbol]
            if t not in b or not np.isfinite(row.entry_open) or row.entry_open <= 0:
                continue
            hedge_notional = -float(row.position_beta) * notional
            spy_entry_open = bars_idx["SPY"].get(t)
            if not np.isfinite(spy_entry_open) or spy_entry_open <= 0:
                continue
            fee = (abs(notional) + abs(hedge_notional)) * (RT_COST_BPS / 20_000)
            costs_paid += fee
            equity -= fee
            active.append({"symbol": row.symbol, "notional": notional, "entry_open": float(row.entry_open),
                           "hedge_notional": hedge_notional, "spy_entry_open": float(spy_entry_open),
                           "exit_bar_time": row.exit_bar_time, "entry_time": t, "entry_fee": fee})
        close_t = t + pd.Timedelta(minutes=30)
        gross = 0.0
        keep = []
        for pos in active:
            b = bars_idx[pos["symbol"]]
            if t not in b:
                keep.append(pos)
                gross += abs(pos["notional"]) + abs(pos["hedge_notional"])
                continue
            px = b[t]
            spy_px = bars_idx["SPY"].get(t)
            if spy_px is None:
                keep.append(pos)
                gross += abs(pos["notional"]) + abs(pos["hedge_notional"])
                continue
            unreal = pos["notional"] * (px / pos["entry_open"] - 1.0) + pos["hedge_notional"] * (spy_px / pos["spy_entry_open"] - 1.0)
            if pos["exit_bar_time"] == t:
                exit_fee = (abs(pos["notional"]) + abs(pos["hedge_notional"])) * (RT_COST_BPS / 20_000)
                costs_paid += exit_fee
                equity += unreal - exit_fee
                trade_pnls.append(unreal - pos["entry_fee"] - exit_fee)
            else:
                # Equity includes marked-to-market P&L for live positions.
                keep.append(pos)
                gross += abs(pos["notional"]) + abs(pos["hedge_notional"])
        active = keep
        max_gross = max(max_gross, current_gross + desired_total*scale)
        # Re-mark portfolio to current completed-bar prices.
        unrealized = 0.0
        for pos in active:
            b = bars_idx[pos["symbol"]]
            if t in b:
                spy_px = bars_idx["SPY"].get(t)
                if spy_px is not None:
                    unrealized += pos["notional"] * (b[t] / pos["entry_open"] - 1.0) + pos["hedge_notional"] * (spy_px / pos["spy_entry_open"] - 1.0)
        mark_equity = equity + unrealized
        peak = max(peak, mark_equity)
        mdd = max(mdd, peak - mark_equity)
    return {"policy": label, "trades": len(trade_pnls), "net_pnl_dollars": float(equity - STARTING_CAPITAL),
            "ending_equity_dollars": float(equity), "max_drawdown_dollars": float(mdd),
            "max_drawdown_pct_start": float(100*mdd/STARTING_CAPITAL), "max_gross_dollars": float(max_gross),
            "total_costs_dollars": float(costs_paid), "mean_closed_trade_pnl": float(np.mean(trade_pnls)) if trade_pnls else 0.0}


def run() -> dict:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    symbols = [s for s in protocol["training_symbols"] if s != "SPY"]
    results, fold_data, prediction_frames = [], {}, []
    print(f"Preparing development cohorts ({len(symbols)} symbols)...", flush=True)
    cohorts = {period: _rows(period, symbols) for period in ("pre", "mid", "post")}
    for period in ("pre", "post"):
        if period == "pre":
            rows, bars = cohorts["pre"]
        else:
            rows = pd.concat([cohorts[p][0] for p in ("pre", "mid", "post")], ignore_index=True)
            bars = cohorts["post"][1]
        print(f"Preparing {period} chronological evaluation ({len(rows):,} rows)...", flush=True)
        train, cal, test = _split(rows, period)
        print(f"{period}: fit={len(train):,}, calibration={len(cal):,}, test={len(test):,}", flush=True)
        xtr, xcal, xtest = train[FEATURES_NEW], cal[FEATURES_NEW], test[FEATURES_NEW]
        hgb_classifier = lambda: HistGradientBoostingClassifier(max_iter=60, learning_rate=0.05, max_leaf_nodes=7,
            min_samples_leaf=200, l2_regularization=20.0, early_stopping=False, random_state=20261028)
        classifier_specs = [
            ("logistic_residual", make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=500)), "label_exec", "resid_return", True),
            ("hist_classifier_raw", hgb_classifier(), "label_raw", "exec_return", False),
            ("hist_classifier_residual", hgb_classifier(), "label_exec", "resid_return", True),
        ]
        for name, model, label_col, return_col, neutral in classifier_specs:
            print(f"{period}: fitting classifier {name}", flush=True)
            model.fit(xtr, train[label_col].astype(int))
            p, _ = _calibrate_prob(model, cal, test, label_col)
            baseline_rate = float(cal[label_col].mean())
            metrics = _metrics(test[label_col], p, test[return_col], baseline_rate)
            conditional_up = float(cal.loc[cal[label_col].eq(1), return_col].mean())
            conditional_down = float(cal.loc[cal[label_col].eq(0), return_col].mean())
            mu = p * conditional_up + (1-p) * conditional_down
            pred = test[["symbol", "period", "timestamp_utc", "entry_time", "exit_bar_time", "exec_return", "resid_return", "beta_spy",
                         "label_raw", "label_exec", "rv_30m", "rv_resid_30m", "slot_sin", "slot_cos", "spy_r1", "spy_r3", "spy_range1"]].copy()
            pred["era"], pred["model"] = period, name
            pred["probability_up"], pred["expected_return"] = p, mu
            prediction_frames.append(pred)
            port = _portfolio(test, mu, bars, name, market_neutral=neutral)
            results.append({"era": period, "model": name, "target": return_col, "prediction": metrics, "portfolio": port})
        regressor_specs = [
            ("ridge_residual", make_pipeline(StandardScaler(), Ridge(alpha=100.0)), "resid_return", True),
            ("hist_return_raw", HistGradientBoostingRegressor(loss="squared_error", max_iter=60, learning_rate=0.05,
                max_leaf_nodes=7, min_samples_leaf=200, l2_regularization=20.0, early_stopping=False,
                random_state=20261028), "exec_return", False),
            ("hist_return_residual", HistGradientBoostingRegressor(loss="squared_error", max_iter=60, learning_rate=0.05,
                max_leaf_nodes=7, min_samples_leaf=200, l2_regularization=20.0, early_stopping=False,
                random_state=20261028), "resid_return", True),
        ]
        for name, model, return_col, neutral in regressor_specs:
            print(f"{period}: fitting return model {name}", flush=True)
            ytr_ret = train[return_col].to_numpy(float)
            ycal_ret = cal[return_col].to_numpy(float)
            model.fit(xtr, ytr_ret)
            pcal = np.asarray(model.predict(xcal), float)
            ptest = np.asarray(model.predict(xtest), float)
            # Calibration is a disjoint, earlier chronological segment.
            den = float(np.dot(pcal-pcal.mean(), pcal-pcal.mean()))
            slope = float(np.dot(pcal-pcal.mean(), ycal_ret-ycal_ret.mean()) / den) if den > 1e-16 else 0.0
            intercept = float(ycal_ret.mean() - slope*pcal.mean())
            mu = intercept + slope*ptest
            pred = test[["symbol", "period", "timestamp_utc", "entry_time", "exit_bar_time", "exec_return", "resid_return", "beta_spy",
                         "label_raw", "label_exec", "rv_30m", "rv_resid_30m", "slot_sin", "slot_cos", "spy_r1", "spy_r3", "spy_range1"]].copy()
            pred["era"], pred["model"] = period, name
            pred["raw_return_forecast"] = ptest
            pred["expected_return"] = mu
            prediction_frames.append(pred)
            corr = float(pd.Series(mu).corr(pd.Series(test[return_col].to_numpy(float)), method="spearman"))
            port = _portfolio(test, mu, bars, name, market_neutral=neutral)
            results.append({"era": period, "model": name, "prediction": {
                "n": int(len(test)), "spearman_rank_ic": corr,
                "mean_forecast_bps": float(np.mean(mu)*10_000),
                "mean_realized_bps": float(test[return_col].mean()*10_000),
                "mean_abs_forecast_bps": float(np.mean(np.abs(mu))*10_000),
                "calibration_intercept": intercept, "calibration_slope": slope,
            }, "target": return_col, "portfolio": port})
        fold_data[period] = {"rows": len(rows), "fit": len(train), "calibration": len(cal), "test": len(test),
                             "test_start": str(test.timestamp_utc.min()), "test_end": str(test.timestamp_utc.max()),
                             "symbols": int(test.symbol.nunique())}
        pd.concat(prediction_frames, ignore_index=True).to_parquet(REPORT_DIR / "oos_predictions.parquet", index=False)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "development_only_no_model_promoted",
        "data_source": "Model 14 registered development symbols only; duplicate FB/META issuer samples removed; no Model 14 evaluation stock used.",
        "execution": {"signal": "after completed bar t", "entry": "open of t+1", "exit": "close of t+3",
                      "holding": "90 minutes from next-bar open", "round_trip_cost_bps": RT_COST_BPS,
                      "gross_exposure_limit_multiple": GROSS_LIMIT, "portfolio_starting_capital": STARTING_CAPITAL},
        "split_protocol": fold_data,
        "results": results,
        "important_limitations": [
            "Development time windows are 2016-2024 after adding 2019-2021; 2025-2026 observations are still absent.",
            "The 10 bps cost is a fixed assumption, not an empirical spread/slippage model.",
            "This study does not include delisted names outside the existing registered cohort or point-in-time universe sampling.",
            "The consumed Model 14 30-stock cohort is excluded from every Model 15 fit, calibration, and score.",
            "A favorable development result would still require a new frozen-model lockbox and paper trading before any market-competitiveness claim."
        ]
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "development_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    run()
