"""Model 14: threshold-free probability-sized long/short portfolio experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from market_research.acquire import AlpacaAcquisitionClient, acquire_symbol_year
from market_research.resample import resample_symbol_bars
from market_research.trend_experiment import FEATURES, make_rows

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/model14"
STATE_PATH = REPORT / "protocol.json"
REGISTRY_PATH = ROOT / "reports/trend_experiment/stock_registry.json"
DATA_ROOT = ROOT / "data/challenges/model14"
MODEL_PATH = REPORT / "model_14.joblib"
TRAIN_PERIODS = {"pre": (2016, 2018), "post": (2022, 2024)}
STARTING_CAPITAL = 100_000.0
UNIT = 100.0
MAX_DRAWDOWN = 30_000.0
ROUND_TRIP_COST_BPS = 10.0
EVAL_SYMBOLS = [
    "ORCL", "IBM", "TXN", "AVGO", "ACN", "MCD", "SBUX", "LOW", "TGT", "CVS",
    "PFE", "MRK", "LLY", "MDT", "GS", "C", "MS", "BLK", "AXP", "USB",
    "SLB", "EOG", "OXY", "DE", "HON", "UPS", "RTX", "DUK", "SO", "AMT",
]
MODEL_FEATURES = FEATURES + ["regime_post"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_state() -> dict:
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    REPORT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def register() -> None:
    if STATE_PATH.exists():
        print(json.dumps(read_state(), indent=2))
        return
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    training = sorted(set(registry["ever_trained_symbols"]) | set(registry["all_evaluated_symbols"]))
    overlap = sorted(set(training).intersection(EVAL_SYMBOLS))
    if overlap:
        raise RuntimeError(f"Evaluation overlap: {overlap}")
    existing = [s for s in EVAL_SYMBOLS if any((DATA_ROOT / role / period / "candles/30min" / f"{s}_30min.parquet").exists()
                for role in ("train", "eval") for period in TRAIN_PERIODS)]
    if existing:
        raise RuntimeError(f"Evaluation data existed before registration: {existing}")
    state = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": 14,
        "objective": "maximize validation net profit subject to portfolio maximum drawdown below 30% of starting capital",
        "position_rule": "position_dollars = 100 * (2 * probability_up - 1)",
        "holding_period": "three completed 30-minute candles, approximately 90 trading minutes",
        "starting_capital": STARTING_CAPITAL,
        "unit_dollars": UNIT,
        "maximum_drawdown_dollars": MAX_DRAWDOWN,
        "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
        "periods": TRAIN_PERIODS,
        "training_symbols": training,
        "evaluation_symbols": EVAL_SYMBOLS,
        "training_overlap": overlap,
        "registry_sha256": digest(REGISTRY_PATH),
        "model": None,
        "evaluation": None,
    }
    save_state(state)
    print(json.dumps(state, indent=2))


def period_dir(role: str, period: str) -> Path:
    return DATA_ROOT / role / period


def prepare(role: str, period: str, symbol: str) -> None:
    state = read_state()
    allowed = set(state["training_symbols"] if role == "train" else state["evaluation_symbols"])
    allowed.add("SPY")
    if symbol not in allowed:
        raise ValueError(f"{symbol} is not allowed for {role}")
    start, end = TRAIN_PERIODS[period]
    frames = []
    with AlpacaAcquisitionClient(feed="sip") as client:
        for year in range(start, end + 1):
            result = acquire_symbol_year(client, symbol, year, output_base=ROOT / "data/raw")
            print(f"{role}/{period} {symbol} {year}: {result['status']} ({result['bars_count']:,})", flush=True)
            if result["bars_count"] and result["file_path"]:
                frames.append(pd.read_parquet(Path(result["file_path"])))
    if not frames:
        print(f"No data for {symbol} in {start}-{end}", flush=True)
        return
    base = period_dir(role, period)
    result = resample_symbol_bars(pd.concat(frames, ignore_index=True), symbol,
        output_dir_20=base / "candles/20min", output_dir_30=base / "candles/30min", output_dir_norm=None)
    print(json.dumps(result), flush=True)


def market_context(role: str, period: str) -> pd.DataFrame:
    spy = pd.read_parquet(period_dir(role, period) / "candles/30min/SPY_30min.parquet").sort_values("timestamp_utc")
    g = spy.groupby("session_date", sort=False)
    spy["spy_r1"] = g.close.pct_change(1, fill_method=None)
    spy["spy_r3"] = g.close.pct_change(3, fill_method=None)
    spy["spy_range1"] = (spy.high - spy.low) / spy.open
    spy["timestamp_utc"] = pd.to_datetime(spy.timestamp_utc, utc=True)
    return spy[["timestamp_utc", "spy_r1", "spy_r3", "spy_range1"]]


def load_role(role: str, symbols: list[str]) -> pd.DataFrame:
    frames = []
    periods = TRAIN_PERIODS if role == "train" else {"post": TRAIN_PERIODS["post"]}
    for period in periods:
        market = market_context(role, period)
        for symbol in symbols:
            path = period_dir(role, period) / f"candles/30min/{symbol}_30min.parquet"
            if not path.exists():
                continue
            rows = make_rows(path, market=market, deadzone=0.0)
            rows["period"] = period
            rows["regime_post"] = float(period == "post")
            frames.append(rows)
    if not frames:
        raise RuntimeError(f"No prepared {role} rows")
    return pd.concat(frames, ignore_index=True)


def split_three(data: pd.DataFrame):
    from market_research.validation import chronological_split
    return chronological_split(data)


def portfolio_metrics(frame: pd.DataFrame, probabilities: np.ndarray) -> dict:
    position = UNIT * (2 * probabilities - 1)
    gross = position * frame.future_return.to_numpy(float)
    costs = np.abs(position) * ROUND_TRIP_COST_BPS / 10_000
    entry = pd.to_datetime(frame.timestamp_utc, utc=True)
    trades = pd.DataFrame({"exit": entry + pd.Timedelta(minutes=90), "net": gross - costs, "position": position})
    by_time = trades.groupby("exit", sort=True).agg(net=("net", "sum"))
    # Each signal remains open for the next three half-hour slots. Sum the
    # overlapping cohorts to measure actual concurrent gross exposure.
    exposure = pd.concat([pd.DataFrame({"time": entry + pd.Timedelta(minutes=offset), "gross": np.abs(position)})
                          for offset in (0, 30, 60)], ignore_index=True).groupby("time").gross.sum()
    equity = STARTING_CAPITAL + by_time.net.cumsum()
    drawdown = equity.cummax() - equity
    y = frame.label.to_numpy(int)
    return {
        "rows": len(frame), "net_profit": float(by_time.net.sum()), "ending_equity": float(equity.iloc[-1]),
        "return_pct": float(100 * by_time.net.sum() / STARTING_CAPITAL),
        "maximum_drawdown_dollars": float(drawdown.max()),
        "maximum_drawdown_pct": float(100 * drawdown.max() / STARTING_CAPITAL),
        "maximum_concurrent_gross_exposure": float(exposure.max()),
        "auc": float(roc_auc_score(y, probabilities)), "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "mean_absolute_position": float(np.mean(np.abs(position))),
        "profitable": bool(by_time.net.sum() > 0), "drawdown_pass": bool(drawdown.max() < MAX_DRAWDOWN),
    }


def calibrated_probabilities(estimator, calibrator, x):
    raw = np.clip(estimator.predict_proba(x)[:, 1], 1e-6, 1 - 1e-6)
    logits = np.log(raw / (1 - raw)).reshape(-1, 1)
    return calibrator.predict_proba(logits)[:, 1]


def train() -> None:
    state = read_state()
    data = load_role("train", state["training_symbols"])
    fit, cal, val = split_three(data)
    candidates = []
    for leaves in (7, 15, 31):
        for l2 in (1.0, 10.0, 50.0):
            est = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, max_leaf_nodes=leaves,
                min_samples_leaf=100, l2_regularization=l2, random_state=20261028, early_stopping=False)
            weights = 1 + np.clip(np.abs(fit.future_return.to_numpy()) * 10_000, 0, 25)
            est.fit(fit[MODEL_FEATURES], fit.label.astype(int), sample_weight=weights)
            raw_cal = np.clip(est.predict_proba(cal[MODEL_FEATURES])[:, 1], 1e-6, 1 - 1e-6)
            calibrator = LogisticRegression(C=1.0, random_state=20261028).fit(
                np.log(raw_cal / (1 - raw_cal)).reshape(-1, 1), cal.label.astype(int))
            p = calibrated_probabilities(est, calibrator, val[MODEL_FEATURES])
            score = portfolio_metrics(val, p)
            candidates.append({"leaves": leaves, "l2": l2, "metrics": score, "estimator": est, "calibrator": calibrator})
    eligible = [c for c in candidates if c["metrics"]["drawdown_pass"]]
    if not eligible:
        raise RuntimeError("No candidate satisfies the drawdown constraint")
    best = max(eligible, key=lambda c: c["metrics"]["net_profit"])
    # Preserve a chronological calibration slice; refit the ranker on the first 80%.
    fit2 = pd.concat([fit, cal], ignore_index=True)
    est = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, max_leaf_nodes=best["leaves"],
        min_samples_leaf=100, l2_regularization=best["l2"], random_state=20261028, early_stopping=False)
    weights2 = 1 + np.clip(np.abs(fit2.future_return.to_numpy()) * 10_000, 0, 25)
    est.fit(fit2[MODEL_FEATURES], fit2.label.astype(int), sample_weight=weights2)
    raw = np.clip(est.predict_proba(val[MODEL_FEATURES])[:, 1], 1e-6, 1 - 1e-6)
    calibrator = LogisticRegression(C=1.0, random_state=20261028).fit(
        np.log(raw / (1 - raw)).reshape(-1, 1), val.label.astype(int))
    dump({"estimator": est, "calibrator": calibrator, "features": MODEL_FEATURES,
          "training_symbols": state["training_symbols"], "periods": TRAIN_PERIODS}, MODEL_PATH)
    if state.get("model"):
        state.setdefault("development_attempts", []).append(state["model"])
    state["model"] = {
        "trained_utc": datetime.now(timezone.utc).isoformat(), "path": str(MODEL_PATH.relative_to(ROOT)),
        "sha256": digest(MODEL_PATH), "parameters": {"max_leaf_nodes": best["leaves"], "l2": best["l2"]},
        "development_rows": len(data), "validation": best["metrics"],
        "features": MODEL_FEATURES, "sample_weight": "1 + min(abs(future_return)*10000, 25)",
        "candidates": [{"leaves": c["leaves"], "l2": c["l2"], **c["metrics"]} for c in candidates],
    }
    save_state(state)
    print(json.dumps(state["model"], indent=2))


def evaluate() -> None:
    state = read_state()
    if not state["model"] or digest(MODEL_PATH) != state["model"]["sha256"]:
        raise RuntimeError("Frozen model 14 is absent or changed")
    payload = load(MODEL_PATH)
    overlap = set(payload["training_symbols"]).intersection(state["evaluation_symbols"])
    if overlap:
        raise RuntimeError(f"Training/evaluation overlap: {sorted(overlap)}")
    data = load_role("eval", state["evaluation_symbols"])
    p = calibrated_probabilities(payload["estimator"], payload["calibrator"], data[payload["features"]])
    aggregate = portfolio_metrics(data, p)
    per_stock = []
    for symbol, g in data.groupby("symbol"):
        idx = g.index.to_numpy()
        # data has a fresh RangeIndex from load_role.
        per_stock.append({"symbol": symbol, **portfolio_metrics(g, p[idx])})
    predictions = data[["symbol", "period", "timestamp_utc", "future_return", "label"]].copy()
    predictions["probability_up"] = p
    predictions["position_dollars"] = UNIT * (2 * p - 1)
    predictions.to_parquet(REPORT / "sealed_predictions.parquet", index=False)
    state["evaluation"] = {"evaluated_utc": datetime.now(timezone.utc).isoformat(), "model_sha256": digest(MODEL_PATH),
                           "aggregate": aggregate, "per_stock": per_stock,
                           "pass": bool(aggregate["net_profit"] > 0 and aggregate["maximum_drawdown_dollars"] < MAX_DRAWDOWN)}
    save_state(state)
    print(json.dumps(state["evaluation"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    prep = sub.add_parser("prepare-batch")
    prep.add_argument("role", choices=["train", "eval"]); prep.add_argument("period", choices=list(TRAIN_PERIODS)); prep.add_argument("symbols")
    both = sub.add_parser("prepare-both-periods")
    both.add_argument("role", choices=["train", "eval"]); both.add_argument("symbols")
    sub.add_parser("train"); sub.add_parser("evaluate")
    args = parser.parse_args()
    if args.command == "register": register()
    elif args.command == "prepare-batch":
        for symbol in args.symbols.split(","): prepare(args.role, args.period, symbol.strip().upper())
    elif args.command == "prepare-both-periods":
        for period in TRAIN_PERIODS:
            for symbol in args.symbols.split(","): prepare(args.role, period, symbol.strip().upper())
    elif args.command == "train": train()
    else: evaluate()


if __name__ == "__main__":
    main()
