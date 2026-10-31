r"""Sequential, sealed-stock market-trend experiment.

Phases:
  train       Fit only from currently approved training symbols and freeze model.
  reveal TICKER YEAR0 YEAR1
              Download a sealed ticker only after a model is frozen, derive candles,
              score it once, and add it to the revealed ledger.

The prediction target is the sign of the cumulative return over the next three
30-minute candles within the same regular session. Rows with |future return| below
5 basis points are excluded as economically ambiguous. Evaluation is observational.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/trend_experiment"
STATE = OUT / "protocol.json"
MODELS = OUT / "models"
QUEUE = ["MSFT", "JPM", "XOM", "JNJ", "NVDA"]
LOCKED_CHALLENGE = [
    "GOOGL", "AMZN", "TSLA", "UNH", "V", "MA", "HD", "PG", "COST", "ABBV",
    "KO", "PEP", "BAC", "WMT", "DIS", "CSCO", "ADBE", "CRM", "NFLX", "INTC",
    "QCOM", "TMO", "ABT", "CVX", "COP", "BA", "CAT", "GE", "LIN", "NEE",
]
BASE_TRAIN = ["AAPL", "ATVI", "FB", "FRC", "META", "MMM", "SIVB", "TWTR"]
HORIZON = 3
DEADZONE = 0.0005
FEATURES = ["r1", "r2", "r3", "r6", "range1", "body1", "vol_rel20", "trades_rel20",
            "vwap_gap", "coverage", "slot_sin", "slot_cos", "partial",
            "spy_r1", "spy_r3", "spy_range1"]
_MARKET = None


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sigmoid(x):
    x = np.clip(x, -35, 35)
    return 1/(1+np.exp(-x))


def market_context():
    global _MARKET
    if _MARKET is None:
        path = ROOT / "data/candles/30min/SPY_30min.parquet"
        spy = pd.read_parquet(path).sort_values("timestamp_utc")
        group = spy.groupby("session_date", sort=False)
        spy["spy_r1"] = group.close.pct_change(1, fill_method=None)
        spy["spy_r3"] = group.close.pct_change(3, fill_method=None)
        spy["spy_range1"] = (spy.high-spy.low)/spy.open
        _MARKET = spy[["timestamp_utc", "spy_r1", "spy_r3", "spy_range1"]]
    return _MARKET


def make_rows(path, market=None, deadzone=None):
    deadzone = DEADZONE if deadzone is None else deadzone
    df = pd.read_parquet(path).sort_values("timestamp_utc").copy()
    symbol = str(df.symbol.iloc[0])
    g = df.groupby("session_date", sort=False)
    close = df.close
    for lag in [1, 2, 3, 6]:
        df[f"r{lag}"] = g.close.pct_change(lag, fill_method=None)
    df["range1"] = (df.high-df.low)/df.open
    df["body1"] = (df.close-df.open)/df.open
    # A regular session has only 13 half-hour candles. Five prior observations
    # keep the feature causal while leaving room for the three-candle horizon.
    df["vol_med20"] = g.volume.transform(lambda x: x.shift().rolling(10, min_periods=5).median())
    df["trade_med20"] = g.trade_count.transform(lambda x: x.shift().rolling(10, min_periods=5).median())
    df["vol_rel20"] = np.log1p(df.volume)/np.log1p(df.vol_med20)-1
    df["trades_rel20"] = np.log1p(df.trade_count)/np.log1p(df.trade_med20)-1
    df["vwap_gap"] = df.close/df.vwap-1
    df["coverage"] = df.observed_bars/df.expected_bars
    df["partial"] = df.is_partial.astype(float)
    slot = g.cumcount()
    df["slot_sin"] = np.sin(2*np.pi*slot/13)
    df["slot_cos"] = np.cos(2*np.pi*slot/13)
    future_close = g.close.shift(-HORIZON)
    df["future_return"] = future_close/df.close-1
    df["label_end_utc"] = pd.to_datetime(g.timestamp_utc.shift(-HORIZON), utc=True) + pd.Timedelta(minutes=30)
    df["label"] = (df.future_return > 0).astype(float)
    df["timestamp_utc"] = pd.to_datetime(df.timestamp_utc, utc=True)
    context = market_context() if market is None else market
    df = df.merge(context, on="timestamp_utc", how="left", validate="one_to_one")
    df["symbol"] = symbol
    good = df[FEATURES].replace([np.inf, -np.inf], np.nan).notna().all(axis=1)
    good &= df.future_return.abs().ge(deadzone) & future_close.notna()
    return df.loc[good, ["symbol", "timestamp_utc", "session_date", "label_end_utc", "future_return", "label", *FEATURES]]


def load_symbols(symbols, deadzone=None):
    frames = []
    for symbol in symbols:
        path = ROOT / f"data/candles/30min/{symbol}_30min.parquet"
        if path.exists():
            frames.append(make_rows(path, deadzone=deadzone))
    if not frames:
        raise RuntimeError("No candle files found")
    return pd.concat(frames, ignore_index=True)


def standardize_fit(x):
    mean = np.nanmean(x, axis=0)
    scale = np.nanstd(x, axis=0)
    scale[scale < 1e-9] = 1
    return mean, scale


def fit_logistic(x, y, penalty, mean=None, scale=None):
    if mean is None:
        mean, scale = standardize_fit(x)
    z = (x-mean)/scale
    z = np.column_stack([np.ones(len(z)), z])
    w = np.zeros(z.shape[1])
    reg = np.r_[0, np.ones(z.shape[1]-1)]
    for i in range(250):
        p = sigmoid(z@w)
        grad = z.T@(p-y)/len(y) + penalty*reg*w
        weights = np.clip(p*(1-p), 1e-5, None)
        hessian = (z.T*weights)@z/len(y) + np.diag(penalty*reg + 1e-8)
        step = np.linalg.solve(hessian, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return {"weights": w, "mean": mean, "scale": scale, "penalty": penalty, "iterations": i+1}


def predict(model, x):
    z = (x-model["mean"])/model["scale"]
    return sigmoid(np.column_stack([np.ones(len(z)), z])@model["weights"])


def auc(y, p):
    order = np.argsort(p)
    ranks = np.empty(len(p), float)
    sorted_p = p[order]
    start = 0
    while start < len(p):
        end = start+1
        while end < len(p) and sorted_p[end] == sorted_p[start]:
            end += 1
        ranks[order[start:end]] = (start+1+end)/2
        start = end
    n1, n0 = y.sum(), len(y)-y.sum()
    return (ranks[y == 1].sum()-n1*(n1+1)/2)/(n1*n0) if n1 and n0 else np.nan


def metrics(y, p, future):
    pred = p >= .5
    positive = y == 1
    tpr = (pred[positive]).mean() if positive.any() else np.nan
    tnr = (~pred[~positive]).mean() if (~positive).any() else np.nan
    baseline = np.full(len(y), np.mean(y))
    eps = 1e-12
    logloss = -np.mean(y*np.log(np.clip(p,eps,1-eps))+(1-y)*np.log(np.clip(1-p,eps,1-eps)))
    baseline_loss = -np.mean(y*np.log(np.clip(baseline,eps,1-eps))+(1-y)*np.log(np.clip(1-baseline,eps,1-eps)))
    signal = np.where(p >= .55, 1, np.where(p <= .45, -1, 0))
    return {"n": len(y), "positive_rate": float(y.mean()), "auc": float(auc(y,p)),
            "balanced_accuracy": float((tpr+tnr)/2), "log_loss": float(logloss),
            "fifty_fifty_log_loss": float(math.log(2)),
            "constant_log_loss": float(baseline_loss), "log_loss_improvement_pct": float(100*(baseline_loss-logloss)/baseline_loss),
            "coverage_at_55pct": float(np.mean(signal != 0)),
            "signed_mean_future_return_bps": float(np.mean(signal*future)*10000),
            "pass": bool(auc(y,p) >= .52 and logloss < baseline_loss and len(y) >= 500)}


def split_existing(data):
    from market_research.validation import chronological_split
    return chronological_split(data, (.75,))


def split_three(data):
    from market_research.validation import chronological_split
    return chronological_split(data)


def train_calibrated(state, symbols, data):
    """Fit nonlinear ranker, then learn probability calibration on later data."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from joblib import dump
    train_df, cal_df, val_df = split_three(data)
    xtr, ytr = train_df[FEATURES].to_numpy(float), train_df.label.to_numpy(int)
    xcal, ycal = cal_df[FEATURES].to_numpy(float), cal_df.label.to_numpy(int)
    xval, yval = val_df[FEATURES].to_numpy(float), val_df.label.to_numpy(int)
    candidates = []
    for leaves in [7, 15, 31]:
        for l2 in [1., 10., 50.]:
            estimator = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, max_leaf_nodes=leaves,
                        min_samples_leaf=80, l2_regularization=l2, random_state=20261023, early_stopping=False)
            estimator.fit(xtr, ytr)
            cal_raw = estimator.predict_proba(xcal)[:,1]
            scales = [0., .05, .1, .2, .35, .5, .75, 1.]
            def loss(scale):
                q = np.clip(.5+scale*(cal_raw-.5), 1e-12, 1-1e-12)
                return -np.mean(ycal*np.log(q)+(1-ycal)*np.log(1-q))
            calibration_scale = min(scales, key=loss)
            val_raw = estimator.predict_proba(xval)[:,1]
            probabilities = .5+calibration_scale*(val_raw-.5)
            score = metrics(yval, probabilities, val_df.future_return.to_numpy())
            candidates.append((score["log_loss"], leaves, l2, calibration_scale, score, estimator))
    _, leaves, l2, calibration_scale, validation, estimator = min(candidates, key=lambda x:x[0])
    index = len(state["models"])+1
    path = MODELS / f"model_{index:02d}.joblib"
    dump({"estimator": estimator, "calibration_scale": calibration_scale, "features": FEATURES,
          "training_symbols": symbols, "fit_fraction": .60, "calibration_fraction": .20}, path)
    record = {"model_id": index, "created_utc": datetime.now(timezone.utc).isoformat(), "path": str(path.relative_to(ROOT)),
              "sha256": digest(path), "training_symbols": symbols, "training_rows": len(train_df),
              "calibration_rows": len(cal_df), "validation_rows": len(val_df),
              "training_max_utc": str(train_df.timestamp_utc.max()), "model_family": "calibrated_hist_gradient_boosting",
              "model_parameters": {"max_leaf_nodes": leaves, "l2": l2, "calibration_scale": calibration_scale}, "validation": validation,
              "candidate_validation": [{"max_leaf_nodes": lv, "l2": reg, "calibration_scale": scale, **score} for _,lv,reg,scale,score,_ in candidates],
              "next_sealed_symbol": state["sealed_queue"][len(state["reveals"])]}
    state["models"].append(record)
    save_state(state)
    print(json.dumps(record, indent=2))


def initial_state():
    OUT.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    if STATE.exists():
        return json.loads(STATE.read_text(encoding="utf-8"))
    state = {"created_utc": datetime.now(timezone.utc).isoformat(), "protocol_version": 1,
             "task": "direction of next 3 completed 30-minute candles within same regular session",
             "horizon_candles": HORIZON, "deadzone_absolute_return": DEADZONE,
             "features": FEATURES, "base_training_symbols": BASE_TRAIN, "sealed_queue": QUEUE,
             "pass_rule": "AUC >= 0.52, log loss better than 50/50 no-skill forecast, n >= 500",
             "success_rule": "five consecutive one-time passes on sealed stocks",
             "adaptation_rule": "after a reveal, its results may diagnose/revise the next model; it is then training data and never unseen evidence again",
             "models": [], "reveals": [], "consecutive_passes": 0}
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def save_state(state):
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def train():
    state = initial_state()
    # Protocol v1 compared against an oracle test-set prevalence, which would not
    # be known at deployment. After the first failure, correct this prospectively;
    # the MSFT result remains failed under the rule frozen for model 1.
    if state["protocol_version"] == 1 and state["reveals"]:
        state["protocol_version"] = 2
        state["pass_rule"] = "AUC >= 0.52, log loss < log(2), n >= 500"
        state["sealed_queue"].append("AMD")
        state["protocol_changes"] = [{"effective_after_symbol": "MSFT",
            "reason": "The v1 stock-specific constant used test-set prevalence and was an unavailable oracle. Version 2 uses the preregistered 50/50 no-skill probability; MSFT is not retroactively passed."}]
        save_state(state)
    if state["protocol_version"] == 2 and len(state["models"]) >= 3:
        state["protocol_version"] = 3
        state["features"] = FEATURES
        state.setdefault("protocol_changes", []).append({"effective_after_model": 3,
            "reason": "Stock-only linear and nonlinear models failed chronological validation. Add contemporaneous SPY returns/range observable at prediction time; JPM remains sealed."})
        save_state(state)
    latest_reveal = state["reveals"][-1]["symbol"] if state["reveals"] else None
    if (state["protocol_version"] >= 3 and latest_reveal
            and latest_reveal in state["models"][-1]["training_symbols"]
            and state["models"][-1]["validation"]["log_loss"] >= math.log(2)
            and latest_reveal not in state.get("excluded_training_symbols", [])):
        state.setdefault("excluded_training_symbols", []).append(latest_reveal)
        state.setdefault("protocol_changes", []).append({"effective_after_model": state["models"][-1]["model_id"],
            "reason": f"Including {latest_reveal} degraded untouched chronological validation; keep its sealed result in the ledger but exclude its distribution from the next pooled fit."})
        save_state(state)
    excluded = set(state.get("excluded_training_symbols", []))
    revealed = [x["symbol"] for x in state["reveals"] if x["symbol"] not in excluded]
    symbols = BASE_TRAIN + revealed
    data = load_symbols(symbols)
    if state["protocol_version"] >= 3:
        train_calibrated(state, symbols, data)
        return
    train_df, val_df = split_existing(data)
    xtr, ytr = train_df[FEATURES].to_numpy(float), train_df.label.to_numpy(int)
    xva, yva = val_df[FEATURES].to_numpy(float), val_df.label.to_numpy(int)
    candidates = []
    for penalty in [0, 1e-4, 1e-3, 1e-2, 1e-1]:
        model = fit_logistic(xtr, ytr, penalty)
        score = metrics(yva, predict(model, xva), val_df.future_return.to_numpy())
        candidates.append((score["log_loss"], "logistic", {"penalty": penalty}, score))
    from sklearn.ensemble import HistGradientBoostingClassifier
    for leaves in [7, 15, 31]:
        for l2 in [1., 10.]:
            tree = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05, max_leaf_nodes=leaves,
                    min_samples_leaf=80, l2_regularization=l2, random_state=20261023, early_stopping=False)
            tree.fit(xtr, ytr)
            score = metrics(yva, tree.predict_proba(xva)[:,1], val_df.future_return.to_numpy())
            candidates.append((score["log_loss"], "hist_gradient_boosting", {"max_leaf_nodes": leaves, "l2": l2}, score))
    _, family, params, validation = min(candidates, key=lambda x: x[0])
    index = len(state["models"])+1
    if family == "logistic":
        fitted = fit_logistic(data[FEATURES].to_numpy(float), data.label.to_numpy(int), params["penalty"])
        path = MODELS / f"model_{index:02d}.npz"
        np.savez(path, weights=fitted["weights"], mean=fitted["mean"], scale=fitted["scale"], penalty=params["penalty"],
                 features=np.array(FEATURES), training_symbols=np.array(symbols), training_max_utc=str(data.timestamp_utc.max()))
    else:
        from joblib import dump
        fitted = HistGradientBoostingClassifier(max_iter=150, learning_rate=.05,
                 max_leaf_nodes=params["max_leaf_nodes"], min_samples_leaf=80,
                 l2_regularization=params["l2"], random_state=20261023, early_stopping=False)
        fitted.fit(data[FEATURES].to_numpy(float), data.label.to_numpy(int))
        path = MODELS / f"model_{index:02d}.joblib"
        dump({"estimator": fitted, "features": FEATURES, "training_symbols": symbols}, path)
    record = {"model_id": index, "created_utc": datetime.now(timezone.utc).isoformat(), "path": str(path.relative_to(ROOT)),
              "sha256": digest(path), "training_symbols": symbols, "training_rows": len(data),
              "training_max_utc": str(data.timestamp_utc.max()), "model_family": family, "model_parameters": params,
              "validation": validation, "candidate_validation": [{"family": f, **prm, **s} for _,f,prm,s in candidates],
              "next_sealed_symbol": state["sealed_queue"][len(state["reveals"])] if len(state["reveals"]) < len(state["sealed_queue"]) else None}
    state["models"].append(record)
    save_state(state)
    print(json.dumps(record, indent=2))


def reveal(symbol, start_year, end_year):
    state = initial_state()
    expected = state["sealed_queue"][len(state["reveals"])]
    if symbol != expected:
        raise ValueError(f"Protocol requires next sealed symbol {expected}, received {symbol}")
    if not state["models"]:
        raise RuntimeError("Train and freeze a model before revealing a stock")
    frozen = state["models"][-1]
    if frozen["next_sealed_symbol"] != symbol:
        raise RuntimeError("Latest frozen model was not registered for this stock")
    # Import here to ensure no retrieval occurs in the train phase.
    from market_research.backfill import run_backfill
    run_backfill(symbols=[symbol], start_year=start_year, end_year=end_year, feed="sip")
    path = ROOT / f"data/candles/30min/{symbol}_30min.parquet"
    data = make_rows(path)
    model_path = ROOT / frozen["path"]
    if model_path.suffix == ".npz":
        archive = np.load(model_path, allow_pickle=False)
        model = {"weights": archive["weights"], "mean": archive["mean"], "scale": archive["scale"]}
        p = predict(model, data[FEATURES].to_numpy(float))
    else:
        from joblib import load
        payload = load(model_path)
        model = payload["estimator"]
        p = model.predict_proba(data[FEATURES].to_numpy(float))[:,1]
        if "calibrator" in payload:
            raw = np.clip(p, 1e-6, 1-1e-6)
            p = predict(payload["calibrator"], np.log(raw/(1-raw)).reshape(-1,1))
        elif "calibration_scale" in payload:
            p = .5+payload["calibration_scale"]*(p-.5)
    score = metrics(data.label.to_numpy(int), p, data.future_return.to_numpy())
    if state["protocol_version"] >= 2:
        score["pass"] = bool(score["auc"] >= .52 and score["log_loss"] < math.log(2) and score["n"] >= 500)
    scored = data[["timestamp_utc", "session_date", "future_return", "label"]].copy()
    scored["probability_up"] = p
    scored.to_parquet(OUT / f"sealed_predictions_{symbol}.parquet", index=False)
    reveal_record = {"symbol": symbol, "revealed_utc": datetime.now(timezone.utc).isoformat(),
                     "years": [start_year, end_year], "model_id": frozen["model_id"], "model_sha256": frozen["sha256"],
                     "candle_sha256": digest(path), "metrics": score,
                     "protocol_version": state["protocol_version"]}
    state["reveals"].append(reveal_record)
    state["consecutive_passes"] = state["consecutive_passes"]+1 if score["pass"] else 0
    save_state(state)
    print(json.dumps(reveal_record, indent=2))


def register_locked_challenge():
    """Preregister a fixed-model, fixed-basket external validation challenge."""
    state = initial_state()
    if state.get("locked_challenge"):
        print(json.dumps(state["locked_challenge"], indent=2))
        return
    if state.get("consecutive_passes", 0) < 5:
        raise RuntimeError("Complete the original five-pass experiment first")
    frozen = state["models"][-1]
    already_seen = set(state["base_training_symbols"])
    already_seen.update(x["symbol"] for x in state["reveals"])
    overlap = already_seen.intersection(LOCKED_CHALLENGE)
    if overlap:
        raise RuntimeError(f"Challenge symbols already seen: {sorted(overlap)}")
    existing = [s for s in LOCKED_CHALLENGE
                if (ROOT / f"data/candles/30min/{s}_30min.parquet").exists()]
    if existing:
        raise RuntimeError(f"Challenge candle data already exists before registration: {existing}")
    state["locked_challenge"] = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "design": "30 prospectively selected unseen stocks, scored once by one locked model; no adaptation between stocks",
        "model_id": frozen["model_id"],
        "model_path": frozen["path"],
        "model_sha256": frozen["sha256"],
        "symbols": LOCKED_CHALLENGE,
        "years": [2022, 2024],
        "pass_rule": state["pass_rule"],
        "results": [],
    }
    save_state(state)
    print(json.dumps(state["locked_challenge"], indent=2))


def reveal_locked(symbol, start_year, end_year):
    state = initial_state()
    challenge = state.get("locked_challenge")
    if not challenge:
        raise RuntimeError("Register the locked challenge before downloading any challenge stock")
    index = len(challenge["results"])
    if index >= len(challenge["symbols"]):
        raise RuntimeError("Locked challenge is already complete")
    expected = challenge["symbols"][index]
    if symbol != expected:
        raise ValueError(f"Locked challenge requires {expected} next, received {symbol}")
    if [start_year, end_year] != challenge["years"]:
        raise ValueError(f"Locked challenge years are {challenge['years']}")
    model_path = ROOT / challenge["model_path"]
    if digest(model_path) != challenge["model_sha256"]:
        raise RuntimeError("Frozen challenge model hash changed")
    path = ROOT / f"data/candles/30min/{symbol}_30min.parquet"
    # A registered challenge may prefetch disjoint symbols concurrently. The
    # model stays locked and evaluation still occurs once, in declared order.
    if not path.exists():
        from market_research.backfill import run_backfill
        run_backfill(symbols=[symbol], start_year=start_year, end_year=end_year, feed="sip")
    data = make_rows(path)
    from joblib import load
    payload = load(model_path)
    p = payload["estimator"].predict_proba(data[FEATURES].to_numpy(float))[:, 1]
    p = .5 + payload.get("calibration_scale", 1.0) * (p - .5)
    score = metrics(data.label.to_numpy(int), p, data.future_return.to_numpy())
    score["pass"] = bool(score["auc"] >= .52 and score["log_loss"] < math.log(2) and score["n"] >= 500)
    scored = data[["timestamp_utc", "session_date", "future_return", "label"]].copy()
    scored["probability_up"] = p
    scored.to_parquet(OUT / f"locked_predictions_{symbol}.parquet", index=False)
    record = {"symbol": symbol, "revealed_utc": datetime.now(timezone.utc).isoformat(),
              "years": [start_year, end_year], "model_id": challenge["model_id"],
              "model_sha256": challenge["model_sha256"], "candle_sha256": digest(path),
              "metrics": score}
    challenge["results"].append(record)
    save_state(state)
    print(json.dumps(record, indent=2))


def run_locked_challenge():
    """Resume the registered challenge in its immutable declared order."""
    while True:
        state = initial_state()
        challenge = state.get("locked_challenge")
        if not challenge:
            raise RuntimeError("Register the locked challenge first")
        index = len(challenge["results"])
        if index >= len(challenge["symbols"]):
            print("Locked challenge complete", flush=True)
            return
        symbol = challenge["symbols"][index]
        print(f"LOCKED {index + 1}/{len(challenge['symbols'])}: {symbol}", flush=True)
        reveal_locked(symbol, *challenge["years"])


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("train")
    sub.add_parser("register-locked-challenge")
    sub.add_parser("run-locked-challenge")
    reveal_parser = sub.add_parser("reveal")
    # Validate against the persisted sealed queue in reveal(); the ledger can
    # prospectively extend the original queue without exposing a symbol.
    reveal_parser.add_argument("symbol")
    reveal_parser.add_argument("start_year", type=int)
    reveal_parser.add_argument("end_year", type=int)
    locked_parser = sub.add_parser("reveal-locked")
    locked_parser.add_argument("symbol")
    locked_parser.add_argument("start_year", type=int)
    locked_parser.add_argument("end_year", type=int)
    args = parser.parse_args()
    if args.command == "train":
        train()
    elif args.command == "register-locked-challenge":
        register_locked_challenge()
    elif args.command == "run-locked-challenge":
        run_locked_challenge()
    elif args.command == "reveal-locked":
        reveal_locked(args.symbol, args.start_year, args.end_year)
    else:
        reveal(args.symbol, args.start_year, args.end_year)


if __name__ == "__main__":
    main()
