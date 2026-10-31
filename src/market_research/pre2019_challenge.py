"""Locked backward regime-transfer test for the 2016-2018 market period."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from market_research.acquire import AlpacaAcquisitionClient, acquire_symbol_year
from market_research.resample import resample_symbol_bars
from market_research.trend_experiment import FEATURES, make_rows, metrics

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/trend_experiment"
STATE = OUT / "protocol.json"
REGISTRY = OUT / "stock_registry.json"
REGISTRY_MD = OUT / "STOCK_REGISTRY.md"
PERIOD_ROOT = ROOT / "data/challenges/pre2019_2016_2018"
START_YEAR, END_YEAR = 2016, 2018


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_state() -> dict:
    return json.loads(STATE.read_text(encoding="utf-8"))


def save_state(state: dict) -> None:
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def build_registry(state: dict) -> dict:
    per_model = {str(m["model_id"]): m["training_symbols"] for m in state["models"]}
    ever_trained = sorted({s for values in per_model.values() for s in values})
    sequential = [r["symbol"] for r in state["reveals"]]
    locked = state["locked_challenge"]["symbols"]
    final_model = state["models"][-1]
    pre2019 = state.get("pre2019_challenge", {}).get("symbols", [])
    all_evaluated = sorted(set(sequential + locked + pre2019))
    registry = {
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Canonical role registry. A symbol in a selected model's training set must never be described as unseen evaluation for that model.",
        "models": per_model,
        "ever_trained_symbols": ever_trained,
        "sequential_evaluation_symbols": sequential,
        "locked_30_evaluation_symbols": locked,
        "pre2019_temporal_evaluation_symbols": pre2019,
        "all_evaluated_symbols": all_evaluated,
        "final_model": {
            "model_id": final_model["model_id"],
            "sha256": final_model["sha256"],
            "training_symbols": final_model["training_symbols"],
            "forbidden_as_unseen_evaluation": final_model["training_symbols"],
        },
    }
    REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    lines = [
        "# Stock Role Registry", "",
        "This is the canonical guard against training/evaluation overlap. A ticker in the selected model's training list must not be described as unseen for that model.", "",
        f"## Final model {final_model['model_id']} training symbols", "",
        ", ".join(final_model["training_symbols"]), "",
        "## Ever used in training by any model", "", ", ".join(ever_trained), "",
        "## Sequential evaluation symbols", "", ", ".join(sequential), "",
        "## Locked 30-stock evaluation symbols (2022-2024)", "", ", ".join(locked), "",
        "## Pre-COVID temporal evaluation symbols (2016-2018)", "", ", ".join(pre2019), "",
        "## All evaluated symbols", "", ", ".join(all_evaluated), "",
    ]
    REGISTRY_MD.write_text("\n".join(lines), encoding="utf-8")
    return registry


def register() -> None:
    state = load_state()
    if state.get("pre2019_challenge"):
        print(json.dumps(state["pre2019_challenge"], indent=2))
        return
    registry = build_registry(state)
    frozen = state["models"][-1]
    symbols = list(state["locked_challenge"]["symbols"])
    overlap = sorted(set(symbols).intersection(frozen["training_symbols"]))
    if overlap:
        raise RuntimeError(f"Model-training overlap blocks evaluation: {overlap}")
    state["pre2019_challenge"] = {
        "registered_utc": datetime.now(timezone.utc).isoformat(),
        "design": "Backward regime-transfer test on 2016-2018 using the same 30-stock basket and one unchanged model",
        "interpretation": "Tests robustness to pre-COVID conditions; not a forward-in-time deployment simulation",
        "model_id": frozen["model_id"],
        "model_path": frozen["path"],
        "model_sha256": frozen["sha256"],
        "symbols": symbols,
        "market_context_symbol": "SPY",
        "years": [START_YEAR, END_YEAR],
        "training_overlap": overlap,
        "registry_path": str(REGISTRY.relative_to(ROOT)),
        "registry_sha256": digest(REGISTRY),
        "pass_rule": state["locked_challenge"]["pass_rule"],
        "results": [],
    }
    save_state(state)
    print(json.dumps(state["pre2019_challenge"], indent=2))


def prepare(symbol: str) -> None:
    """Acquire raw yearly files and derive period-isolated candles."""
    state = load_state()
    challenge = state.get("pre2019_challenge")
    allowed = set(challenge["symbols"]) | {challenge["market_context_symbol"]}
    if symbol not in allowed:
        raise ValueError(f"{symbol} is not preregistered for this challenge")
    frames = []
    with AlpacaAcquisitionClient(feed="sip") as client:
        for year in range(START_YEAR, END_YEAR + 1):
            result = acquire_symbol_year(client, symbol, year, output_base=ROOT / "data/raw")
            print(f"{symbol} {year}: {result['status']} ({result['bars_count']:,})", flush=True)
            if result["bars_count"] and result["file_path"]:
                frames.append(pd.read_parquet(ROOT / result["file_path"]))
    if not frames:
        raise RuntimeError(f"No raw data acquired for {symbol}")
    raw = pd.concat(frames, ignore_index=True)
    result = resample_symbol_bars(
        raw, symbol,
        output_dir_20=PERIOD_ROOT / "candles/20min",
        output_dir_30=PERIOD_ROOT / "candles/30min",
        output_dir_norm=None,
    )
    print(json.dumps(result, indent=2), flush=True)


def market_rows() -> pd.DataFrame:
    path = PERIOD_ROOT / "candles/30min/SPY_30min.parquet"
    spy = pd.read_parquet(path).sort_values("timestamp_utc")
    group = spy.groupby("session_date", sort=False)
    spy["spy_r1"] = group.close.pct_change(1, fill_method=None)
    spy["spy_r3"] = group.close.pct_change(3, fill_method=None)
    spy["spy_range1"] = (spy.high - spy.low) / spy.open
    spy["timestamp_utc"] = pd.to_datetime(spy.timestamp_utc, utc=True)
    return spy[["timestamp_utc", "spy_r1", "spy_r3", "spy_range1"]]


def run() -> None:
    from joblib import load
    state = load_state()
    challenge = state["pre2019_challenge"]
    if set(challenge["symbols"]).intersection(state["models"][-1]["training_symbols"]):
        raise RuntimeError("Training/evaluation overlap detected")
    model_path = ROOT / challenge["model_path"]
    if digest(model_path) != challenge["model_sha256"]:
        raise RuntimeError("Frozen model hash changed")
    payload = load(model_path)
    market = market_rows()
    while len(challenge["results"]) < len(challenge["symbols"]):
        symbol = challenge["symbols"][len(challenge["results"])]
        path = PERIOD_ROOT / f"candles/30min/{symbol}_30min.parquet"
        if not path.exists():
            raise FileNotFoundError(f"Prepare {symbol} before scoring")
        data = make_rows(path, market=market)
        p = payload["estimator"].predict_proba(data[FEATURES].to_numpy(float))[:, 1]
        p = .5 + payload.get("calibration_scale", 1.0) * (p - .5)
        score = metrics(data.label.to_numpy(int), p, data.future_return.to_numpy())
        score["pass"] = bool(score["auc"] >= .52 and score["log_loss"] < math.log(2) and score["n"] >= 500)
        scored = data[["timestamp_utc", "session_date", "future_return", "label"]].copy()
        scored["probability_up"] = p
        scored.to_parquet(OUT / f"pre2019_predictions_{symbol}.parquet", index=False)
        record = {"symbol": symbol, "revealed_utc": datetime.now(timezone.utc).isoformat(),
                  "years": [START_YEAR, END_YEAR], "model_id": challenge["model_id"],
                  "model_sha256": challenge["model_sha256"], "candle_sha256": digest(path),
                  "metrics": score}
        challenge["results"].append(record)
        save_state(state)
        print(json.dumps(record, indent=2), flush=True)
    build_registry(state)
    challenge["registry_sha256_current"] = digest(REGISTRY)
    challenge["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save_state(state)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("register")
    prep = sub.add_parser("prepare")
    prep.add_argument("symbol")
    batch = sub.add_parser("prepare-batch")
    batch.add_argument("symbols", help="Comma-separated preregistered symbols")
    sub.add_parser("run")
    args = parser.parse_args()
    if args.command == "register":
        register()
    elif args.command == "prepare":
        prepare(args.symbol.upper())
    elif args.command == "prepare-batch":
        for symbol in args.symbols.split(","):
            prepare(symbol.strip().upper())
    else:
        run()


if __name__ == "__main__":
    main()
