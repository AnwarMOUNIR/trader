"""Fit and freeze the selected Model 15 candidate before acquiring lockbox stocks."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingRegressor

from market_research.model15_research import (
    FEATURES_NEW, PROTOCOL_PATH, REPORT_DIR, _rows, _split,
)
from market_research.model14_experiment import REPORT as MODEL14_REPORT

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = REPORT_DIR / "model_15.joblib"
PROTOCOL_OUT = REPORT_DIR / "protocol.json"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run() -> dict:
    model14_protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    old_protocol = json.loads(MODEL14_REPORT.joinpath("protocol.json").read_text(encoding="utf-8"))
    training_symbols = [s for s in old_protocol["training_symbols"] if s != "SPY"]
    overlap = set(training_symbols).intersection(old_protocol["evaluation_symbols"])
    if overlap:
        raise RuntimeError(f"Consumed evaluation stocks leaked into Model 15 training: {sorted(overlap)}")
    cohorts = {period: _rows(period, training_symbols) for period in ("pre", "mid", "post")}
    development = pd.concat([cohorts[p][0] for p in ("pre", "mid", "post")], ignore_index=True)
    train, calibration, _ = _split(development, "post")
    estimator = HistGradientBoostingRegressor(loss="squared_error", max_iter=60, learning_rate=0.05,
        max_leaf_nodes=7, min_samples_leaf=200, l2_regularization=20.0, early_stopping=False,
        random_state=20261028)
    estimator.fit(train[FEATURES_NEW], train.exec_return.to_numpy(float))
    raw_cal = np.asarray(estimator.predict(calibration[FEATURES_NEW]), float)
    y_cal = calibration.exec_return.to_numpy(float)
    variance = float(np.dot(raw_cal - raw_cal.mean(), raw_cal - raw_cal.mean()))
    slope = float(np.dot(raw_cal - raw_cal.mean(), y_cal - y_cal.mean()) / variance) if variance > 1e-16 else 0.0
    intercept = float(y_cal.mean() - slope * raw_cal.mean())
    payload = {"estimator": estimator, "features": FEATURES_NEW, "calibration_intercept": intercept,
               "calibration_slope": slope, "target": "close(t+3)/open(t+1)-1", "horizon_bars": 3,
               "training_symbols": training_symbols, "periods": {"fit": [2016, 2023], "calibration": ["2024-01-01", "2024-06-28"]},
               "issuer_alias_policy": {"FB": "use for 2016-2021", "META": "use for 2022-2024", "never both in one window": True},
               "model_parameters": {"max_iter": 60, "learning_rate": 0.05, "max_leaf_nodes": 7,
                                    "min_samples_leaf": 200, "l2_regularization": 20.0, "random_state": 20261028}}
    dump(payload, MODEL_PATH)
    digest = _sha(MODEL_PATH)
    protocol = {
        "model_id": 15, "status": "frozen_research_candidate_lockbox_pending",
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "artifact": str(MODEL_PATH.relative_to(ROOT)), "sha256": digest,
        "training_symbols": training_symbols, "training_overlap_with_model14_eval": [],
        "fit_rows": int(len(train)), "calibration_rows": int(len(calibration)),
        "fit_max_timestamp": str(train.timestamp_utc.max()),
        "calibration_min_timestamp": str(calibration.timestamp_utc.min()),
        "calibration_max_timestamp": str(calibration.timestamp_utc.max()),
        "target": payload["target"], "features": FEATURES_NEW,
        "calibration": {"method": "chronological linear calibration on disjoint 2024 H1", "intercept": intercept, "slope": slope},
        "model_parameters": payload["model_parameters"],
        "position_rule": "10% fractional Kelly on cost-adjusted expected return; 1.67% NAV per-tranche cap; gross portfolio cap 1x",
        "round_trip_cost_bps_assumed": 10,
        "execution": {"entry": "next 30-minute bar open", "exit": "close of the third subsequent 30-minute bar"},
        "excluded_evaluation_symbols": old_protocol["evaluation_symbols"],
        "development_evidence": "reports/model15/replay_diagnostics.json; 2024 H2 is development evidence, not a lockbox",
        "lockbox": None,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROTOCOL_OUT.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps({"protocol": protocol, "model14_training_symbols": len(model14_protocol["training_symbols"])}, indent=2))
    return protocol


if __name__ == "__main__":
    run()
