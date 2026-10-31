"""Internal-only target research using already revealed/training symbols."""
from pathlib import Path
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import market_research.trend_experiment as te

ROOT = Path(__file__).resolve().parents[2]


def main():
    symbols = te.BASE_TRAIN + ["MSFT"]
    results = []
    for horizon in [1, 2, 3, 6]:
        for deadzone in [0, .0005, .001, .002]:
            te.HORIZON = horizon
            data = te.load_symbols(symbols, deadzone=deadzone)
            train, val = te.split_existing(data)
            xtr, ytr = train[te.FEATURES].to_numpy(float), train.label.to_numpy(int)
            xva, yva = val[te.FEATURES].to_numpy(float), val.label.to_numpy(int)
            for leaves in [None, 7]:
                if leaves is None:
                    model = te.fit_logistic(xtr, ytr, .1)
                    p = te.predict(model, xva)
                    family = "logistic"
                else:
                    model = HistGradientBoostingClassifier(max_iter=100, learning_rate=.03,
                            max_leaf_nodes=leaves, min_samples_leaf=120,
                            l2_regularization=20, random_state=20261023, early_stopping=False)
                    model.fit(xtr, ytr)
                    p = model.predict_proba(xva)[:,1]
                    family = "hist"
                result = te.metrics(yva, p, val.future_return.to_numpy())
                results.append({"horizon": horizon, "deadzone": deadzone, "family": family, **result})
                print(json.dumps(results[-1]))
    from datetime import datetime, timezone
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out = ROOT / f"reports/trend_experiment/target_research_corrected_{run_id}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
