"""Rule-based, cross-sectional intraday strategy research on existing dev cohorts.

Uses only Model 14 training symbols in their existing pre/post development
archives. Evaluation dates were previously observed; this is development, not
an untouched test. It deliberately does not read Model 18 sector data.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_research.model15_research import _rows
from market_research.trend_experiment import ROOT

OUT = ROOT / "reports/model19"
PROTOCOL = ROOT / "reports/model14/protocol.json"
CAPITAL = 100_000.0
RT_COSTS = (10, 25)
RULES = ("resid_mom3", "resid_reversal1", "vwap_reversal", "mom_reversal_mix")


def _score(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["resid_r1"] = d.r1 - d.beta_spy * d.spy_r1
    d["resid_r3"] = d.r3 - d.beta_spy * d.spy_r3
    # Cross-sectional ranks keep the deterministic rules scale-adaptive.
    d["q_mom3"] = d.groupby("timestamp_utc").resid_r3.rank(pct=True)
    d["q_rev1"] = d.groupby("timestamp_utc").resid_r1.rank(pct=True, ascending=False)
    d["q_vwaprev"] = d.groupby("timestamp_utc").vwap_gap.rank(pct=True, ascending=False)
    d["score_resid_mom3"] = d.q_mom3
    d["score_resid_reversal1"] = d.q_rev1
    d["score_vwap_reversal"] = d.q_vwaprev
    d["score_mom_reversal_mix"] = (d.q_mom3 + d.q_rev1) / 2
    # Only rebalance every third 30-minute bar, so consecutive 90-minute
    # tranches do not overlap. Restrict to complete same-session targets.
    local = pd.to_datetime(d.timestamp_utc, utc=True).dt.tz_convert("America/New_York")
    minutes = local.dt.hour * 60 + local.dt.minute - (9 * 60 + 30)
    d["slot"] = (minutes // 30).astype(int)
    d = d.loc[d.slot.isin([0, 3, 6, 9]) & d.exec_return.notna()].copy()
    return d


def _backtest(d: pd.DataFrame, rule: str, cost_bps: int, side: str = "long_short") -> dict:
    score = d[f"score_{rule}"]
    trades = []
    for ts, g in d.assign(_score=score).groupby("timestamp_utc", sort=True):
        g = g.dropna(subset=["_score"])
        n = max(1, int(np.floor(len(g) * 0.20)))
        longs = g.nlargest(n, "_score")
        shorts = g.nsmallest(n, "_score")
        if side == "long_only":
            picks = longs
            signed_weights = np.full(len(picks), 1 / len(picks))
            gross = CAPITAL
            gross_ret = float(np.dot(signed_weights, picks.exec_return.to_numpy()))
        else:
            picks = pd.concat([longs, shorts]).drop_duplicates("symbol")
            longs = picks.nlargest(n, "_score")
            shorts = picks.nsmallest(n, "_score")
            if not len(longs) or not len(shorts):
                continue
            gross = CAPITAL
            # Equal dollar long and short sleeves, each half of gross exposure.
            long_ret = float(longs.exec_return.mean())
            short_ret = float(shorts.exec_return.mean())
            gross_ret = 0.5 * long_ret - 0.5 * short_ret
        fee = gross * cost_bps / 10_000
        pnl = gross * gross_ret - fee
        trades.append((ts, pnl, gross_ret, fee))
    if not trades:
        return {"signals": 0}
    z = pd.DataFrame(trades, columns=["timestamp", "pnl", "gross_return", "fee"])
    z["date"] = pd.to_datetime(z.timestamp, utc=True).dt.tz_convert("America/New_York").dt.date
    daily = z.groupby("date").pnl.sum()
    equity = CAPITAL + daily.cumsum()
    peak = equity.cummax().clip(lower=CAPITAL)
    mdd = float((peak - equity).max())
    spy = d.drop_duplicates("timestamp_utc").set_index("timestamp_utc").spy_exec_return
    spy_rows = []
    for ts in z.timestamp:
        if ts in spy.index:
            spy_rows.append((ts, CAPITAL * float(spy.loc[ts]) - CAPITAL * cost_bps / 10_000))
    spy_daily = pd.DataFrame(spy_rows, columns=["timestamp", "pnl"])
    spy_daily["date"] = pd.to_datetime(spy_daily.timestamp, utc=True).dt.tz_convert("America/New_York").dt.date
    spy_total = float(spy_daily.groupby("date").pnl.sum().sum()) if len(spy_daily) else None
    return {
        "signals": int(len(z)), "net_pnl": float(z.pnl.sum()), "return_pct": float(z.pnl.sum() / CAPITAL * 100),
        "gross_pnl": float((z.pnl + z.fee).sum()), "fees": float(z.fee.sum()),
        "max_drawdown_dollars_daily_exit_marks": mdd,
        "mean_gross_return_bps_per_signal": float(z.gross_return.mean() * 10_000),
        "positive_signal_fraction": float((z.pnl > 0).mean()),
        "spy_same_signal_dates_net_pnl": spy_total,
        "active_vs_spy_same_signal_dates": (float(z.pnl.sum() - spy_total) if spy_total is not None else None),
    }


def _evaluate(era: str, symbols: list[str]) -> dict:
    d, _bars = _rows(era, symbols)
    d = _score(d)
    ts = pd.to_datetime(d.timestamp_utc, utc=True)
    if era == "pre":
        windows = {"validation_2018H1": ("2018-01-01", "2018-07-01"), "development_test_2018H2": ("2018-07-01", "2019-01-01")}
    else:
        windows = {"validation_2024H1": ("2024-01-01", "2024-07-01"), "development_test_2024H2": ("2024-07-01", "2025-01-01")}
    out = {"rows": int(len(d)), "symbols": sorted(d.symbol.unique().tolist()), "windows": {}}
    for name, (start, end) in windows.items():
        w = d.loc[(ts >= start) & (ts < end)].copy()
        out["windows"][name] = {}
        for rule in RULES:
            out["windows"][name][rule] = {}
            for side in ("long_short", "long_only"):
                out["windows"][name][rule][side] = {
                    str(cost): _backtest(w, rule, cost, side) for cost in RT_COSTS
                }
    return out


def run() -> dict:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    symbols = [s for s in protocol["training_symbols"] if s not in {"SPY", "FB", "META"}]
    # Include the point-in-time FB/META issuer alias in its valid era only.
    pre_symbols = sorted(set(symbols + ["FB"]))
    post_symbols = sorted(set(symbols + ["META"]))
    result = {
        "status": "development_only_previous_dates_and_training_cohort",
        "data_scope": "Model 14 training-symbol development archives, pre and post only; Model 14 evaluation symbols excluded",
        "execution": "signals after completed 30-minute bar; next-bar-open to third subsequent bar close; one new tranche every three bars; no overlapping tranches",
        "portfolio": "100k gross per signal; long-short sleeves each 50k, or 100k long-only; fixed round-trip cost on gross notional",
        "rules": list(RULES), "cost_bps": list(RT_COSTS),
        "pre": _evaluate("pre", pre_symbols), "post": _evaluate("post", post_symbols),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rule_strategy_sweep.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
