r"""Reproducible exploratory event studies; cached local data only.

First run causal_context.py --fetch, then this script. Outputs reports/causal.
No brokerage endpoints, source mutations, or automated news attribution.
"""
from pathlib import Path
from datetime import datetime, timezone
from html import escape
import json
import hashlib
import sys
from importlib.metadata import version
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/causal"
CACHE = ROOT / "data/context/causal"

METHOD_SOURCES = [
    ("ECA repository (research code, not actively maintained; MATLAB)", "https://github.com/jmmccracken/ECA"),
    ("Facure: counterfactuals and confounding", "https://matheusfacure.github.io/python-causality-handbook/01-Introduction-To-Causality.html"),
    ("Facure: difference in differences", "https://matheusfacure.github.io/python-causality-handbook/13-Difference-in-Differences.html"),
    ("Facure: synthetic control", "https://matheusfacure.github.io/python-causality-handbook/15-Synthetic-Control.html"),
    ("MacKinlay: Event Studies in Economics and Finance", "https://www.bu.edu/econ/files/2011/01/MacKinlay-1996-Event-Studies-in-Economics-and-Finance.pdf"),
]
MACRO_SOURCES = [
    ("BLS CPI release, November 10 2022 at 08:30 ET", "https://www.bls.gov/news.release/archives/cpi_11102022.htm"),
    ("FOMC statement, February 1 2023 at 14:00 ET", "https://www.federalreserve.gov/newsevents/pressreleases/monetary20230201a.htm"),
    ("Powell press conference, February 1 2023", "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20230201.pdf"),
    ("FB/META same-security ticker change", "https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2022-125"),
    ("First Republic earnings release schedule", "https://www.nasdaq.com/press-release/first-republic-bank-announces-first-quarter-2023-earnings-release-date-2023-04-07"),
    ("US payroll release, May 5 2023", "https://www.bls.gov/news.release/archives/empsit_05052023.htm"),
    ("US payroll release, May 3 2024", "https://www.bls.gov/news.release/archives/empsit_05032024.htm"),
]


def benchmark_panel():
    parts = []
    for symbol in ["SPY", "QQQ", "KRE", "XLI"]:
        p = CACHE / f"{symbol}_daily.parquet"
        if not p.exists():
            continue
        df = pd.read_parquet(p).sort_values("session_date")
        df["benchmark_return_pct"] = df.Close.pct_change(fill_method=None)*100
        df["benchmark_adjusted_return_pct"] = df["Adj Close"].pct_change(fill_method=None)*100
        df["benchmark"] = symbol
        parts.append(df[["session_date", "benchmark", "benchmark_return_pct", "benchmark_adjusted_return_pct"]])
    if not parts:
        raise RuntimeError("No benchmark cache; run causal_context.py --fetch first")
    return pd.concat(parts, ignore_index=True)


def estimate(data, event, benchmark, left=-140, right=-21, min_coverage=0, adjusted=False):
    idx = int(data.loc[data.session_date.eq(event["reaction_date"]), "calendar_index"].iloc[0])
    train = data.loc[data.calendar_index.between(idx+left, idx+right) & data.minute_coverage_pct.ge(min_coverage)].copy()
    # Exclude the known mechanical wealth transfer from raw-parent-return fitting.
    if event["symbol"] == "MMM":
        train = train.loc[train.session_date.ne("2024-04-01")]
    xcol = "benchmark_adjusted_return_pct" if adjusted else "benchmark_return_pct"
    train = train.dropna(subset=["close_return_pct", xcol])
    base = {"event_id": event["id"], "benchmark": benchmark, "left": left, "right": right,
            "min_coverage": min_coverage, "adjusted_benchmark": adjusted, "training_n": len(train)}
    if len(train) < 60:
        return {**base, "status": "insufficient pre-event data"}, None
    x = np.column_stack([np.ones(len(train)), train[xcol].to_numpy()])
    y = train.close_return_pct.to_numpy()
    beta, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != 2:
        return {**base, "status": "rank deficient"}, None
    residual = y-x@beta
    inverse = np.linalg.inv(x.T@x)
    hat = np.sum((x@inverse)*x, axis=1)
    loo = residual/(1-hat)
    lo, hi = np.quantile(loo, [.025, .975])
    r2 = 1-np.sum(residual**2)/np.sum((y-y.mean())**2)
    day = data.loc[data.session_date.eq(event["reaction_date"])].iloc[0]
    predicted = beta[0]+beta[1]*day[xcol]
    ar = day.close_return_pct-predicted
    result = {**base, "status": "ok", "training_first": train.session_date.min(), "training_last": train.session_date.max(),
              "alpha_pct": beta[0], "beta": beta[1], "r_squared": r2,
              "training_under95pct_minutes": int(train.minute_coverage_pct.lt(95).sum()),
              "expected_return_pct": predicted, "abnormal_return_pp": ar,
              "loo_residual_q025_pp": lo, "loo_residual_q975_pp": hi,
              "descriptive_loo_tail_fraction": (1+np.sum(np.abs(loo)>=abs(ar)))/(len(loo)+1)}
    paths = data.loc[data.calendar_index.between(idx-10, idx+5)].copy()
    paths["event_day"] = paths.calendar_index-idx
    paths["expected_return_pct"] = beta[0]+beta[1]*paths[xcol]
    paths["abnormal_return_pp"] = paths.close_return_pct-paths.expected_return_pct
    paths["event_id"] = event["id"]
    paths["benchmark"] = benchmark
    paths["estimation_last_index"] = train.calendar_index.max()
    for a, b, name in [(-5, -1, "pre5_car_pp"), (-1, 1, "car_minus1_plus1_pp"), (0, 1, "car_0_plus1_pp"), (0, 2, "car_0_plus2_pp")]:
        selected = paths.loc[paths.event_day.between(a, b)]
        result[name] = selected.abnormal_return_pp.sum() if len(selected)==b-a+1 and selected.abnormal_return_pp.notna().all() else np.nan
    return result, paths


def macro_analysis(panel, benchmarks):
    rows = []
    for date, title in [("2022-11-10", "CPI release"), ("2023-02-01", "FOMC day")]:
        for row in panel.loc[panel.session_date.eq(date)].itertuples():
            rows.append({"date": date, "episode": title, "asset": row.symbol, "close_return_pct": row.close_return_pct,
                         "overnight_return_pct": row.overnight_return_pct, "intraday_return_pct": row.intraday_return_pct})
        for row in benchmarks.loc[benchmarks.session_date.eq(date)].itertuples():
            rows.append({"date": date, "episode": title, "asset": row.benchmark, "close_return_pct": row.benchmark_return_pct,
                         "overnight_return_pct": np.nan, "intraday_return_pct": np.nan})
    intraday = []
    # Exact minute closes only. No interpolation through halts or missing bars.
    for p in (ROOT / "data/normalized").glob("*.parquet"):
        df = pd.read_parquet(p, columns=["symbol", "timestamp_utc", "close"])
        df.timestamp_utc = pd.to_datetime(df.timestamp_utc, utc=True)
        ny = df.timestamp_utc.dt.tz_convert("America/New_York")
        mask = ny.dt.strftime("%Y-%m-%d").eq("2023-02-01")
        day = df.loc[mask].copy()
        if day.empty:
            continue
        day["time"] = ny.loc[mask].dt.strftime("%H:%M")
        day = day.set_index("time")
        if "13:59" not in day.index:
            continue
        base = day.loc["13:59", "close"]
        for t in ["14:09", "14:29", "14:59", "15:59"]:
            intraday.append({"symbol": str(day.symbol.iloc[0]), "date": "2023-02-01", "baseline_last_bar": "13:59",
                             "end_last_bar": t, "return_from_preannouncement_pct": (day.loc[t, "close"]/base-1)*100 if t in day.index else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(intraday)


def md_table(df):
    def fmt(x):
        if pd.isna(x):
            return "—"
        return f"{x:.2f}" if isinstance(x, (float, np.floating)) else str(x).replace("|", "/")
    return "\n".join(["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"]*len(df.columns)) + " |"] +
                     ["| " + " | ".join(fmt(x) for x in row) + " |" for row in df.itertuples(index=False, name=None)])


def html_table(df):
    return "<div class='scroll'>"+df.to_html(index=False, border=0, na_rep="—", float_format=lambda x: f"{x:.2f}")+"</div>"


def chart(rows):
    # Standalone SVG: measured return and model residual on a shared zero-centered axis.
    rows = rows.loc[rows.close_return_pct.notna()].copy()
    maximum = max(10, rows.close_return_pct.abs().max(), rows.abnormal_return_pp.abs().max())*1.1
    width, left, plotwidth = 1000, 290, 570
    zero = left+plotwidth/2
    height = 70+len(rows)*44
    parts = [f"<svg role='img' aria-label='Event returns and market model residuals' viewBox='0 0 {width} {height}'>",
             f"<line x1='{zero}' y1='35' x2='{zero}' y2='{height-15}' stroke='#65758c'/>"
             "<text x='10' y='20' font-size='14'>Blue: observed return (%) · Orange: SPY model residual (pp)</text>"]
    for n, row in enumerate(rows.itertuples()):
        y = 50+n*44
        parts.append(f"<text x='8' y='{y+11}' font-size='13'>{escape(row.symbol+' '+row.reaction_date)}</text>")
        for dy, val, color in [(0, row.close_return_pct, "#236cba"), (15, row.abnormal_return_pp, "#c76a22")]:
            if pd.notna(val):
                amount = abs(val)/maximum*(plotwidth/2)
                x = zero if val>=0 else zero-amount
                parts.append(f"<rect x='{x:.2f}' y='{y+dy}' width='{amount:.2f}' height='11' fill='{color}'><title>{val:.2f}</title></rect>")
        ar = f"{row.abnormal_return_pp:+.2f} pp" if pd.notna(row.abnormal_return_pp) else "model unavailable"
        parts.append(f"<text x='870' y='{y+11}' font-size='12'>{row.close_return_pct:+.2f}%</text><text x='870' y='{y+26}' font-size='12'>{ar}</text>")
    parts.append("</svg>")
    return "".join(parts)


def path_chart(path, title):
    frame = path.sort_values("event_day").copy()
    # Start immediately before day -5. Never connect a line across absent daily returns.
    frame = frame.loc[frame.event_day.between(-5, 3)]
    if len(frame)!=9 or frame.abnormal_return_pp.isna().any():
        return ""
    frame["cumulative"] = frame.abnormal_return_pp.cumsum()
    points = [(-6, 0)]+list(zip(frame.event_day, frame.cumulative))
    low, high = min(0, frame.cumulative.min()), max(0, frame.cumulative.max())
    margin = max(1, (high-low)*.1)
    low, high = low-margin, high+margin
    x = lambda n: 60+(n+6)/9*600
    y = lambda n: 210-(n-low)/(high-low)*160
    poly = " ".join(f"{x(a):.1f},{y(b):.1f}" for a,b in points)
    return (f"<h3>{escape(title)}</h3><svg viewBox='0 0 720 255' role='img' aria-label='Cumulative model residual event path'>"
            f"<line x1='60' y1='{y(0)}' x2='660' y2='{y(0)}' stroke='#aaa'/>"
            f"<line x1='{x(-.5)}' y1='40' x2='{x(-.5)}' y2='215' stroke='#c76a22' stroke-dasharray='4'/>"
            f"<polyline fill='none' stroke='#236cba' stroke-width='3' points='{poly}'/>"
            f"<text x='5' y='45'>{high:.1f}</text><text x='5' y='210'>{low:.1f}</text>"
            "<text x='60' y='240'>−6</text><text x='450' y='240'>0: reaction session</text><text x='650' y='240'>+3</text>"
            "<text x='60' y='20'>Cumulative abnormal return (percentage points), starting before day −5</text></svg>")


def main():
    events = json.loads((OUT / "event_catalog.json").read_text(encoding="utf-8"))
    panel = pd.read_parquet(OUT / "daily_panel.parquet")
    benchmarks = benchmark_panel()
    results, sensitivities, paths = [], [], []
    for event in events:
        own = panel.loc[panel.symbol.eq(event["symbol"])]
        day = own.loc[own.session_date.eq(event["reaction_date"])].iloc[0]
        result = {**event, **{c: day[c] for c in ["close_return_pct", "overnight_return_pct", "intraday_return_pct", "minute_coverage_pct", "endpoint_complete"]}}
        if event["category"] == "corporate_action":
            result["status"] = "excluded: mechanical corporate action"
        else:
            for benchmark, left, right, coverage, adjusted, name in [
                ("SPY", -140, -21, 0, False, "primary"),
                (event["alternative_benchmark"], -140, -21, 0, False, "alternative benchmark"),
                ("SPY", -80, -11, 0, False, "shorter training"),
                ("SPY", -260, -21, 0, False, "longer training"),
                ("SPY", -140, -21, 95, False, "training coverage >=95%"),
                ("SPY", -140, -21, 0, True, "adjusted ETF sensitivity"),
            ]:
                data = own.merge(benchmarks.loc[benchmarks.benchmark.eq(benchmark)], on="session_date", how="left", validate="one_to_one")
                model, path = estimate(data, event, benchmark, left, right, coverage, adjusted)
                model["specification"] = name
                sensitivities.append(model)
                if name == "primary":
                    result.update(model)
                    if path is not None:
                        paths.append(path)
        results.append(result)
    result = pd.DataFrame(results)
    sensitivity = pd.DataFrame(sensitivities)
    ranges = sensitivity.loc[sensitivity.status.eq("ok")].groupby("event_id").abnormal_return_pp.agg(sensitivity_min_pp="min", sensitivity_max_pp="max")
    result = result.merge(ranges, left_on="id", right_index=True, how="left", validate="one_to_one")
    path = pd.concat(paths, ignore_index=True)
    # Do not turn the spin-off into evidence of a company-value collapse.
    mmm = panel.loc[panel.symbol.eq("MMM") & panel.session_date.eq("2024-04-01")].iloc[0]
    solv = pd.read_parquet(CACHE / "SOLV_daily.parquet")
    solv_close = float(solv.loc[solv.session_date.eq("2024-04-01"), "Close"].iloc[0])
    wealth = {"date": "2024-04-01", "mmm_previous_regular_close": float(mmm.previous_close), "mmm_regular_close": float(mmm.close),
              "solv_daily_close": solv_close, "solv_shares_per_mmm": .25,
              "raw_parent_price_return_pct": float(mmm.close_return_pct),
              "reconstructed_wealth_return_pct": float((mmm.close+.25*solv_close)/mmm.previous_close-1)*100,
              "limitation": "Mixed local last-minute MMM close and Yahoo SOLV daily close; excludes taxes, fees and other cash flows."}
    (OUT / "spinoff_reconstruction.json").write_text(json.dumps(wealth, indent=2), encoding="utf-8")
    macro, intraday = macro_analysis(panel, benchmarks)
    top = panel.loc[panel.close_return_pct.abs().groupby(panel.symbol).rank(method="first", ascending=False).le(5),
                    ["symbol", "session_date", "close_return_pct", "overnight_return_pct", "intraday_return_pct", "minute_coverage_pct"]].copy()
    event_keys = {(e["symbol"], e["reaction_date"]) for e in events}
    top["catalogued_reaction_date"] = [(r.symbol, r.session_date) in event_keys for r in top.itertuples()]
    top["context_status"] = np.where(top.catalogued_reaction_date, "catalogued; see event caveats", "unassigned; no cause inferred from magnitude")
    for name, df in [("event_results", result), ("model_sensitivity", sensitivity), ("event_paths", path),
                     ("macro_context", macro), ("fomc_intraday", intraday), ("largest_moves", top)]:
        df.to_parquet(OUT / f"{name}.parquet", index=False)
    sources = [{"title": t, "url": u, "role": "method"} for t,u in METHOD_SOURCES]+[
        {"title": t, "url": u, "role": "context"} for t,u in MACRO_SOURCES]+[
        {"title": e["title"], "url": e["source"], "publication_date": e["source_date"], "role": "historical event"} for e in events]
    (OUT / "sources.json").write_text(json.dumps({"checked_date": "2026-10-23", "sources": sources}, indent=2), encoding="utf-8")
    generate_reports(result, sensitivity, path, macro, intraday, top, wealth, sources)
    # Mathematical and data-integrity checks directly target failure modes of this analysis.
    valid = panel.dropna(subset=["close_return_pct", "overnight_return_pct", "intraday_return_pct"])
    assert np.allclose((1+valid.overnight_return_pct/100)*(1+valid.intraday_return_pct/100), 1+valid.close_return_pct/100)
    assert not panel.duplicated(["symbol", "session_date"]).any()
    assert (path.estimation_last_index < path.calendar_index-path.event_day-10).all()
    assert sensitivity.loc[sensitivity.status.eq("ok"), "training_n"].ge(60).all()
    assert result.loc[result.id.eq("ATVI_20220118"), "status"].iloc[0] == "insufficient pre-event data"
    assert result.loc[result.category.eq("corporate_action"), "abnormal_return_pp"].isna().all()
    assert (result.loc[result.status.eq("ok"), "close_return_pct"]-result.loc[result.status.eq("ok"), "expected_return_pct"]-result.loc[result.status.eq("ok"), "abnormal_return_pp"]).abs().max()<1e-10
    # Recover a known injected shock, then check that changing event/future outcomes
    # cannot change the pre-event model. This tests identification-code leakage.
    index = np.arange(300)
    market = np.sin(index*.37)*2
    synthetic = pd.DataFrame({"calendar_index": index, "session_date": [f"d{i}" for i in index],
                             "minute_coverage_pct": 100., "benchmark_return_pct": market,
                             "close_return_pct": .1+1.2*market})
    synthetic.loc[220, "close_return_pct"] += 7
    synthetic_event = {"symbol": "TEST", "id": "known_truth", "reaction_date": "d220"}
    known, _ = estimate(synthetic, synthetic_event, "TEST")
    assert np.isclose(known["abnormal_return_pp"], 7)
    synthetic.loc[220:, "close_return_pct"] += 50
    changed, _ = estimate(synthetic, synthetic_event, "TEST")
    assert np.isclose(changed["beta"], known["beta"]) and np.isclose(changed["alpha_pct"], known["alpha_pct"])
    assert np.isclose(changed["abnormal_return_pp"], 57)
    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(), "canonical_series": panel.symbol.nunique(), "events": len(result),
                "models_available": int(result.status.eq("ok").sum()), "verification": "passed: decomposition, unique dates, no event leakage, sample thresholds, corporate-action exclusion, AR reconciliation, known injected effect and post-event perturbation",
                "python": sys.version, "packages": {name: version(name) for name in ["pandas", "numpy", "pyarrow", "exchange-calendars", "yfinance"]},
                "input_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__), ROOT/"src/market_research/causal_context.py", OUT/"event_catalog.json", OUT/"daily_panel.parquet"]+sorted(CACHE.glob("*.parquet"))+sorted((ROOT/"data/normalized").glob("*.parquet"))}}
    (OUT / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(result[["id", "close_return_pct", "overnight_return_pct", "intraday_return_pct", "abnormal_return_pp", "training_n", "status"]].to_string(index=False))
    print("Spin-off:", json.dumps(wealth))
    print("Verified:", manifest["verification"])


def generate_reports(result, sensitivity, path, macro, intraday, top, wealth, sources):
    lookup = result.set_index("id")
    def observed(event):
        return lookup.loc[event, "close_return_pct"]
    def abnormal(event):
        return lookup.loc[event, "abnormal_return_pp"]
    findings = [
        f"Meta shows large repricing around earnings bundles: {observed('META_20220203'):+.2f}% on February 3, 2022; {observed('META_20221027'):+.2f}% on October 27, 2022; {observed('META_20230202'):+.2f}% on February 2, 2023; and {observed('META_20240202'):+.2f}% on February 2, 2024. Most of these changes occurred between the previous close and the next open. Growth, costs and capital-return expectations are plausible channels, but their individual effects are not separated.",
        f"Twitter moves in opposite directions as deal prospects change: {observed('TWTR_20220404'):+.2f}% on the stake disclosure, {observed('TWTR_20220711'):+.2f}% after the purported termination, and {observed('TWTR_20221004'):+.2f}% on renewed closing intentions. The October 4 SPY-model residual is {abnormal('TWTR_20221004'):+.2f} pp. This pattern is consistent with takeover-probability repricing, conditional on other news and market-model assumptions.",
        f"Activision gained {observed('ATVI_20220118'):+.2f}% on the acquisition-announcement reaction day but fell {lookup.loc['ATVI_20220118', 'intraday_return_pct']:+.2f}% after opening. It later fell {observed('ATVI_20230426'):+.2f}% on the CMA block and rose {observed('ATVI_20230711'):+.2f}% around the US court decision. The first jump has no fitted model because local prehistory is too short.",
        f"Bank distress dwarfs ordinary co-movement: SIVB fell {observed('SIVB_20230309'):+.2f}% on March 9, 2023, and FRC fell {observed('FRC_20230313'):+.2f}% on March 13. Sector adjustment attenuates both, but KRE itself was affected. Funding stress and equity-value risk fit the documented timeline; the regressions cannot isolate deposit outflows from confidence feedback and other bank news.",
        f"Apple's May 2024 earnings/capital-return bundle coincides with {observed('AAPL_20240503'):+.2f}% observed and {abnormal('AAPL_20240503'):+.2f} pp market-model abnormal return. Apple Intelligence was announced June 10, when the stock fell {observed('AAPL_20240610'):+.2f}%; its day 0:+1 CAR is {lookup.loc['AAPL_20240610', 'car_0_plus1_pp']:+.2f} pp. Moving the announcement date to the subsequent gain would misstate the evidence.",
        f"The formal 3M settlement-announcement day rose {observed('MMM_20230829'):+.2f}% but has only {abnormal('MMM_20230829'):+.2f} pp SPY-model residual, with sensitivity estimates from {lookup.loc['MMM_20230829', 'sensitivity_min_pp']:+.2f} to {lookup.loc['MMM_20230829', 'sensitivity_max_pp']:+.2f} pp. The preceding five-session CAR is {lookup.loc['MMM_20230829', 'pre5_car_pp']:+.2f} pp. This weakens a simple claim that the formal announcement caused the day's entire gain and makes prior information important to investigate.",
        f"3M's spin-off day is a mechanical-price counterexample: {wealth['raw_parent_price_return_pct']:+.2f}% in the raw parent price versus {wealth['reconstructed_wealth_return_pct']:+.2f}% after adding the distributed SOLV shares at closing value. Its July 26, 2024 earnings day instead shows {observed('MMM_20240726'):+.2f}% observed return and {abnormal('MMM_20240726'):+.2f} pp market-model residual.",
    ]
    compact = result[["symbol", "reaction_date", "close_return_pct", "overnight_return_pct", "intraday_return_pct", "abnormal_return_pp", "training_n", "status"]]
    diagnostics = result[["id", "r_squared", "pre5_car_pp", "car_minus1_plus1_pp", "car_0_plus1_pp", "car_0_plus2_pp", "loo_residual_q025_pp", "loo_residual_q975_pp", "minute_coverage_pct"]]
    methods = [
        "Question: what information plausibly changed expected cash flows, discount rates, takeover probabilities or shareholder rights around a large observed move? We estimate event-linked price repricing; we do not identify a universal causal law for stock increases and decreases.",
        "Events were chosen retrospectively from historical knowledge and price inspection, not preregistered. The catalog distinguishes history-led from price-led choices, but both can have selection bias. Largest-move rows without documented context remain explicitly unassigned. These examples cannot estimate how often news causes gains, average treatment effects, or a trading strategy's performance.",
        "Daily target prices come from local Alpaca raw minute bars within the XNYS regular session. Consecutive exchange sessions and both closing endpoint minutes are required for close-to-close returns; overnight returns additionally require the opening minute. Daily prices use last-minute closes, which can differ from official closing-auction prices. FB 2021 and META 2022–2024 are combined only in the analysis, following the documented same-CUSIP ticker change; source ticker labels are retained.",
        "Returns: close-to-close = 100*(close/previous close−1); overnight = 100*(open/previous close−1); intraday = 100*(close/open−1). Overnight and intraday compound; they do not add exactly. Missing years and delistings never receive zero returns or forward-filled prices. Sparse sessions remain flagged; missing minute bars do not by themselves prove lost data.",
        "Primary model: stock daily price return = alpha + beta*SPY daily price return + residual. Fit ordinary least squares using exchange-session offsets −140 through −21, at least 60 valid observations, then freeze coefficients before examining the event. Abnormal return (AR) is observed minus model-expected return, in percentage points. CAR sums AR over the stated session offsets; it is not a compounded investment return.",
        "Sensitivity models use the same event with QQQ (technology/takeovers), KRE (banks), or XLI (3M); shorter (−80:−11) and longer (−260:−21) histories; training sessions with at least 95% minute coverage; and adjusted SPY returns. Targets remain raw price returns. The known 3M spin-off day is removed from MMM training; other dividend/corporate-action adjustments are not reconstructed.",
        "The 2.5% and 97.5% reference values are quantiles of leave-one-out pre-event regression residuals. The descriptive tail fraction ranks the event AR against their absolute magnitudes with a plus-one correction. They are neither causal confidence intervals nor calibrated hypothesis-test p-values; heavy tails, serial dependence, regime changes and hindsight selection remain. No statistical-significance claims or multiple-testing discoveries are made.",
        "SPY/QQQ include some target stocks and respond to common shocks; KRE can contain distressed banks and transmit contagion. These are imperfect expected-return benchmarks, not untreated controls. Sector subtraction can remove part of the shock of interest. Small R-squared, unstable beta, anticipation and low coverage weaken interpretation. FRC April 2023 training can span the March crisis. A plausible conditional attribution requires event timing, a stable counterfactual relation, no other relevant simultaneous news, limited leakage and spillovers, and consistent security rights.",
        "We do not fit difference in differences or synthetic controls: this archive lacks a sufficiently broad, comparable untreated donor panel, several shocks spill across firms, and common pretrends are unverified. ECA's time-series tools could generate leads after stationarity, lag and common-cause checks, but predictive direction, transfer entropy or reconstruction alone would not identify a news mechanism. The event-study design makes the treatment and timing more explicit.",
    ]
    mechanism_text = [
        ("Earnings and guidance", "Investors price future outcomes relative to prior expectations. A company can report declining revenue and still rise if the outlook or capital-return news is more favorable than expected. Our data lack contemporaneous consensus estimates, so a specific surprise channel remains a hypothesis."),
        ("Takeovers and regulatory decisions", "A target's price can be viewed as a weighted value of a deal outcome and a standalone outcome, discounted for time and risk. Changes in completion probability can move the price sharply even if operating results do not change. We do not infer an exact probability without estimating standalone value and discounting."),
        ("Bank distress", "Funding outflows, capital needs and asset losses can threaten the value left for shareholders. Prices, withdrawals and confidence can reinforce one another. A deposit guarantee does not automatically protect common equity; no single-day regression disentangles this feedback."),
        ("Inflation and monetary policy", "Macro information can change expected interest rates, required returns and future activity. Many assets respond together. Subtracting a moving benchmark does not measure the total policy effect. To estimate a rate-surprise effect, we would need an expectations-based shock series such as futures surprises, rather than the announced rate change alone."),
        ("Corporate actions", "A lower quoted parent share price can reflect a transfer of value into another security. A raw stock-price chart can therefore show a sharp fall without a comparable loss to the original shareholder."),
    ]
    spinoff_text = (f"On April 1, 2024 MMM's local raw close-to-close price return was {wealth['raw_parent_price_return_pct']:+.2f}%. "
                    f"Adding the distributed 0.25 SOLV share at its Yahoo daily close gives a reconstructed shareholder wealth change of {wealth['reconstructed_wealth_return_pct']:+.2f}%. "
                    "The difference is a mechanical security-rights adjustment, not evidence that the parent suddenly destroyed that much shareholder value. "
                    "This reconstruction mixes vendors and closing definitions and excludes taxes/fees; it is a diagnostic, not a complete adjusted-return series.")
    macro_text = ("BLS released October 2022 CPI at 08:30 ET on November 10: headline inflation was 7.7% year over year and core inflation rose 0.3% month over month. "
                  "The table measures that session's asset moves, without calling the CPI print an identified surprise because no pre-release consensus is stored. "
                  "On February 1, 2023 the Fed released its statement at 14:00 ET, raising the target range by 25 basis points to 4.50–4.75%. "
                  "Intraday responses below use the close of the 13:59 minute as the pre-release baseline and the close of the labeled minute as the endpoint. "
                  "Later windows include press-conference information and other news; the table does not isolate the mechanical effect of a 25-basis-point hike. Meta's own earnings arrived after this regular session.")
    next_steps = [
        "Build a dated event panel for all earnings and regulatory decisions, including small and zero responses, with first-public timestamps and contemporaneous expectations. This avoids selecting only dramatic returns.",
        "Expand matched controls: technology/advertising peers, industrial firms, and bank balance-sheet exposures such as uninsured deposits and duration. Use pre-event comparability and explicit spillover exclusions before fitting panel difference in differences or synthetic controls.",
        "Store adjusted shareholder-return products and cash distributions separately from raw quotes, including spin-offs, cash acquisition proceeds and post-delisting outcomes. The current archive cannot measure terminal shareholder losses or recoveries after trading stops.",
        "For macro effects, obtain historical futures-based rate surprises or release-consensus surprises and use narrow first-public event windows. For bank contagion, distinguish direct exposure, common funding vulnerabilities and contemporaneous feedback.",
    ]
    md = ["# What plausibly moved these stocks? Exploratory causal analysis", "Generated "+datetime.now(timezone.utc).isoformat(),
          "This is a source-linked, retrospective event study of seven canonical stock series (eight source tickers). It separates observed price movements, model-based abnormal returns and causal interpretations. No result below is a separately identified causal effect of an individual headline.",
          "## Main findings", *["- "+m for m in findings],
          "## How to read the results", *["- "+m for m in methods], "## Observed moves and SPY-model residuals", md_table(compact),
          "## Mechanisms to investigate", *[f"### {t}\n\n{s}" for t,s in mechanism_text],
          "## Important counterexample: 3M spin-off", spinoff_text,
          f"[3M distribution terms]({result.loc[result.id.eq('MMM_20240401'), 'source'].iloc[0]})",
          "## Event-by-event context and limits"]
    cards = []
    for row in result.to_dict("records"):
        ar = f"{row['abnormal_return_pp']:+.2f} pp" if pd.notna(row.get("abnormal_return_pp")) else "unavailable / excluded"
        fact = f"Observed close-to-close {row['close_return_pct']:+.2f}%; overnight {row['overnight_return_pct']:+.2f}%; intraday {row['intraday_return_pct']:+.2f}%; SPY-model residual {ar}."
        context = f"Announcement: {row['announcement_date']}. Reaction session: {row['reaction_date']}. Timing: {row['timing']}. Selection: {row['selection']}."
        source_line = f"[Historical source]({row['source']}) (source publication {row['source_date']}; reviewed 2026-10-23)."
        md += [f"### {row['symbol']}: {row['title']}", context, fact, "Plausible mechanism: "+row["mechanism"], "Attribution limit: "+row["caveat"], source_line]
        cards.append(f"<article><h3>{escape(row['symbol']+': '+row['title'])}</h3><p>{escape(context)}</p><p><strong>{escape(fact)}</strong></p>"
                     f"<p>Plausible mechanism: {escape(row['mechanism'])}</p><p class='limit'>Attribution limit: {escape(row['caveat'])}</p>"
                     f"<a href='{escape(row['source'], quote=True)}'>Historical source</a> · published {escape(row['source_date'])}</article>")
    sensitivity_view = sensitivity[["event_id", "specification", "benchmark", "training_n", "abnormal_return_pp", "status"]]
    md += ["## Anticipation, event windows and model diagnostics", "Pre5 CAR uses −5:−1; other CAR columns name their windows. These are descriptive sums of residuals. Missing windows remain blank, including SIVB's post-failure sessions. Event-day 0 is the reaction session, not always the calendar day of the announcement.", md_table(diagnostics),
           "## Sensitivity to benchmark and estimation choices", md_table(sensitivity_view), "## Macro context", macro_text, md_table(macro),
           "## FOMC intraday response from 13:59 close", md_table(intraday),
           "## Largest observed moves, including unassigned cases", "Five largest absolute daily returns per canonical series, among returns meeting closing-endpoint requirements. This ranking is a discovery aid, not a representative treatment sample. A price-selected story can be plausible and still suffer hindsight bias.", md_table(top),
           "## What would make the causal analysis stronger", *["- "+s for s in next_steps],
           "## Sources and reproducibility", "Methods and context: "+"; ".join(f"[{t}]({u})" for t,u in METHOD_SOURCES+MACRO_SOURCES)+".",
           "Daily benchmark caches: data/context/causal, with URLs, retrieval dates and SHA-256 hashes in provenance.json. Target prices: existing local normalized files. Run `.\\.venv\\Scripts\\python.exe src/market_research/causal_context.py --fetch` to fetch missing context caches and rebuild the daily panel; run `.\\.venv\\Scripts\\python.exe src/market_research/causal_analysis.py` to regenerate this report from caches. Event definitions and context links are in reports/causal/event_catalog.json. The HTML report is standalone and needs no network access to display. Source prices, security-master tables, prior EDA reports and acquisition state are preserved."]
    (OUT / "CAUSAL_REPORT.md").write_text("\n\n".join(md)+"\n", encoding="utf-8")
    body = "<h1>What plausibly moved these stocks?</h1><p class='intro'>Exploratory causal analysis · seven stock series · historical event context · benchmark sensitivity</p>"
    body += "<p class='notice'>Observed movements and market-model residuals support investigations of mechanisms. They do not isolate the causal effect of each headline.</p>"
    body += "<nav><a href='#results'>Results</a> · <a href='#cases'>Event stories</a> · <a href='#macro'>Macro context</a> · <a href='#methods'>Methods</a></nav>"
    body += "<h2>Main findings</h2><ul>"+"".join("<li>"+escape(m)+"</li>" for m in findings)+"</ul>"
    body += "<h2 id='results'>Observed returns and market-adjusted residuals</h2>"+chart(result)+html_table(compact)
    body += "<h2>How information can move a stock</h2>"
    body += "<div class='flow'>New information → revised cash flows / discount rates / deal probability → repricing<br><small>Prior expectations and common macro conditions affect both the information context and prices. Volume and sentiment can respond to prices as well as influence them.</small></div>"
    for t,s in mechanism_text:
        body += f"<h3>{escape(t)}</h3><p>{escape(s)}</p>"
    body += "<h2>A raw-price decline that changes after accounting for shareholder rights</h2><p>"+escape(spinoff_text)+"</p>"
    body += "<h2>Anticipation and delayed responses</h2>"
    for eid in ["META_20221027", "TWTR_20221004", "AAPL_20240610", "MMM_20230829"]:
        body += path_chart(path.loc[path.event_id.eq(eid)], eid)
    body += "<h2 id='cases'>Historical event context</h2>"+"".join(cards)
    body += "<h2>Window and model diagnostics</h2>"+html_table(diagnostics)
    body += "<details><summary>All benchmark and training-window sensitivity results</summary>"+html_table(sensitivity_view)+"</details>"
    body += "<h2 id='macro'>Macro episodes</h2><p>"+escape(macro_text)+"</p>"+html_table(macro)+html_table(intraday)
    body += "<details><summary>Largest moves, including unassigned cases</summary>"+html_table(top)+"</details>"
    body += "<h2 id='methods'>Design and interpretation</h2><ol>"+"".join("<li>"+escape(m)+"</li>" for m in methods)+"</ol>"
    body += "<h2>What would strengthen identification</h2><ul>"+"".join("<li>"+escape(s)+"</li>" for s in next_steps)+"</ul>"
    body += "<h2>Methods and context sources</h2><ul>"+"".join(f"<li><a href='{escape(u, quote=True)}'>{escape(t)}</a></li>" for t,u in METHOD_SOURCES+MACRO_SOURCES)+"</ul>"
    css = "body{font:16px system-ui;max-width:1150px;margin:36px auto;padding:0 24px;background:#f5f7fb;color:#172e45;line-height:1.6}h1{font-size:38px;line-height:1.2}h1,h2,h3{color:#123f66}h2{margin-top:38px}a{color:#176ba0}nav,.flow,.notice{background:#e5edf7;padding:18px;border-radius:8px}.notice{border-left:5px solid #c76a22}article{background:white;border:1px solid #dae2eb;padding:20px;margin:20px 0;border-radius:8px}.limit{border-left:3px solid #c76a22;padding-left:12px}.scroll{overflow:auto;background:white}table{border-collapse:collapse;font-size:12px;width:100%}td,th{white-space:nowrap;padding:9px;text-align:right;border-bottom:1px solid #ddd}th{background:#e7eff7}li{margin:12px 0}svg{width:100%;height:auto;background:white;margin:14px 0}summary{cursor:pointer;font-weight:bold;padding:14px;background:#e5edf7}.intro{font-size:18px;color:#53697d}small{color:#53697d}"
    (OUT / "report.html").write_text(f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Exploratory causal stock analysis</title><style>{css}</style><body>{body}</body></html>", encoding="utf-8")


if __name__ == "__main__":
    main()
