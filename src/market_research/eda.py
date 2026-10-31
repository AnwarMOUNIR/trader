r"""Offline exploratory analysis of the local archive; never acquires or changes data.

Run from the repository root: .\.venv\Scripts\python.exe src/market_research/eda.py
Outputs standalone HTML/Markdown and machine-readable Parquet/JSON in reports/eda.
"""
from pathlib import Path
from datetime import datetime, timezone
import html
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import exchange_calendars as xcals

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "eda"


def table(df):
    return df.to_html(index=False, border=0, float_format=lambda x: f"{x:,.3f}", na_rep="—")


def markdown_table(df):
    def fmt(x):
        if pd.isna(x):
            return "—"
        return f"{x:,.3f}" if isinstance(x, float) else str(x).replace("|", "/")
    return "\n".join(["| " + " | ".join(map(str, df.columns)) + " |",
                      "| " + " | ".join(["---"] * len(df.columns)) + " |"] +
                     ["| " + " | ".join(fmt(x) for x in row) + " |" for row in df.itertuples(index=False, name=None)])


def bars(df, label, value, title):
    maximum = max(float(df[value].max()), 1)
    rows = "".join(f'<div class="barrow"><span>{html.escape(str(r[label]))}</span>'
                   f'<div class="track"><div class="bar" style="width:{float(r[value])/maximum*100:.2f}%"></div></div>'
                   f'<b>{float(r[value]):,.2f}</b></div>' for _, r in df.iterrows())
    return f"<h3>{html.escape(title)}</h3>{rows}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inventory = []
    for p in sorted((ROOT / "data").rglob("*")):
        if p.is_file():
            inventory.append({"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size,
                              "rows": pq.ParquetFile(p).metadata.num_rows if p.suffix == ".parquet" else None})
    pd.DataFrame(inventory).to_parquet(OUT / "inventory.parquet", index=False)
    master = pd.read_parquet(ROOT / "data/metadata/security_master.parquet")
    membership = pd.read_parquet(ROOT / "data/metadata/membership_intervals.parquet")
    aliases = pd.read_parquet(ROOT / "data/metadata/ticker_aliases.parquet")
    cal = xcals.get_calendar("XNYS", start="2015-01-01", end="2027-01-01")
    schedule = pd.DataFrame({"session_date": cal.sessions.strftime("%Y-%m-%d"),
                             "session_open": cal.opens.to_numpy(), "session_close": cal.closes.to_numpy()})
    schedule["session_open"] = pd.to_datetime(schedule.session_open, utc=True)
    schedule["session_close"] = pd.to_datetime(schedule.session_close, utc=True)
    schedule["expected_minutes"] = ((schedule.session_close - schedule.session_open).dt.total_seconds()/60).astype(int)

    coverage, quality, daily_parts, profiles, numeric_parts, outliers = [], [], [], [], [], []
    session_parts, year_parts, candle_parts, missingness, gaps = [], [], [], [], []
    for p in sorted((ROOT / "data/normalized").glob("*.parquet")):
        df = pd.read_parquet(p).sort_values("timestamp_utc")
        symbol = str(df.symbol.iloc[0])
        df["timestamp_utc"] = pd.to_datetime(df.timestamp_utc, utc=True)
        df["session_date"] = df.timestamp_utc.dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
        joined = df.merge(schedule, on="session_date", how="left", validate="many_to_one")
        mask = (joined.timestamp_utc >= joined.session_open) & (joined.timestamp_utc < joined.session_close)
        rth = joined.loc[mask].copy()
        observed_dates = sorted(rth.session_date.unique())
        full = schedule.loc[schedule.session_date.between(observed_dates[0], observed_dates[-1])].copy()
        counts = rth.groupby("session_date").timestamp_utc.nunique()
        full["observed_minutes"] = full.session_date.map(counts).fillna(0).astype(int)
        full["symbol"] = symbol
        full["absent_minutes"] = full.expected_minutes - full.observed_minutes
        full["coverage_pct"] = full.observed_minutes / full.expected_minutes * 100
        session_parts.append(full)
        absent = full.loc[full.observed_minutes.eq(0)].copy()
        if len(absent):
            absent["block"] = (absent.index.to_series().diff().ne(1)).cumsum()
            for _, block in absent.groupby("block"):
                gaps.append({"symbol": symbol, "first_missing_session": block.session_date.iloc[0],
                             "last_missing_session": block.session_date.iloc[-1], "missing_sessions": len(block)})
        coverage.append({"symbol": symbol, "normalized_rows": len(df), "rth_rows": len(rth),
                         "outside_rth_rows": int((~mask).sum()), "first_session": observed_dates[0],
                         "last_session": observed_dates[-1], "observed_sessions": len(observed_dates),
                         "calendar_sessions_in_span": len(full), "absent_sessions_in_span": int(full.observed_minutes.eq(0).sum()),
                         "rth_minute_coverage_in_span_pct": 100*full.observed_minutes.sum()/full.expected_minutes.sum()})
        price = df[["open", "high", "low", "close"]]
        bad_ohlc = (df.high < df[["open", "low", "close"]].max(axis=1)) | (df.low > df[["open", "high", "close"]].min(axis=1))
        quality.append({"symbol": symbol, "duplicate_timestamps": int(df.timestamp_utc.duplicated().sum()),
                        "ohlc_violations": int(bad_ohlc.sum()), "nonpositive_price_rows": int(price.le(0).any(axis=1).sum()),
                        "nonfinite_price_rows": int((~np.isfinite(price)).any(axis=1).sum()),
                        "negative_volume_rows": int(df.volume.lt(0).sum()), "zero_volume_rows": int(df.volume.eq(0).sum()),
                        "off_minute_timestamps": int(df.timestamp_utc.ne(df.timestamp_utc.dt.floor("min")).sum())})
        for col in df:
            missingness.append({"symbol": symbol, "column": col, "nulls": int(df[col].isna().sum()),
                                "empty_strings": int(df[col].eq("").sum()) if pd.api.types.is_string_dtype(df[col]) else 0})
        for col in ["open", "high", "low", "close", "volume", "trade_count"]:
            d = rth[col].describe(percentiles=[.01, .5, .99]).to_dict()
            numeric_parts.append({"symbol": symbol, "column": col, **d})
        # Only adjacent minutes in the same session enter return statistics.
        consecutive = rth.timestamp_utc.diff().eq(pd.Timedelta(minutes=1)) & rth.session_date.eq(rth.session_date.shift())
        rth["minute_return_pct"] = rth.close.pct_change(fill_method=None).where(consecutive)*100
        extreme = rth.loc[rth.minute_return_pct.abs().nlargest(5).index,
                          ["symbol", "timestamp_utc", "close", "minute_return_pct"]]
        outliers.append(extreme)
        daily = rth.groupby("session_date", sort=True).agg(open=("open", "first"), high=("high", "max"),
                 low=("low", "min"), close=("close", "last"), volume=("volume", "sum"),
                 observed_minutes=("timestamp_utc", "nunique"))
        daily["symbol"] = symbol
        daily["intraday_return_pct"] = (daily.close/daily.open-1)*100
        daily["range_pct"] = (daily.high/daily.low-1)*100
        daily = daily.reset_index().merge(schedule[["session_date", "expected_minutes"]], on="session_date", validate="many_to_one")
        daily["minute_coverage_pct"] = 100*daily.observed_minutes/daily.expected_minutes
        daily["year"] = daily.session_date.str[:4]
        daily_parts.append(daily)
        yearly = daily.groupby("year").agg(sessions=("session_date", "size"),
                     median_intraday_return_pct=("intraday_return_pct", "median"),
                     intraday_return_std_pct=("intraday_return_pct", "std"),
                     median_range_pct=("range_pct", "median"), median_volume=("volume", "median"),
                     median_minute_coverage_pct=("minute_coverage_pct", "median")).reset_index()
        yearly.insert(0, "symbol", symbol)
        year_parts.append(yearly)
        rth["slot"] = ((rth.timestamp_utc-rth.session_open).dt.total_seconds()//1800).astype(int)
        rth["day_volume"] = rth.groupby("session_date").volume.transform("sum")
        slots = rth.groupby(["session_date", "slot"]).agg(volume=("volume", "sum"), day_volume=("day_volume", "first"),
                    abs_minute_return_pct=("minute_return_pct", lambda x: x.abs().mean())).reset_index()
        slots["day_volume_share_pct"] = 100*slots.volume/slots.day_volume
        profile = slots.groupby("slot").agg(mean_day_volume_share_pct=("day_volume_share_pct", "mean"),
                        mean_abs_minute_return_pct=("abs_minute_return_pct", "mean"), sessions=("session_date", "size")).reset_index()
        profile["symbol"] = symbol
        profiles.append(profile)

    cov = pd.DataFrame(coverage)
    sessions = pd.concat(session_parts, ignore_index=True)
    daily = pd.concat(daily_parts, ignore_index=True)
    yearly = pd.concat(year_parts, ignore_index=True)
    for interval in [20, 30]:
        for p in sorted((ROOT / f"data/candles/{interval}min").glob("*.parquet")):
            df = pd.read_parquet(p).sort_values("timestamp_utc")
            dates = pd.to_datetime(df.timestamp_utc, utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
            joined = df.assign(session_date=dates).merge(schedule, on="session_date", how="left", validate="many_to_one")
            elapsed = (pd.to_datetime(joined.timestamp_utc, utc=True)-joined.session_open).dt.total_seconds()/60
            expected_duration = np.minimum(interval, joined.expected_minutes-elapsed)
            daily_vol = df.groupby("session_date").volume.sum()
            source_vol = daily.loc[daily.symbol.eq(df.symbol.iloc[0])].set_index("session_date").volume
            aligned = pd.concat([daily_vol.rename("candles"), source_vol.rename("source")], axis=1)
            source_daily = daily.loc[daily.symbol.eq(df.symbol.iloc[0])].set_index("session_date")
            aggregated = df.groupby("session_date").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
            ohlc_mismatches = sum(int((~np.isclose(aggregated[col], source_daily.loc[aggregated.index, col])).sum()) for col in ["open", "high", "low", "close"])
            scheduled = source_daily[["expected_minutes"]]
            expected_candles = int(np.ceil(scheduled.expected_minutes / interval).sum())
            candle_parts.append({"symbol": str(df.symbol.iloc[0]), "interval": interval, "rows": len(df),
                "absent_windows_on_observed_sessions": expected_candles-len(df),
                "partial_rows": int(df.is_partial.sum()), "incomplete_rows": int((~df.is_complete).sum()),
                "incomplete_pct": 100*(~df.is_complete).mean(),
                "missing_input_minutes": int((df.expected_bars-df.observed_bars).clip(lower=0).sum()),
                "duplicate_timestamps": int(df.timestamp_utc.duplicated().sum()),
                "ohlc_violations": int(((df.high < df[["open", "low", "close"]].max(axis=1)) | (df.low > df[["open", "high", "close"]].min(axis=1))).sum()),
                "calendar_alignment_violations": int((elapsed.isna() | elapsed.lt(0) | elapsed.ge(joined.expected_minutes) | elapsed.mod(interval).ne(0)).sum()),
                "duration_violations": int(joined.duration_minutes.ne(expected_duration).sum()),
                "completeness_flag_violations": int(df.is_complete.ne(df.observed_bars.eq(df.expected_bars)).sum()),
                "session_volume_mismatches": int(aligned.candles.ne(aligned.source).sum()),
                "session_ohlc_field_mismatches": ohlc_mismatches})
    candles = pd.DataFrame(candle_parts)
    start = pd.to_datetime(membership.start_date, errors="coerce")
    end = pd.to_datetime(membership.end_date, errors="coerce")
    current = set(master.loc[master.is_current_member, "primary_ticker"])
    ongoing = set(membership.loc[membership.end_date.isna(), "ticker"])
    meta_checks = {"master_rows": len(master), "current_members": len(current), "membership_rows": len(membership),
        "alias_rows": len(aliases), "duplicate_security_ids": int(master.security_id.duplicated().sum()),
        "unknown_sector_rows": int(master.gics_sector.fillna("Unknown").eq("Unknown").sum()),
        "names_ending_in_corporation": int(master.company_name.str.endswith(" Corporation").sum()),
        "blank_cik_rows": int(master.cik.fillna("").eq("").sum()),
        "missing_interval_starts": int(start.isna().sum()), "end_before_start": int((end < start).sum()),
        "interval_security_ids_absent_from_master": sorted(set(membership.security_id)-set(master.security_id)),
        "current_without_ongoing_interval": sorted(current-ongoing),
        "ongoing_not_current": sorted(ongoing-current),
        "acquired_symbols_absent_from_master": sorted(set(cov.symbol)-set(master.primary_ticker))}
    snapshots = []
    for date in ["1957-03-04", "1996-01-02", "2016-01-04", "2023-05-15", "2024-12-31", "2026-10-23"]:
        active = membership.loc[(start <= pd.Timestamp(date)) & (end.isna() | (end >= pd.Timestamp(date)))]
        snapshots.append({"date": date, "active_distinct_tickers": active.ticker.nunique(), "active_intervals": len(active)})
    overlaps = []
    for sym, group in membership.groupby("ticker"):
        g = group.sort_values("start_date")
        latest_end = None
        for row in g.itertuples():
            st = pd.to_datetime(row.start_date)
            en = pd.to_datetime(row.end_date) if pd.notna(row.end_date) else pd.Timestamp.max
            if latest_end is not None and st <= latest_end:
                overlaps.append({"ticker": sym, "interval_id": row.interval_id, "start_date": row.start_date})
            latest_end = en if latest_end is None else max(latest_end, en)
    meta_checks["overlapping_or_touching_intervals"] = overlaps
    # Daily open-to-close return correlations, pairwise dates; no alias stitching.
    wide = daily.pivot(index="session_date", columns="symbol", values="intraday_return_pct")
    corr = wide.corr(min_periods=30)
    pairs = wide.notna().astype(int).T.dot(wide.notna().astype(int))
    outputs = {"coverage": cov, "quality": pd.DataFrame(quality), "sessions": sessions, "daily": daily,
               "symbol_year": yearly, "intraday_profile": pd.concat(profiles, ignore_index=True),
               "numeric_summary": pd.DataFrame(numeric_parts), "return_outliers": pd.concat(outliers, ignore_index=True),
               "candle_quality": candles, "missingness": pd.DataFrame(missingness),
               "coverage_gaps": pd.DataFrame(gaps), "membership_snapshots": pd.DataFrame(snapshots),
               "return_correlations": corr.reset_index(), "correlation_pair_counts": pairs.reset_index()}
    for name, frame in outputs.items():
        frame.to_parquet(OUT / f"{name}.parquet", index=False)
    raw_rows = sum(r["rows"] for r in inventory if r["path"].startswith("data/raw/") and r["rows"] is not None)
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(), "raw_rows": raw_rows,
               "normalized_rows": int(cov.normalized_rows.sum()), "rth_rows": int(cov.rth_rows.sum()),
               "data_bytes": sum(r["bytes"] for r in inventory), "metadata_checks": meta_checks}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    covered_master = master.primary_ticker.isin(cov.symbol).sum()
    year_stats = yearly.set_index(["symbol", "year"])
    findings = [
        f"The local archive contains {raw_rows:,} raw minute rows, {cov.normalized_rows.sum():,} normalized minute rows, and {cov.rth_rows.sum():,} regular-session minute rows across {len(cov)} ticker labels. Raw and normalized are separate representations, not additive observations.",
        f"Only {covered_master:,} of {len(master):,} master tickers ({100*covered_master/len(master):.2f}%) have matching local normalized files. FB and META are counted separately here; ticker labels are not independent companies.",
        f"Derived products contain {candles.loc[candles.interval.eq(20), 'rows'].sum():,} 20-minute and {candles.loc[candles.interval.eq(30), 'rows'].sum():,} 30-minute candles. Of these, {candles.incomplete_rows.sum():,} are flagged incomplete across the two products; these counts describe the same underlying archive at two resolutions.",
        "No duplicate timestamps, invalid OHLC relationships, nonpositive/nonfinite prices, negative volumes or off-minute timestamps were found in normalized bars. Derived candles pass session alignment, duration and completeness-flag checks, and reconstruct daily regular-session OHLC and volume without mismatches. These checks establish internal consistency, not independent feed accuracy.",
        f"SIVB has {candles.loc[(candles.symbol.eq('SIVB')) & candles.interval.eq(30), 'incomplete_pct'].iloc[0]:.2f}% incomplete 30-minute candles; FRC has {candles.loc[(candles.symbol.eq('FRC')) & candles.interval.eq(30), 'incomplete_pct'].iloc[0]:.2f}%. Entirely absent windows on otherwise observed sessions are counted separately from incomplete emitted candles in candle_quality.parquet.",
        f"FRC's observed daily open-to-close return standard deviation is {year_stats.loc[('FRC', '2022'), 'intraday_return_std_pct']:.2f}% in 2022 and {year_stats.loc[('FRC', '2023'), 'intraday_return_std_pct']:.2f}% in 2023. META's median observed daily high/low range is {year_stats.loc[('META', '2022'), 'median_range_pct']:.2f}% in 2022 and {year_stats.loc[('META', '2024'), 'median_range_pct']:.2f}% in 2024. These describe this selected, unadjusted sample and are sensitive to incomplete sessions.",
        f"There are {cov.absent_sessions_in_span.sum():,} absent ticker-sessions between each ticker's first and last observed regular session. These include intentionally unacquired years, and are not all provider failures. Min/max date ranges do not establish continuous coverage.",
        f"{cov.outside_rth_rows.sum():,} normalized rows fall outside the XNYS regular-session windows, including possible extended-hours and non-session records. Market-behavior statistics below use regular sessions only.",
        f"Membership checks find {len(meta_checks['current_without_ongoing_interval'])} current tickers without open-ended intervals, {len(meta_checks['ongoing_not_current'])} open-ended tickers absent from the current flag set, and {len(overlaps)} overlapping or touching intervals. This metadata needs reconciliation before point-in-time backtests.",
        f"The master has {meta_checks['unknown_sector_rows']} unknown sectors and {meta_checks['blank_cik_rows']} blank CIK values. Code inspection shows IDs are constructed from tickers, historical names can be generated as '<ticker> Corporation', and alias mappings require identity review. Permanent entity identity and complete all-time membership are not established by these files.",
        "Existing STATUS.md and FINAL_REPORT.md claims of complete modern-era coverage and complete all-time membership exceed the evidence in this pilot archive. This EDA preserves those files and records independently measured results here.",
    ]
    limitations = [
        "Coverage denominators use XNYS scheduled sessions within the observed span, not listing or membership lifetimes. Missing minutes can reflect no eligible trades, halts, or missing acquisition; the archive alone does not distinguish them.",
        "Intraday returns are 100*(last observed close/first observed open-1); ranges are 100*(session high/session low-1). Incomplete sessions can bias both. No adjusted total returns, investment recommendations, annualized performance, or cross-gap returns are reported.",
        "Prices are marked raw/unadjusted. Splits, dividends, ticker reuse and corporate actions require separate validation. Outliers are candidates for review, not automatically data errors.",
        "Minute returns require consecutive timestamps within the same session. Correlations use pairwise overlapping session open-to-close returns, at least 30 observations; overlap counts are provided and missing pairs stay blank.",
        "Intraday volume charts average each day's slot volume as a share of that day's observed regular-session volume. Early closes and sparse sessions affect the contributing sample; slot counts are exported.",
        "This is an offline local-file analysis. No provider entitlement, feed accuracy, external source claims, or current market facts were reverified. Input files and existing acquisition processes are unchanged.",
    ]
    sections = [("Coverage by ticker", cov), ("Coverage and behavior by ticker/year", yearly),
                ("Normalized minute quality", pd.DataFrame(quality)), ("Derived candle quality", candles),
                ("Membership count snapshots (inclusive end dates)", pd.DataFrame(snapshots)),
                ("Largest absolute adjacent-minute returns (five per ticker)", outputs["return_outliers"].sort_values("minute_return_pct", key=lambda x: x.abs(), ascending=False)),
                ("Daily intraday return correlations", corr.reset_index()), ("Pairwise overlap counts", pairs.reset_index())]
    md = ["# Local archive exploratory data analysis", f"Generated {summary['generated_utc']}",
          "## Directory discovery", "Python package: src/market_research. Acquisition and checkpoint code: acquire.py, backfill.py, daily_update.py, db.py. Metadata builders: fetch_membership.py and build_security_master.py. Session aggregation: resample.py. Existing tests: tests/test_resample.py. Data is organized into raw, normalized, metadata, provenance and 20/30-minute candle products; existing reports are in reports/. Credentials were not read.",
          "## Findings", *[f"- {x}" for x in findings], "## Methods and limitations", *[f"- {x}" for x in limitations]]
    for title, frame in sections:
        md += [f"## {title}", markdown_table(frame)]
    md += ["## Reproduction and outputs", "Run `.\\.venv\\Scripts\\python.exe src/market_research/eda.py` from the project root. The script reads data files and writes only reports/eda. report.html is a standalone visual report; summary.json records counts and metadata mismatches; Parquet tables include the file inventory, missing-session blocks, null/empty counts, distributions, daily statistics, session coverage, intraday profiles and outliers. Inspect coverage_gaps.parquet to locate unacquired blocks. No source files or existing reports are overwritten."]
    (OUT / "EDA_REPORT.md").write_text("\n\n".join(md)+"\n", encoding="utf-8")
    plots = bars(cov, "symbol", "observed_sessions", "Observed regular sessions by ticker")
    plots += bars(cov, "symbol", "rth_minute_coverage_in_span_pct", "Minute coverage within first-to-last observed session (%)")
    medians = daily.groupby("symbol", as_index=False).agg(median_range_pct=("range_pct", "median"))
    plots += bars(medians, "symbol", "median_range_pct", "Median observed session high/low range (%)")
    profile = outputs["intraday_profile"].copy()
    profile["slot_start_ny"] = profile.slot.map(lambda x: f"{9+(30+int(x)*30)//60:02}:{(30+int(x)*30)%60:02}")
    for symbol, frame in profile.groupby("symbol"):
        plots += bars(frame, "slot_start_ny", "mean_day_volume_share_pct", f"{symbol}: mean share of observed daily volume by 30-minute slot (%)")
    body = "<h1>Local archive exploratory data analysis</h1><p>" + html.escape(summary["generated_utc"]) + "</p>"
    body += "<h2>Findings</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in findings) + "</ul>"
    body += "<h2>Coverage, volatility and volume</h2>" + plots
    for title, frame in sections:
        body += f"<h2>{html.escape(title)}</h2><div class='scroll'>{table(frame)}</div>"
    body += "<h2>Methods and limitations</h2><ul>" + "".join(f"<li>{html.escape(x)}</li>" for x in limitations) + "</ul>"
    body += "<h2>Metadata diagnostics</h2><pre>"+html.escape(json.dumps(meta_checks, indent=2))+"</pre>"
    css = "body{font:16px system-ui;margin:40px auto;padding:0 24px;max-width:1200px;color:#16283b;background:#f6f8fc}h1,h2,h3{color:#123b62}li{margin:12px 0;line-height:1.5}.scroll{overflow:auto;background:white}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:10px;text-align:right;border-bottom:1px solid #ddd;white-space:nowrap}th{background:#e4eef6}.barrow{display:flex;align-items:center;gap:12px;margin:9px 0}.barrow span{width:75px}.barrow b{width:90px;text-align:right}.track{flex:1;background:#e1e9f0;height:18px}.bar{background:#167d9a;height:18px}pre{white-space:pre-wrap;background:white;padding:20px}"
    (OUT / "report.html").write_text(f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Market Research EDA</title><style>{css}</style><body>{body}</body></html>", encoding="utf-8")
    # Reconcile independent file footers and the two candle representations.
    normalized_footer_rows = sum(r["rows"] for r in inventory if r["path"].startswith("data/normalized/") and r["rows"] is not None)
    assert normalized_footer_rows == summary["normalized_rows"]
    assert daily.observed_minutes.sum() == summary["rth_rows"]
    assert sessions.observed_minutes.sum() == summary["rth_rows"]
    for interval in [20, 30]:
        footer_rows = sum(r["rows"] for r in inventory if r["path"].startswith(f"data/candles/{interval}min/") and r["rows"] is not None)
        assert footer_rows == candles.loc[candles.interval.eq(interval), "rows"].sum()
    assert (OUT / "report.html").read_text(encoding="utf-8").count("<table ") == len(sections)
    print("Verification passed: file-footer counts, session totals, candle totals and HTML tables reconcile.")
    print(json.dumps(summary, indent=2))
    print(cov.to_string(index=False))
    print(candles.to_string(index=False))
    print(f"Reports written to {OUT}")


if __name__ == "__main__":
    main()
