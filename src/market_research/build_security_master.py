"""Build Security Master, Membership Intervals, Ticker Aliases, and Corporate Actions."""

import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def _atomic_write(df: pd.DataFrame, path: Path, parquet: bool) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        if parquet:
            df.to_parquet(tmp, index=False)
        else:
            df.to_csv(tmp, index=False)
        tmp.replace(path)
    finally:
        if tmp.exists():
            tmp.unlink()

def normalize_ticker(ticker: str) -> str:
    if not isinstance(ticker, str):
        return ""
    t = ticker.strip().upper()
    # Replace slash with dot for Alpaca compatibility (e.g. BRK/B -> BRK.B)
    t = t.replace("/", ".")
    # Fix common variations
    return t

def parse_date(d_str: str) -> str | None:
    if not isinstance(d_str, str) or not d_str.strip() or d_str.lower() == "nan":
        return None
    d_clean = re.sub(r"\[.*?\]", "", d_str).strip()
    try:
        dt = pd.to_datetime(d_clean)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def normalize_cik(value) -> str:
    """Normalize a CIK cell without treating missing-value text as an identity."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    return text

def build_security_master():
    sources_dir = Path("data/metadata/sources")
    meta_dir = Path("data/metadata")
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Datasets constituents
    ds_path = sources_dir / "datasets_sp500_constituents.csv"
    ds_df = pd.read_csv(ds_path) if ds_path.exists() else pd.DataFrame()

    # 2. Load Wikipedia Current
    wiki_curr_path = sources_dir / "wikipedia_current_constituents.csv"
    wiki_curr_df = pd.read_csv(wiki_curr_path) if wiki_curr_path.exists() else pd.DataFrame()

    # 3. Load Wikipedia Changes
    wiki_chg_path = sources_dir / "wikipedia_historical_changes.csv"
    wiki_chg_df = pd.read_csv(wiki_chg_path) if wiki_chg_path.exists() else pd.DataFrame()

    # 4. Load fja05680 ticker start/end
    fja_se_path = sources_dir / "fja05680_sp500_ticker_start_end.csv"
    fja_se_df = pd.read_csv(fja_se_path) if fja_se_path.exists() else pd.DataFrame()

    # 5. Load fja05680 components updated
    fja_comp_path = sources_dir / "fja05680_sp500_historical_components_updated.csv"
    fja_comp_df = pd.read_csv(fja_comp_path) if fja_comp_path.exists() else pd.DataFrame()

    # Build universe of all unique tickers
    ticker_info = {}

    # Populate from current constituents
    for _, row in ds_df.iterrows():
        sym = normalize_ticker(row.get("Symbol", ""))
        if not sym:
            continue
        ticker_info[sym] = {
            "company_name": row.get("Security", ""),
            "gics_sector": row.get("GICS Sector", ""),
            "gics_sub_industry": row.get("GICS Sub-Industry", ""),
            "cik": normalize_cik(row.get("CIK", "")),
            "founded": str(row.get("Founded", "")),
            "date_added": parse_date(str(row.get("Date added", ""))),
            "is_current": True,
            "status": "CURRENT",
        }

    for _, row in wiki_curr_df.iterrows():
        sym = normalize_ticker(row.get("Symbol", ""))
        if not sym:
            continue
        if sym not in ticker_info:
            ticker_info[sym] = {
                "company_name": row.get("Security", ""),
                "gics_sector": row.get("GICS Sector", ""),
                "gics_sub_industry": row.get("GICS Sub-Industry", ""),
                "cik": normalize_cik(row.get("CIK", "")),
                "founded": str(row.get("Founded", "")),
                "date_added": parse_date(str(row.get("Date added", ""))),
                "is_current": True,
                "status": "CURRENT",
            }

    # Process fja ticker start end
    membership_intervals = []
    interval_id_counter = 1

    for _, row in fja_se_df.iterrows():
        sym = normalize_ticker(row.get("ticker", ""))
        if not sym:
            continue
        st = parse_date(str(row.get("start_date", "")))
        en = parse_date(str(row.get("end_date", "")))
        
        if sym not in ticker_info:
            ticker_info[sym] = {
                "company_name": f"{sym} Corporation",
                "gics_sector": "Unknown",
                "gics_sub_industry": "Unknown",
                "cik": "",
                "founded": "",
                "date_added": st,
                "is_current": False,
                "status": "HISTORICAL_REMOVED" if en else "ACTIVE",
            }
        
        # Check if original 1957
        is_orig = False
        if ticker_info[sym].get("date_added") == "1957-03-04":
            is_orig = True
            if st and st > "1957-03-04":
                # Override start date with true launch date if verified original
                st = "1957-03-04"

        membership_intervals.append({
            "interval_id": f"INT_{interval_id_counter:05d}",
            "security_id": f"SEC_{sym.replace('.', '_')}",
            "ticker": sym,
            "start_date": st,
            "end_date": en,
            "is_ongoing": en is None,
            "confidence": "HIGH" if st and st >= "1996-01-01" else ("HIGH" if is_orig else "MEDIUM"),
            "source": "fja05680_clenow_wikipedia",
            "notes": "Original 1957 launch member" if is_orig else "Point-in-time constituent interval",
        })
        interval_id_counter += 1

    # Rebuild membership spells from dated constituent snapshots when
    # available. The date is the first/last observed snapshot; exact effective
    # entry/exit may fall between snapshots and is therefore marked uncertain.
    if {"date", "tickers"}.issubset(fja_comp_df.columns) and not fja_comp_df.empty:
        observed = []
        for _, row in fja_comp_df.iterrows():
            day = parse_date(str(row.get("date", "")))
            if not day:
                continue
            observed.append((day, {normalize_ticker(t) for t in str(row.get("tickers", "")).split(",") if normalize_ticker(t)}))
        observed.sort(key=lambda item: item[0])
        membership_intervals = []
        presence = set().union(*(tickers for _, tickers in observed))
        for ticker in sorted(presence):
            if ticker not in ticker_info:
                ticker_info[ticker] = {
                    "company_name": f"{ticker} Corporation", "gics_sector": "Unknown",
                    "gics_sub_industry": "Unknown", "cik": "", "founded": "",
                    "date_added": None, "is_current": False, "status": "HISTORICAL_REMOVED",
                }
        for ticker in sorted(presence):
            spells = []
            active_start = None
            for index, (day, members) in enumerate(observed):
                present = ticker in members
                if present and active_start is None:
                    active_start = day
                if not present and active_start is not None:
                    prior_day = observed[index - 1][0] if index else active_start
                    spells.append((active_start, prior_day, False))
                    active_start = None
            if active_start is not None:
                spells.append((active_start, observed[-1][0], True))
            for spell_no, (start, end, ongoing) in enumerate(spells, 1):
                sec_id = f"SEC_{ticker.replace('.', '_')}"
                membership_intervals.append({
                    "interval_id": f"INT_{interval_id_counter:05d}", "security_id": sec_id,
                    "ticker": ticker, "start_date": start, "end_date": None if ongoing else end,
                    "is_ongoing": ongoing, "confidence": "OBSERVED_SNAPSHOT",
                    "start_boundary_uncertain": True, "end_boundary_uncertain": True,
                    "spell_number": spell_no, "source": "fja05680_dated_constituent_snapshots",
                    "notes": "Observed membership spell; effective boundary lies between snapshots or is otherwise uncertain",
                })
                interval_id_counter += 1

    # Process corporate actions and changes from Wikipedia
    corporate_actions = []
    ticker_aliases = []
    ca_counter = 1
    alias_counter = 1

    for _, row in wiki_chg_df.iterrows():
        eff_date = parse_date(str(row.get("Effective Date", "")))
        add_sym = normalize_ticker(str(row.get("Added_Ticker", "")))
        add_name = str(row.get("Added_Security", ""))
        rem_sym = normalize_ticker(str(row.get("Removed_Ticker", "")))
        rem_name = str(row.get("Removed_Security", ""))
        reason = str(row.get("Reason", ""))

        if eff_date and (add_sym or rem_sym):
            ca_id = f"CA_{ca_counter:04d}"
            ca_counter += 1
            action_type = "INDEX_REBALANCE"
            r_lower = reason.lower()
            if "acquir" in r_lower or "merg" in r_lower or "bought" in r_lower:
                action_type = "MERGER_ACQUISITION"
            elif "spin" in r_lower:
                action_type = "SPINOFF"
            elif "reorganiz" in r_lower:
                action_type = "REORGANIZATION"
            elif "ticker" in r_lower or "symbol" in r_lower:
                action_type = "TICKER_CHANGE"

            sec_id = f"SEC_{rem_sym.replace('.', '_')}" if rem_sym else f"SEC_{add_sym.replace('.', '_')}"
            corporate_actions.append({
                "action_id": ca_id,
                "effective_date": eff_date,
                "security_id": sec_id,
                "action_type": action_type,
                "added_ticker": add_sym if add_sym != "NAN" else None,
                "added_security": add_name if add_name != "nan" else None,
                "removed_ticker": rem_sym if rem_sym != "NAN" else None,
                "removed_security": rem_name if rem_name != "nan" else None,
                "description": reason,
            })

            # Handle known aliases / renames
            if "symbol change" in r_lower or "ticker change" in r_lower or "renamed" in r_lower:
                ticker_aliases.append({
                    "alias_id": f"ALIAS_{alias_counter:04d}",
                    "security_id": f"SEC_{add_sym.replace('.', '_')}",
                    "alias_ticker": rem_sym,
                    "canonical_ticker": add_sym,
                    "effective_start": None,
                    "effective_end": eff_date,
                    "alias_type": "PREVIOUS_TICKER",
                    "reason": reason,
                })
                alias_counter += 1

    # Add well-known explicit aliases
    explicit_aliases = [
        ("FB", "META", "Renamed from Facebook to Meta Platforms on 2022-06-09", "2012-05-18", "2022-06-08"),
        ("BRK/B", "BRK.B", "Slash format alias for Berkshire Hathaway Class B", "1996-05-09", None),
        ("BF/B", "BF.B", "Slash format alias for Brown-Forman Class B", "1996-01-02", None),
        ("ANTM", "ELV", "Anthem renamed to Elevance Health 2022-06-28", "2002-12-02", "2022-06-27"),
        ("DISCA", "WBD", "Discovery Inc merged into Warner Bros. Discovery 2022-04-11", "2010-06-18", "2022-04-08"),
        ("TWTR", "TWTR", "Delisted upon private acquisition by X Holdings II Oct 2022", "2018-06-07", "2022-10-27"),
        ("ATVI", "ATVI", "Delisted upon acquisition by Microsoft Oct 2023", "2015-06-11", "2023-10-13"),
    ]
    for alt, can, rsn, s_dt, e_dt in explicit_aliases:
        if alt == "GOOG" and can == "GOOGL":
            continue
        ticker_aliases.append({
            "alias_id": f"ALIAS_{alias_counter:04d}",
            "security_id": f"SEC_{can.replace('.', '_')}",
            "alias_ticker": alt,
            "canonical_ticker": can,
            "effective_start": s_dt,
            "effective_end": e_dt,
            "alias_type": "EXPLICIT_MAPPING",
            "reason": rsn,
        })
        alias_counter += 1

    # Check for any remaining tickers in fja components not yet in ticker_info
    if {"date", "tickers"}.issubset(fja_comp_df.columns):
        for dt, t_str in zip(fja_comp_df["date"], fja_comp_df["tickers"]):
            for t in str(t_str).split(","):
                norm = normalize_ticker(t)
                if norm and norm not in ticker_info:
                    ticker_info[norm] = {
                        "company_name": f"{norm} Corporation",
                        "gics_sector": "Unknown",
                        "gics_sub_industry": "Unknown",
                        "cik": "",
                        "founded": "",
                        "date_added": dt,
                        "is_current": False,
                        "status": "HISTORICAL_REMOVED",
                    }

    # Construct security_master DataFrame
    master_records = []
    for sym, d in sorted(ticker_info.items()):
        sec_id = f"SEC_{sym.replace('.', '_')}"
        
        # Calculate min and max observed date in membership intervals
        sec_intervals = [i for i in membership_intervals if i["security_id"] == sec_id]
        min_date = min([i["start_date"] for i in sec_intervals if i["start_date"]]) if sec_intervals else d.get("date_added")
        max_date = None
        if not d.get("is_current"):
            end_dates = [i["end_date"] for i in sec_intervals if i["end_date"]]
            if end_dates:
                max_date = max(end_dates)

        # Detect share class
        share_class = "Common"
        if sym == "GOOG":
            share_class = "Class C"
        elif sym == "GOOGL":
            share_class = "Class A"
        elif ".A" in sym or "/A" in sym:
            share_class = "Class A"
        elif ".B" in sym or "/B" in sym:
            share_class = "Class B"
        elif ".C" in sym or "/C" in sym:
            share_class = "Class C"

        is_orig = d.get("date_added") == "1957-03-04" or (min_date and min_date <= "1957-03-04")
        if is_orig and not min_date:
            min_date = "1957-03-04"

        master_records.append({
            "security_id": sec_id,
            "issuer_id": (f"ISSUER_CIK_{d.get('cik')}" if d.get("cik") else None),
            "primary_ticker": sym,
            "company_name": d.get("company_name", f"{sym} Corporation"),
            "share_class": share_class,
            "cik": d.get("cik", ""),
            "gics_sector": d.get("gics_sector", "Unknown"),
            "gics_sub_industry": d.get("gics_sub_industry", "Unknown"),
            "founded_year": d.get("founded", ""),
            "is_current_member": d.get("is_current", False),
            "is_original_1957_member": is_orig,
            "first_observed_membership": min_date,
            "last_observed_membership": max_date,
            "status": "CURRENT" if d.get("is_current") else ("DELISTED_OR_ACQUIRED" if max_date else "HISTORICAL_REMOVED"),
            "identity_resolution": "ISSUER_CIK_ONLY_TICKER_REUSE_UNRESOLVED" if d.get("cik") else "AMBIGUOUS_TICKER_REUSE_POSSIBLE",
        })

    master_df = pd.DataFrame(master_records)
    intervals_df = pd.DataFrame(membership_intervals)
    aliases_df = pd.DataFrame(ticker_aliases)
    actions_df = pd.DataFrame(corporate_actions)

    # Save to Parquet and CSV
    _atomic_write(master_df, meta_dir / "security_master.parquet", parquet=True)
    _atomic_write(master_df, meta_dir / "security_master.csv", parquet=False)

    _atomic_write(intervals_df, meta_dir / "membership_intervals.parquet", parquet=True)
    _atomic_write(intervals_df, meta_dir / "membership_intervals.csv", parquet=False)

    _atomic_write(aliases_df, meta_dir / "ticker_aliases.parquet", parquet=True)
    _atomic_write(aliases_df, meta_dir / "ticker_aliases.csv", parquet=False)

    _atomic_write(actions_df, meta_dir / "corporate_actions.parquet", parquet=True)
    _atomic_write(actions_df, meta_dir / "corporate_actions.csv", parquet=False)

    print("=== Security Master Generation Summary ===")
    print(f"Total Securities: {len(master_df)}")
    print(f"Current Members: {master_df['is_current_member'].sum()}")
    print(f"Historical (Non-current) Members: {(~master_df['is_current_member']).sum()}")
    print(f"Original 1957 Members Identified: {master_df['is_original_1957_member'].sum()}")
    print(f"Membership Intervals Logged: {len(intervals_df)}")
    print(f"Ticker Aliases Logged: {len(aliases_df)}")
    print(f"Corporate Actions Logged: {len(actions_df)}")
    print(f"Files saved successfully in {meta_dir}")

if __name__ == "__main__":
    build_security_master()
