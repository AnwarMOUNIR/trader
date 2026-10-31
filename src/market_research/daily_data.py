"""Session-aligned daily bars and conservative action-window eligibility.

This module does not adjust raw OHLC for corporate actions.  It quarantines
known affected observations instead; absence of an action list is explicitly
reported as missing action coverage.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import exchange_calendars as xcals


SUPPORTED_ACTIONS = {"split", "dividend", "spinoff", "delisting"}
FEATURE_LOOKBACK_SESSIONS = 60
TARGET_HORIZON_SESSIONS = 5


def _date_strings(values: Iterable) -> pd.Index:
    dates = pd.to_datetime(list(values), errors="raise")
    if isinstance(dates, pd.DatetimeIndex) and dates.tz is not None:
        dates = dates.tz_convert(None)
    return pd.Index(dates.strftime("%Y-%m-%d"))


def _actions_frame(actions) -> pd.DataFrame:
    if actions is None:
        return pd.DataFrame(columns=["date", "type"])
    if isinstance(actions, pd.DataFrame):
        frame = actions.copy()
    else:
        frame = pd.DataFrame(list(actions))
    if frame.empty:
        return pd.DataFrame(columns=["date", "type"])
    if "date" not in frame or "type" not in frame:
        raise ValueError("actions must supply explicit date and type fields")
    frame = frame[["date", "type"]].copy()
    frame["date"] = _date_strings(frame.date)
    frame["type"] = frame.type.astype(str).str.lower()
    unknown = sorted(set(frame.type) - SUPPORTED_ACTIONS)
    if unknown:
        raise ValueError(f"unsupported corporate action types: {unknown}")
    return frame.drop_duplicates().sort_values(["date", "type"]).reset_index(drop=True)


def prepare_daily(daily: pd.DataFrame, symbol: str, actions=None) -> pd.DataFrame:
    """Reindex raw daily bars to XNYS sessions and create eligibility/labels.

    Inputs need a ``date`` column or a date-like index and raw ``open``,
    ``high``, ``low``, ``close`` columns. Existing columns and unadjusted OHLC
    values are preserved. A row is feature eligible only when its 60-return
    trailing history (61 OHLC observations) is complete and it is outside every known action's
    affected lookback. Labels use exactly the next session open and fifth
    session close; missing intermediate bars or a known action in that target
    window makes the label ineligible. No price-jump action inference or total-
    return accounting is performed.
    """
    if not isinstance(daily, pd.DataFrame):
        raise TypeError("daily must be a pandas DataFrame")
    if daily.empty:
        raise ValueError(f"{symbol}: daily bars are empty")
    frame = daily.copy()
    if "date" in frame.columns:
        raw_dates = _date_strings(frame.pop("date"))
    else:
        raw_dates = _date_strings(frame.index)
    frame = frame.reset_index(drop=True)
    frame.insert(0, "date", raw_dates.to_numpy())
    if frame.date.duplicated().any():
        raise ValueError(f"{symbol}: duplicate daily dates")
    required = {"open", "high", "low", "close"}
    missing_cols = required - set(frame.columns)
    if missing_cols:
        raise ValueError(f"{symbol}: missing required OHLC columns {sorted(missing_cols)}")
    frame = frame.sort_values("date").reset_index(drop=True)
    start, end = frame.date.iloc[0], frame.date.iloc[-1]
    calendar = xcals.get_calendar("XNYS", start="1950-01-01", end="2035-12-31")
    sessions = calendar.sessions_in_range(start, end)
    sessions = pd.Index(s.strftime("%Y-%m-%d") for s in sessions)
    if not set(frame.date).issubset(set(sessions)):
        bad = sorted(set(frame.date) - set(sessions))
        raise ValueError(f"{symbol}: daily bars include non-XNYS session dates {bad[:5]}")
    # Return only sessions in the observed date range. Near the end of the
    # input, labels without an observed fifth-session close stay unavailable.
    session_dates = sessions
    aligned = frame.set_index("date").reindex(session_dates)
    aligned.index.name = "date"
    aligned = aligned.reset_index()
    aligned["session_missing"] = aligned[list(required)].isna().any(axis=1)

    actions_frame = _actions_frame(actions)
    all_action_dates = set(actions_frame.date)
    non_session_actions = sorted(date for date in all_action_dates
                                 if not calendar.is_session(pd.Timestamp(date)))
    if non_session_actions:
        raise ValueError(f"{symbol}: action dates must be XNYS sessions: {non_session_actions}")
    # Event feeds may include history or future events outside the requested
    # bars. Validate their dates above, then apply only events in this frame.
    action_dates = all_action_dates & set(session_dates)

    numeric_valid = np.ones(len(aligned), dtype=bool)
    numeric: dict[str, np.ndarray] = {}
    for col in ("open", "high", "low", "close"):
        vals = pd.to_numeric(aligned[col], errors="coerce").to_numpy(float)
        numeric[col] = vals
        numeric_valid &= np.isfinite(vals) & (vals > 0)
    numeric_valid &= (numeric["high"] >= numeric["low"])
    numeric_valid &= (numeric["high"] >= np.maximum(numeric["open"], numeric["close"]))
    numeric_valid &= (numeric["low"] <= np.minimum(numeric["open"], numeric["close"]))
    if "volume" in aligned.columns:
        vol = pd.to_numeric(aligned.volume, errors="coerce").to_numpy(float)
        numeric_valid &= np.isfinite(vol) & (vol >= 0)
    if "session_complete" in aligned.columns:
        session_complete = aligned.session_complete.fillna(False).astype(bool).to_numpy()
        numeric_valid &= session_complete
    aligned["bar_complete"] = numeric_valid
    # A 60-session return needs both endpoints: t-60 through t inclusive.
    # Checking only 60 bars can admit an invalid baseline for mom60/beta60.
    history_bars = FEATURE_LOOKBACK_SESSIONS + 1
    complete_history = pd.Series(numeric_valid).rolling(
        history_bars, min_periods=history_bars).sum().eq(history_bars)
    feature_ok = complete_history.to_numpy(dtype=bool, copy=True)

    date_pos = {date: i for i, date in enumerate(aligned.date)}
    action_types = {date: ",".join(group.type) for date, group in actions_frame.groupby("date", sort=True)}
    # A delisting before the supplied price span must not silently resurrect
    # that security under a reused ticker.
    prior_delisting = bool(((actions_frame.type == "delisting") & (actions_frame.date < start)).any())
    delisting_positions = [0] if prior_delisting else []
    if prior_delisting:
        feature_ok[:] = False
    for action_date in action_dates:
        pos = date_pos[action_date]
        if "delisting" in action_types[action_date].split(","):
            feature_ok[pos:] = False
            delisting_positions.append(pos)
        else:
            feature_ok[pos:min(pos + FEATURE_LOOKBACK_SESSIONS, len(feature_ok))] = False
    aligned["feature_eligible"] = feature_ok

    entry_dates, exit_dates, entry_prices, exit_prices, target_returns, label_ok = [], [], [], [], [], []
    for i in range(len(aligned)):
        entry_i, exit_i = i + 1, i + TARGET_HORIZON_SESSIONS
        in_range = exit_i < len(aligned)
        entry_date = aligned.date.iloc[entry_i] if entry_i < len(aligned) else None
        exit_date = aligned.date.iloc[exit_i] if exit_i < len(aligned) else None
        entry_px = aligned.open.iloc[entry_i] if in_range else np.nan
        exit_px = aligned.close.iloc[exit_i] if in_range else np.nan
        target_valid = bool(in_range and numeric_valid[entry_i:exit_i + 1].all()
                            and np.isfinite(float(entry_px)) and float(entry_px) > 0
                            and np.isfinite(float(exit_px)) and float(exit_px) > 0)
        if target_valid:
            # Any explicitly supplied event in sessions t+1 through t+5
            # invalidates the raw-price label; this is a quarantine, not an
            # adjusted total-return target.
            target_valid = not any(aligned.date.iloc[j] in action_dates
                                   for j in range(entry_i, exit_i + 1))
            if target_valid:
                target_valid = not any(position <= exit_i for position in delisting_positions)
        entry_dates.append(entry_date)
        exit_dates.append(exit_date)
        entry_prices.append(float(entry_px) if in_range and pd.notna(entry_px) else np.nan)
        exit_prices.append(float(exit_px) if in_range and pd.notna(exit_px) else np.nan)
        target_returns.append(float(exit_px) / float(entry_px) - 1
                              if target_valid else np.nan)
        label_ok.append(target_valid)
    aligned["entry_date"] = entry_dates
    aligned["exit_date"] = exit_dates
    aligned["entry_open"] = entry_prices
    aligned["exit_close"] = exit_prices
    aligned["target_return"] = target_returns
    aligned["label_eligible"] = label_ok
    aligned["corporate_action"] = aligned.date.map(action_types).astype("string").fillna("").astype(str)
    coverage = ("not_supplied" if actions is None else
                "known_events_only_coverage_not_certified")
    aligned["action_coverage"] = coverage
    aligned["missing_action_coverage"] = True
    aligned.attrs["symbol"] = symbol
    aligned.attrs["action_coverage"] = coverage
    aligned.attrs["action_events_supplied"] = actions is not None
    aligned.attrs["accounting_note"] = (
        "Raw OHLC retained. Known corporate-action windows are quarantined; "
        "this helper does not provide split-adjusted shares or total-return accounting."
    )
    return aligned
