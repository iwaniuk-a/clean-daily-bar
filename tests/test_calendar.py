import pandas as pd
import pytest
from cleanbars.calendars import expected_sessions, build_calendar_audit
from cleanbars.normalize import CANONICAL_DTYPES, INDEX_NAMES
from cleanbars.calendars import build_session_completeness_audit
from cleanbars.calendars import filter_complete_sessions


def test_expected_sessions_excludes_weekend():
    # 2026-07-10 is a Friday, 2026-07-13 is a Monday
    sessions = expected_sessions("2026-07-10", "2026-07-13")

    assert len(sessions) == 2
    assert pd.Timestamp("2026-07-10") in sessions
    assert pd.Timestamp("2026-07-13") in sessions
    assert pd.Timestamp("2026-07-11") not in sessions  # Saturday excluded
    assert pd.Timestamp("2026-07-12") not in sessions  # Sunday excluded

def test_expected_sessions_rejects_inverted_range():
    with pytest.raises(ValueError, match="later than"):
        expected_sessions("2026-07-10", "2026-07-01")

def test_build_calendar_audit_scenarios():
    # Synthetic calendar covering Fri, Mon, Tue
    sessions = pd.DatetimeIndex([
        pd.Timestamp("2026-07-10"),
        pd.Timestamp("2026-07-13"),
        pd.Timestamp("2026-07-14")
    ])

    # Asset A: Exists Fri, Sat (unexpected), skips Mon (missing), exists Tue
    # Asset B: First observation is Mon (IPO scenario)
    idx = pd.MultiIndex.from_tuples([
        (pd.Timestamp("2026-07-10"), "A"),
        (pd.Timestamp("2026-07-11"), "A"),  # Saturday
        (pd.Timestamp("2026-07-14"), "A"),
        (pd.Timestamp("2026-07-13"), "B"),
        (pd.Timestamp("2026-07-14"), "B"),
    ], names=["date", "asset_id"])

    panel = pd.DataFrame({"close": [1, 2, 3, 4, 5]}, index=idx)
    audit = build_calendar_audit(panel, sessions)

    # 1. Verify index contract (unique, sorted, named)
    assert audit.index.is_unique
    assert audit.index.is_monotonic_increasing
    assert audit.index.names == ("date", "asset_id")

    # 2. A panel missing one known weekday session flags exactly one missing_vendor_row
    assert audit.loc[(pd.Timestamp("2026-07-13"), "A"), "missing_vendor_row"] == True
    assert audit.loc[(slice(None), "A"), "missing_vendor_row"].sum() == 1

    # 3. A Saturday observation flags unexpected_session
    assert audit.loc[(pd.Timestamp("2026-07-11"), "A"), "unexpected_session"] == True
    assert audit.loc[(slice(None), "A"), "unexpected_session"].sum() == 1

    # 4. An asset with a later first observation is not flagged for earlier sessions
    # Friday 2026-07-10 should NOT exist in Asset B's audit window at all
    assert (pd.Timestamp("2026-07-10"), "B") not in audit.index
    # Asset B should have exactly zero missing flags
    assert audit.loc[(slice(None), "B"), "missing_vendor_row"].sum() == 0

def test_calendar_audit_respects_audit_start():
    dates = pd.to_datetime(["2026-07-01", "2026-07-02"]).as_unit("ns")
    idx = pd.MultiIndex.from_arrays([dates, ["AAPL", "AAPL"]], names=["date", "asset_id"])
    panel = pd.DataFrame(index=idx)
    sessions = pd.DatetimeIndex(["2026-07-01", "2026-07-02"])

    audit_start = pd.Timestamp("2026-07-02")
    audit = build_calendar_audit(panel, sessions, audit_start=audit_start)

    assert len(audit) == 1
    assert audit.index.get_level_values("date")[0] == pd.Timestamp("2026-07-02")

def test_calendar_audit_default_has_existing_behavior():
    dates = pd.to_datetime(["2026-07-01", "2026-07-02"]).as_unit("ns")
    idx = pd.MultiIndex.from_arrays([dates, ["AAPL", "AAPL"]], names=["date", "asset_id"])
    panel = pd.DataFrame(index=idx)
    sessions = pd.DatetimeIndex(["2026-07-01", "2026-07-02"])

    audit = build_calendar_audit(panel, sessions)
    assert len(audit) == 2

def test_calendar_audit_rejects_timezone_aware_start():
    dates = pd.to_datetime(["2026-07-01"]).as_unit("ns")
    idx = pd.MultiIndex.from_arrays([dates, ["AAPL"]], names=["date", "asset_id"])
    panel = pd.DataFrame(index=idx)
    sessions = pd.DatetimeIndex(["2026-07-01"])

    with pytest.raises(ValueError, match="timezone-naive"):
        build_calendar_audit(panel, sessions, audit_start=pd.Timestamp("2026-07-01", tz="UTC"))

def test_calendar_audit_rejects_intraday_start():
    dates = pd.to_datetime(["2026-07-01"]).as_unit("ns")
    idx = pd.MultiIndex.from_arrays([dates, ["AAPL"]], names=["date", "asset_id"])
    panel = pd.DataFrame(index=idx)
    sessions = pd.DatetimeIndex(["2026-07-01"])

    with pytest.raises(ValueError, match="normalized"):
        build_calendar_audit(panel, sessions, audit_start=pd.Timestamp("2026-07-01 12:00:00"))

def _make_audit_panel(dates_str, retrieved_ats):
    """Helper to build an arbitrary valid canonical panel for completeness auditing."""
    dates = pd.to_datetime(dates_str).as_unit("ns")
    idx = pd.MultiIndex.from_arrays([dates, ["AAPL"]*len(dates)], names=INDEX_NAMES)
    data = {}
    
    for col, dtype in CANONICAL_DTYPES.items():
        if col == "retrieved_at":
            data[col] = pd.to_datetime(retrieved_ats).tz_convert("UTC")
        elif dtype == "Float64":
            data[col] = [1.0] * len(dates)
        elif dtype == "Int64":
            data[col] = [100] * len(dates)
        elif dtype == "string":
            data[col] = ["mock"] * len(dates)
        elif dtype == "datetime64[ns, UTC]":
            data[col] = [pd.Timestamp("2026-07-01", tz="UTC")] * len(dates)
        else:
            data[col] = [1] * len(dates)
            
    df = pd.DataFrame(data, index=idx)
    for col, dtype in CANONICAL_DTYPES.items():
        df[col] = df[col].astype(dtype)
        
    return df

def test_session_completeness_flags_before_close():
    panel = _make_audit_panel(["2026-07-17"], ["2026-07-17 15:00:00+00:00"])
    audit = build_session_completeness_audit(panel)
    
    assert audit["potentially_incomplete"].iloc[0] == True
    assert audit["session_close_utc"].iloc[0] == pd.Timestamp("2026-07-17 20:00:00", tz="UTC")

def test_session_completeness_accepts_after_close():
    # Exactly on close
    panel1 = _make_audit_panel(["2026-07-17"], ["2026-07-17 20:00:00+00:00"])
    audit1 = build_session_completeness_audit(panel1)
    assert audit1["potentially_incomplete"].iloc[0] == False

    # Well after close
    panel2 = _make_audit_panel(["2026-07-17"], ["2026-07-17 21:00:00+00:00"])
    audit2 = build_session_completeness_audit(panel2)
    assert audit2["potentially_incomplete"].iloc[0] == False

def test_session_completeness_handles_different_dates():
    # 2026-07-16 retrieved AFTER close (not flagged)
    # 2026-07-17 retrieved BEFORE close (flagged)
    panel = _make_audit_panel(
        ["2026-07-16", "2026-07-17"],
        ["2026-07-16 21:00:00+00:00", "2026-07-17 15:00:00+00:00"]
    )
    audit = build_session_completeness_audit(panel)
    
    assert audit["potentially_incomplete"].iloc[0] == False
    assert audit["potentially_incomplete"].iloc[1] == True

def test_session_completeness_handles_non_session_row():
    # 2026-07-18 is a Saturday. It has no close time.
    panel = _make_audit_panel(["2026-07-18"], ["2026-07-18 15:00:00+00:00"])
    audit = build_session_completeness_audit(panel)
    
    assert pd.isna(audit["session_close_utc"].iloc[0])
    assert audit["potentially_incomplete"].iloc[0] == False

def test_session_completeness_does_not_mutate_panel():
    panel = _make_audit_panel(["2026-07-17"], ["2026-07-17 15:00:00+00:00"])
    panel_copy = panel.copy(deep=True)
    build_session_completeness_audit(panel)
    pd.testing.assert_frame_equal(panel, panel_copy)

def test_filter_complete_sessions_removes_flagged_rows():
    panel = _make_audit_panel(
        ["2026-07-16", "2026-07-17"],
        ["2026-07-16 21:00:00+00:00", "2026-07-17 15:00:00+00:00"]
    )
    audit = build_session_completeness_audit(panel)
    filtered = filter_complete_sessions(panel, audit)
    
    assert len(filtered) == 1
    assert filtered.index.get_level_values("date")[0] == pd.Timestamp("2026-07-16")

def test_filter_complete_sessions_preserves_complete_rows():
    panel = _make_audit_panel(["2026-07-16"], ["2026-07-16 21:00:00+00:00"])
    audit = build_session_completeness_audit(panel)
    filtered = filter_complete_sessions(panel, audit)
    
    assert len(filtered) == 1

def test_filter_complete_sessions_rejects_misaligned_audit():
    panel = _make_audit_panel(["2026-07-16"], ["2026-07-16 21:00:00+00:00"])
    audit = build_session_completeness_audit(panel).iloc[0:0]  # Empty audit
    
    with pytest.raises(ValueError, match="exactly match panel index"):
        filter_complete_sessions(panel, audit)

def test_filter_complete_sessions_rejects_missing_flags():
    panel = _make_audit_panel(["2026-07-16"], ["2026-07-16 21:00:00+00:00"])
    audit = build_session_completeness_audit(panel)
    # Cast to pandas nullable boolean to safely inject pd.NA
    audit["potentially_incomplete"] = audit["potentially_incomplete"].astype("boolean")
    audit.loc[audit.index[0], "potentially_incomplete"] = pd.NA
    
    with pytest.raises(ValueError, match="missing values"):
        filter_complete_sessions(panel, audit)

def test_filter_complete_sessions_does_not_mutate_inputs():
    panel = _make_audit_panel(
        ["2026-07-16", "2026-07-17"],
        ["2026-07-16 21:00:00+00:00", "2026-07-17 15:00:00+00:00"]
    )
    audit = build_session_completeness_audit(panel)
    
    panel_copy = panel.copy(deep=True)
    audit_copy = audit.copy(deep=True)
    
    filter_complete_sessions(panel, audit)
    
    pd.testing.assert_frame_equal(panel, panel_copy)
    pd.testing.assert_frame_equal(audit, audit_copy)