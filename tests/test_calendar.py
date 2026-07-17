import pandas as pd
import pytest
from cleanbars.calendars import expected_sessions, build_calendar_audit

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
import pandas as pd
import pytest
from cleanbars.calendars import build_calendar_audit

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
