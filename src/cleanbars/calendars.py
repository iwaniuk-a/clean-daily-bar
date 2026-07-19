import pandas as pd
import exchange_calendars as xcals
from cleanbars.normalize import validate_panel

def expected_sessions(start_date, end_date, calendar_name="XNYS"):

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError("start_date cannot be later than end_date.")

    calendar = xcals.get_calendar("XNYS", start="1960-01-01")

    sessions = calendar.sessions_in_range(start, end)

    sessions = sessions.tz_localize(None).normalize()
    sessions.name = "date"

    return sessions

def build_calendar_audit(panel, sessions, audit_start=None):
    import pandas as pd
    if audit_start is not None:
        audit_start = pd.Timestamp(audit_start)
        if audit_start.tzinfo is not None:
            raise ValueError("audit_start must be timezone-naive")
        if audit_start != audit_start.normalize():
            raise ValueError("audit_start must be normalized")

        # Filter expected sessions and observed dates to the audit window
        sessions = sessions[sessions >= audit_start]
        panel = panel.loc[panel.index.get_level_values("date") >= audit_start]

    # Gather all possible dates and unique assets
    dates_in_panel = panel.index.get_level_values("date")
    assets = panel.index.get_level_values("asset_id").unique()
    all_dates = sessions.union(dates_in_panel.unique()).sort_values()

    full_idx = pd.MultiIndex.from_product([all_dates, assets], names=["date", "asset_id"])
    audit = pd.DataFrame(index=full_idx)

    audit["_date"] = audit.index.get_level_values("date")
    audit["observed"] = audit.index.isin(panel.index)
    audit["expected"] = audit["_date"].isin(sessions)

    # Use transform on grouped observed dates to find the bounds for each asset
    # .where() places NaT where it wasn't observed, which .transform("min") safely ignores
    observed_dates = audit["_date"].where(audit["observed"])
    min_dates = observed_dates.groupby(level="asset_id").transform("min")
    max_dates = observed_dates.groupby(level="asset_id").transform("max")

    # Filter out anything outside the asset's active lifespan
    is_active = (audit["_date"] >= min_dates) & (audit["_date"] <= max_dates)
    audit = audit[is_active].copy()

    # Calculate the final missing/unexpected flags
    audit["missing_vendor_row"] = audit["expected"] & ~audit["observed"]
    audit["unexpected_session"] = audit["observed"] & ~audit["expected"]

    # Return exactly the required columns in the correct order
    return audit[["missing_vendor_row", "unexpected_session", "observed", "expected"]]


def build_session_completeness_audit(panel, calendar_name="XNYS",):
    validate_panel(panel)

    calendar = xcals.get_calendar(calendar_name, start="1960-01-01")
    dates = panel.index.get_level_values("date")
    unique_dates = dates.unique()

    close_times = {}
    for d in unique_dates:
        if calendar.is_session(d):
            close_times[d] = calendar.session_close(d).tz_convert("UTC")
        else:
            close_times[d] = pd.NaT

    audit = pd.DataFrame(index=panel.index)
    audit['retrieved_at'] = panel["retrieved_at"]
    audit["session_close_utc"] = pd.Series(
        dates.map(close_times),
        index=panel.index,
        dtype="datetime64[ns, UTC]"
    )

    # Flag is True if retrieved before the session closed.
    # Evaluating against NaT natively results in False (for non-sessions)
    audit["potentially_incomplete"] = audit["retrieved_at"] < audit["session_close_utc"]

    return audit

def filter_complete_sessions(panel, completeness_audit):
    validate_panel(panel)
    
    if not completeness_audit.index.equals(panel.index):
        raise ValueError("Audit index must exactly match panel index.")
    
    if "potentially_incomplete" not in completeness_audit.columns:
        raise ValueError("Audit must contain a 'potentially_incomplete' column")
    
    if completeness_audit["potentially_incomplete"].isna().any():
        raise ValueError("'potentially_incomplete' column can not have any missing values")
    
    filtered = panel.loc[~completeness_audit["potentially_incomplete"]].copy()
    validate_panel(filtered)

    return filtered