import pandas as pd
import exchange_calendars as xcals

def expected_sessions(start_date, end_date, calendar_name="XNYS"):

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError("start_date cannot be later than end_date.")
    
    calendar = xcals.get_calendar(calendar_name)

    sessions = calendar.sessions_in_range(start, end)

    sessions = sessions.tz_localize(None).normalize()
    sessions.name = "date"

    return sessions

def build_calendar_audit(panel, sessions):
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

