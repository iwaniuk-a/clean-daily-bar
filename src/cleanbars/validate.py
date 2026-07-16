import pandas as pd
from cleanbars.normalize import validate_panel


# here we will flag abmnormalities but not repair them
def build_structural_checks(panel):
    # Validate the incoming schema first
    validate_panel(panel)

    # Vectorized price calculations
    price_columns = ["open_raw", "high_raw", "low_raw", "close_raw"]
    body_min = panel[["open_raw", "close_raw"]].min(axis=1)
    body_max = panel[["open_raw", "close_raw"]].max(axis=1)

    # Construct the boolean checks
    nonpositive_price = panel[price_columns].le(0).any(axis=1)
    negative_volume = panel["volume_raw"].lt(0)
    low_above_body = panel["low_raw"].gt(body_min)
    high_below_body = panel["high_raw"].lt(body_max)
    high_below_low = panel["high_raw"].lt(panel["low_raw"])

    # Return as a new DataFrame preserving the original index and strict column order
    return pd.DataFrame({
        "nonpositive_price": nonpositive_price,
        "negative_volume": negative_volume,
        "low_above_body": low_above_body,
        "high_below_body": high_below_body,
        "high_below_low": high_below_low
    }, index=panel.index)

def summarize_checks(checks):
    return checks.groupby(level="asset_id").sum().astype(int)