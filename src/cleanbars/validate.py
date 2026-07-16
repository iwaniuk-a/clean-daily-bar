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


def build_jump_checks(panel, threshold=0.25):
    validate_panel(panel)

    if threshold <= 0:
        raise ValueError('threshold must be positive')
    
    returns = panel["close_raw"].groupby(level="asset_id").pct_change(fill_method=None)
    jumps = returns.abs().gt(threshold).fillna(False)

    return pd.DataFrame({
        "raw_close_return": returns,
        "suspicious_jump": jumps
    }, index=panel.index)

def build_vendor_comparison(yfinance_panel, alpha_vantage_panel):
    validate_panel(yfinance_panel)
    validate_panel(alpha_vantage_panel)

    yf = yfinance_panel[["close_raw", "volume_raw"]].rename(
    columns={
        "close_raw": "yfinance_close_raw",
        "volume_raw": "yfinance_volume_raw",
    }
    ).copy()
    yf["yfinance_observed"] = True

    av = alpha_vantage_panel[["close_raw", "volume_raw"]].rename(
    columns={
        "close_raw": "alpha_vantage_close_raw",
        "volume_raw": "alpha_vantage_volume_raw"
    }
    ).copy()
    av["alpha_vantage_observed"] = True

    joined = yf.join(av, how="outer").sort_index()
    
    joined["yfinance_observed"] = (
        joined["yfinance_observed"].fillna(False).astype(bool)
    )
    joined["alpha_vantage_observed"] = (
        joined["alpha_vantage_observed"].fillna(False).astype(bool)
    )

    # 1. Close Differences
    joined["close_abs_diff"] = (joined["yfinance_close_raw"] - joined["alpha_vantage_close_raw"]).abs()
    close_denom = (joined["yfinance_close_raw"].abs() + joined["alpha_vantage_close_raw"].abs()) / 2.0
    joined["close_rel_diff"] = (joined["close_abs_diff"] / close_denom)
    
    both_zero_close = (joined["yfinance_close_raw"] == 0) & (joined["alpha_vantage_close_raw"] == 0)
    joined.loc[both_zero_close, "close_rel_diff"] = 0.0
    
    # 2. Volume Differences
    joined["volume_abs_diff"] = (joined["yfinance_volume_raw"] - joined["alpha_vantage_volume_raw"]).abs()
    vol_denom = (joined["yfinance_volume_raw"].abs() + joined["alpha_vantage_volume_raw"].abs()) / 2.0
    joined["volume_rel_diff"] = (joined["volume_abs_diff"] / vol_denom)
    
    both_zero_vol = (joined["yfinance_volume_raw"] == 0) & (joined["alpha_vantage_volume_raw"] == 0)
    joined.loc[both_zero_vol, "volume_rel_diff"] = 0.0
    
    return joined[[
        "yfinance_observed",
        "alpha_vantage_observed",
        "yfinance_close_raw",
        "alpha_vantage_close_raw",
        "close_abs_diff",
        "close_rel_diff",
        "yfinance_volume_raw",
        "alpha_vantage_volume_raw",
        "volume_abs_diff",
        "volume_rel_diff",
    ]]