import pandas as pd
import time
from cleanbars.download import (
    resolve_vendor_symbol, 
    download_yfinance_symbol,
    download_alpha_vantage_symbol
)
from cleanbars.normalize import (
    combine_normalized_panels, 
    normalize_yfinance_symbol,
    normalize_alpha_vantage_symbol
)

def build_yfinance_universe(
    config,
    retrieved_at,
    downloader=download_yfinance_symbol,
    normalizer=normalize_yfinance_symbol,
):
    universe = config["universe"]
    symbol_overrides = config["symbol_overrides"]
    yf_settings = config["vendors"]["yfinance"]

    if not universe:
        raise ValueError("Universe cannot be empty.")

    if len(universe) != len(set(universe)):
        raise ValueError("Universe contains duplicate canonical asset IDs.")

    retrieved_at = pd.Timestamp(retrieved_at)

    if retrieved_at.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware.")

    retrieved_at = retrieved_at.tz_convert("UTC")
    
    normalized_panels = []
    raw_by_asset = {}
    
    for asset_id in universe:
        vendor_symbol = resolve_vendor_symbol(asset_id, "yfinance", symbol_overrides)
        
        try:
            raw = downloader(vendor_symbol, yf_settings)
            norm = normalizer(raw, asset_id, vendor_symbol, retrieved_at)
            
            normalized_panels.append(norm)
            raw_by_asset[asset_id] = raw
        except Exception as exc:
            raise RuntimeError(
                f"Failed to process asset '{asset_id}' (Vendor: {vendor_symbol})"
            ) from exc
            
    combined = combine_normalized_panels(normalized_panels)
    return combined, raw_by_asset

def build_alpha_vantage_universe(
    config,
    api_key,
    retrieved_at,
    session,
    downloader=download_alpha_vantage_symbol,
    normalizer=normalize_alpha_vantage_symbol,
    request_interval_seconds=1.1,
    sleeper=time.sleep,
):
    if request_interval_seconds < 0:
        raise ValueError("request_interval_seconds must be non-negative.")
    
    universe = config["universe"]
    symbol_overrides = config["symbol_overrides"]
    av_settings = config["vendors"]["alpha_vantage"]

    if not universe:
        raise ValueError("Universe cannot be empty.")

    if len(universe) != len(set(universe)):
        raise ValueError("Universe contains duplicate canonical asset IDs.")

    if not api_key or not str(api_key).strip():
        raise ValueError("API key cannot be blank.")

    retrieved_at = pd.Timestamp(retrieved_at)

    if retrieved_at.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware.")

    retrieved_at = retrieved_at.tz_convert("UTC")
    
    normalized_panels = []
    raw_by_asset = {}
    
    for position, asset_id in enumerate(universe):
        if position > 0:
            sleeper(request_interval_seconds)
            
        vendor_symbol = resolve_vendor_symbol(asset_id, "alpha_vantage", symbol_overrides)
        
        try:
            raw = downloader(vendor_symbol, av_settings, api_key, session)
            norm = normalizer(raw, asset_id, vendor_symbol, retrieved_at)
            normalized_panels.append(norm)
            raw_by_asset[asset_id] = raw
        except Exception as exc:
            raise RuntimeError(f"Failed to process asset '{asset_id}' (Vendor: {vendor_symbol})") from exc
            
    return combine_normalized_panels(normalized_panels), raw_by_asset
 

