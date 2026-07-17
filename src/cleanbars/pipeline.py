import pandas as pd
from cleanbars.download import resolve_vendor_symbol, download_yfinance_symbol
from cleanbars.normalize import combine_normalized_panels, normalize_yfinance_symbol

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
