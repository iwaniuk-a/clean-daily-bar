import pandas as pd
from cleanbars.download import resolve_vendor_symbol
from cleanbars.normalize import combine_normalized_panels


def build_yfinance_universe(
    config,
    retrieved_at,
    downloader,
    normalizer,
):
    if not config.get("universe"):
        raise ValueError("Universe cannot be empty.")
    if retrieved_at.tzinfo is None:
        raise ValueError("retrieved_at must be timezone-aware.")
    
    symbol_overrides = config["symbol_overrides"]
    yf_settings = config["vendors"]["yfinance"]
    
    normalized_panels = []
    raw_by_asset = {}
    seen_assets = set()
    
    for asset_id in config["universe"]:
        if asset_id in seen_assets:
            raise ValueError(f"Duplicate canonical asset ID: {asset_id}")
        seen_assets.add(asset_id)
        
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