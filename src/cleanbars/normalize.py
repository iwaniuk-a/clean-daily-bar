import pandas as pd
from pathlib import Path

CANONICAL_COLUMNS = (
    "vendor_symbol",
    "open_raw",
    "high_raw",
    "low_raw",
    "close_raw",
    "volume_raw",
    "adj_close",
    "cash_dividend",
    "split_factor",
    "source",
    "retrieved_at",
)
CANONICAL_DTYPES = {
    "vendor_symbol": "string",
    "open_raw": "Float64",
    "high_raw": "Float64",
    "low_raw": "Float64",
    "close_raw": "Float64",
    "volume_raw": "Int64",
    "adj_close": "Float64",
    "cash_dividend": "Float64",
    "split_factor": "Float64",
    "source": "string",
    "retrieved_at": "datetime64[ns, UTC]",
}
INDEX_NAMES = ("date", "asset_id")

def validate_index(panel):
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("Index must be a pandas MultiIndex")
    if tuple(panel.index.names) != INDEX_NAMES:
        raise ValueError(f"Index names must exactly match {INDEX_NAMES}")
    if not panel.index.is_unique:
        raise ValueError("Index keys must be unique")
    if not panel.index.is_monotonic_increasing:
        raise ValueError("Index must be monotonically increasing")
    
    dates = panel.index.get_level_values("date")

    if not isinstance(dates, pd.DatetimeIndex):
        raise ValueError("Dates must me in datetime format")
    if dates.tz is not None:
        raise ValueError("Dates must be timezone-naive")
    if not dates.equals(dates.normalize()):
        raise ValueError("Every timestamp must equal its normalized value")
    
def validate_columns(panel):
    missing = [column for column in CANONICAL_COLUMNS if column not in panel.columns]
    unexpected = [column for column in panel.columns if column not in CANONICAL_COLUMNS]
    if missing:
        raise ValueError("Missing canonical columns")
    if unexpected:
        raise ValueError("Unexpected extra columns")
    if tuple(panel.columns) != CANONICAL_COLUMNS:
        raise ValueError("Correct columns in the wrong order")
    
def validate_dtypes(panel):
    mismatches = {}
    for column, expected_dtype in CANONICAL_DTYPES.items():
        actual_dtype = str(panel[column].dtype)
        if actual_dtype != expected_dtype:
            mismatches[column] = f"expected {expected_dtype}, got {actual_dtype}"
            
    if mismatches:
        raise ValueError(f"Column dtype mismatches found: {mismatches}")
    
def validate_panel(panel):
    if panel.index.get_level_values("date").dtype != "datetime64[ns]":
        raise ValueError("date level in index must have nanosecond precision (datetime64[ns]).")
    validate_index(panel)
    validate_columns(panel)
    validate_dtypes(panel)

###
def save_panel_parquet(panel, path):
    panel_copy = panel.copy()
    panel_copy = panel_copy.sort_index()

    out_path = Path(path)
    validate_panel(panel_copy)
    panel_copy.to_parquet(path=out_path, engine="pyarrow", compression="zstd")

    reloaded = pd.read_parquet(path = out_path)
    validate_panel(reloaded)

    if len(panel_copy) != len(reloaded):
        raise ValueError("Round-trip failed: row count differs.")
        
    if tuple(panel_copy.columns) != tuple(reloaded.columns):
        raise ValueError("Round-trip failed: column order differs.")
        
    if tuple(panel_copy.index.names) != tuple(reloaded.index.names):
        raise ValueError("Round-trip failed: index names differ.")
        
    if tuple(panel_copy.dtypes.astype(str)) != tuple(reloaded.dtypes.astype(str)):
        raise ValueError("Round-trip failed: dtypes differ.")
        
    return out_path

###
def normalize_yfinance_symbol(raw, asset_id, vendor_symbol, retrieved_at):
    df = raw.copy()
    
    if not isinstance(df.columns, pd.MultiIndex):
        raise ValueError("Raw yfinance columns must be a pd.MultiIndex.")
        
    if vendor_symbol not in df.columns.get_level_values("Ticker"):
        raise ValueError(f"Symbol '{vendor_symbol}' not found in 'Ticker' level.")
    df = df.xs(vendor_symbol, axis=1, level="Ticker")
    
    required_fields = ["Open", "High", "Low", "Close", "Volume", "Adj Close", "Dividends", "Stock Splits"]
    for field in required_fields:
        if field not in df.columns:
            raise ValueError(f"Required field '{field}' is missing from raw data.")
            
    df = df[required_fields].copy()
    
    rename_map = {
        "Open": "open_raw",
        "High": "high_raw",
        "Low": "low_raw",
        "Close": "close_raw",
        "Volume": "volume_raw",
        "Adj Close": "adj_close",
        "Dividends": "cash_dividend",
        "Stock Splits": "split_factor",
    }
    df = df.rename(columns=rename_map)
    df["split_factor"] = df["split_factor"].replace(0.0, 1.0)
    
    # Robust retrieved_at parsing
    retrieved_at = pd.Timestamp(retrieved_at)
    if retrieved_at.tzinfo is None:
        raise ValueError("retrieved_at must be a timezone-aware Timestamp.")
    
    df["vendor_symbol"] = vendor_symbol
    df["source"] = "yfinance"
    df["retrieved_at"] = retrieved_at.tz_convert("UTC")
    
    # Date formatting to explicit nanosecond precision
    dates = pd.DatetimeIndex(pd.to_datetime(df.index))
    if dates.tz is not None:
        dates = dates.tz_localize(None)
    dates = dates.normalize().as_unit("ns")
    
    df.index = pd.MultiIndex.from_arrays(
        [dates, [asset_id] * len(df)], 
        names=INDEX_NAMES
    )
    
    df = df.reindex(columns=list(CANONICAL_COLUMNS))
    for col, dtype in CANONICAL_DTYPES.items():
        df[col] = df[col].astype(dtype)
        
    df.columns.name = None
    df = df.sort_index()
    
    validate_panel(df)
    
    return df