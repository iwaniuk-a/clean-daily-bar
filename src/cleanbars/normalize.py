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