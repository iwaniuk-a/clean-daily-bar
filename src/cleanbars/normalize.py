import pandas as pd

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