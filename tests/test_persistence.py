import pandas as pd
import pytest

from cleanbars.normalize import save_panel_parquet
from cleanbars.normalize import validate_index
from cleanbars.normalize import CANONICAL_COLUMNS, CANONICAL_DTYPES, INDEX_NAMES
from cleanbars.normalize import validate_columns
from cleanbars.normalize import validate_dtypes
from cleanbars.normalize import validate_panel

def _build_fully_typed_panel(sorted_idx=True):

    if sorted_idx:
        idx = [(pd.Timestamp("2026-07-01"), "AAPL"), (pd.Timestamp("2026-07-02"), "AAPL")]
    else:
        idx = [(pd.Timestamp("2026-07-02"), "AAPL"), (pd.Timestamp("2026-07-01"), "AAPL")]
        
    index = pd.MultiIndex.from_tuples(idx, names=INDEX_NAMES)
    
    data = {}
    for col, dtype in CANONICAL_DTYPES.items():
        if dtype == "string":
            data[col] = pd.Series(["A", "B"], dtype=dtype)
        elif dtype == "Float64":
            data[col] = pd.Series([1.5, 2.5], dtype=dtype)
        elif dtype == "Int64":
            data[col] = pd.Series([100, 200], dtype=dtype)
        elif dtype == "datetime64[ns, UTC]":
            data[col] = pd.Series(
                [pd.Timestamp("2026-07-01", tz="UTC"), pd.Timestamp("2026-07-02", tz="UTC")], 
                dtype=dtype
            )
            
    return pd.DataFrame(data, index=index)

def test_save_panel_parquet_round_trip(tmp_path):
    panel = _build_fully_typed_panel(sorted_idx=True)

    output_path = tmp_path / "panel.parquet"
    save_panel_parquet(panel, output_path)
    reloaded = pd.read_parquet(output_path)
    
    pd.testing.assert_frame_equal(panel, reloaded)

    result = save_panel_parquet(panel, output_path)

    assert result == output_path
    assert output_path.exists()

def test_save_panel_parquet_sorts_index(tmp_path):
    panel = _build_fully_typed_panel(sorted_idx=False)
    output_path = tmp_path / "panel_unsorted.parquet"
    
    # Verify the original in-memory panel remains unsorted
    assert not panel.index.is_monotonic_increasing
    
    save_panel_parquet(panel, output_path)
    
    # Verify the original is still unsorted (immutability)
    assert not panel.index.is_monotonic_increasing
    
    # Verify the loaded Parquet panel is monotonically increasing
    reloaded = pd.read_parquet(output_path)
    assert reloaded.index.is_monotonic_increasing

def test_save_panel_parquet_rejects_invalid_panel(tmp_path):
    output_path = tmp_path / "panel_missing.parquet"

    index = pd.MultiIndex.from_tuples([(pd.Timestamp("2026-07-01"), "AAPL")], names=INDEX_NAMES)

    panel = pd.DataFrame(columns=CANONICAL_COLUMNS[:-1], index = index)
    with pytest.raises(ValueError, match="Missing"):
        save_panel_parquet(panel, output_path)

    assert not output_path.exists()