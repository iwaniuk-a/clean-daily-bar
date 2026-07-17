import pytest
import pandas as pd
import importlib
import os
from cleanbars.normalize import CANONICAL_DTYPES, INDEX_NAMES

module_name = None
for f in os.listdir("src/cleanbars"):
    if f.endswith(".py"):
        with open(os.path.join("src/cleanbars", f)) as file:
            if "def build_jump_forensics" in file.read():
                module_name = f[:-3]
                break

mod = importlib.import_module(f"cleanbars.{module_name}")
build_jump_forensics = mod.build_jump_forensics

def _make_jump_panel(aapl_closes=[100.0, 101.0, 100.0], msft_closes=[50.0, 51.0]):
    """Creates a fresh, immutable panel using lists to prevent silent index alignment failures."""
    dates = pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-03", "2026-07-01", "2026-07-02"]).as_unit("ns")
    assets = ["AAPL", "AAPL", "AAPL", "MSFT", "MSFT"]
    idx = pd.MultiIndex.from_arrays([dates, assets], names=INDEX_NAMES)
    
    data = {}
    for col, dtype in CANONICAL_DTYPES.items():
        if dtype == "Float64":
            data[col] = [100.0]*5
        elif dtype == "string":
            data[col] = ["test"]*5
        elif dtype == "datetime64[ns, UTC]":
            data[col] = [pd.Timestamp("2026-07-01", tz="UTC")]*5
        else:
            data[col] = [1.0]*5
            
    # Set explicit non-action defaults so jump checks correctly trigger
    data["split_factor"] = [pd.NA]*5
    data["cash_dividend"] = [pd.NA]*5
    data["source"] = ["yfinance"]*5
    data["vendor_symbol"] = ["A"]*5
    
    # Inject the specific test sequences directly as lists
    data["close_raw"] = aapl_closes + msft_closes
    
    # Create the dataframe, then apply dtypes cleanly
    df = pd.DataFrame(data, index=idx)
    for col, dtype in CANONICAL_DTYPES.items():
        df[col] = df[col].astype(dtype)
        
    return df.sort_index()

def test_build_jump_forensics_no_suspicious_rows():
    # 1% moves -> no jumps
    panel = _make_jump_panel()
    res = build_jump_forensics(panel, threshold=0.25)
    assert res.empty

def test_build_jump_forensics_previous_close_resets_by_asset():
    # MSFT drops 50.0 -> 10.0 (80% drop)
    panel = _make_jump_panel(msft_closes=[50.0, 10.0])
    res = build_jump_forensics(panel, threshold=0.25)
    
    # Extract specific row safely
    msft_jump = res.xs("MSFT", level="asset_id")
    assert msft_jump.loc[pd.Timestamp("2026-07-02"), "previous_close_raw"] == 50.0

def test_build_jump_forensics_exact_output_column_order():
    panel = _make_jump_panel(msft_closes=[50.0, 10.0])
    res = build_jump_forensics(panel, threshold=0.25)
    
    expected = ["previous_close_raw", "close_raw", "raw_close_return", "abs_return", "split_factor", "cash_dividend", "source"]
    assert list(res.columns) == expected

def test_build_jump_forensics_no_implicit_sorting():
    # AAPL jumps: 100 -> 200 (+100%), then 200 -> 10 (-95%)
    panel = _make_jump_panel(aapl_closes=[100.0, 200.0, 10.0])
    res = build_jump_forensics(panel, threshold=0.25)
    
    # Should retain input order (date ascending), not absolute return descending
    dates = res.index.get_level_values("date")
    assert dates[0] == pd.Timestamp("2026-07-02")
    assert dates[1] == pd.Timestamp("2026-07-03")

def test_build_jump_forensics_input_panel_unchanged():
    panel = _make_jump_panel()
    panel_copy = panel.copy(deep=True)
    build_jump_forensics(panel, threshold=0.25)
    pd.testing.assert_frame_equal(panel, panel_copy)
