import math
import pandas as pd
from cleanbars.normalize import CANONICAL_COLUMNS, CANONICAL_DTYPES, INDEX_NAMES
from cleanbars.validate import build_vendor_comparison

def _make_panel(rows, index_tuples):
    df = pd.DataFrame(rows, columns=["open_raw", "high_raw", "low_raw", "close_raw", "volume_raw"]) 
    df["vendor_symbol"] = "TICKER"
    df["adj_close"] = df["close_raw"]
    df["cash_dividend"] = 0.0
    df["split_factor"] = 1.0
    df["source"] = "TEST"
    df["retrieved_at"] = pd.Timestamp("2026-07-01", tz="UTC")
    for col, dtype in CANONICAL_DTYPES.items():
        df[col] = df[col].astype(dtype)
    df = df.loc[:, list(CANONICAL_COLUMNS)]
    df.index = pd.MultiIndex.from_tuples(index_tuples, names=INDEX_NAMES)
    return df.sort_index()

def test_build_vendor_comparison():
    # Setup yfinance panel
    yf_rows = [
        [10.0, 10.0, 10.0, 100.0, 1000],  # MATCH
        [10.0, 10.0, 10.0, 100.0, 2000],  # DIFF
        [10.0, 10.0, 10.0, 100.0, 1000],  # YF_ONLY
        [10.0, 10.0, 10.0, 0.0, 0],       # ZERO
    ]
    yf_idx = [
        (pd.Timestamp("2026-07-01"), "MATCH"),
        (pd.Timestamp("2026-07-01"), "DIFF"),
        (pd.Timestamp("2026-07-01"), "YF_ONLY"),
        (pd.Timestamp("2026-07-01"), "ZERO"),
    ]
    yf_panel = _make_panel(yf_rows, yf_idx)
    yf_panel_original = yf_panel.copy(deep=True)
    
    # Setup alpha_vantage panel
    av_rows = [
        [10.0, 10.0, 10.0, 100.0, 1000],  # MATCH
        [10.0, 10.0, 10.0, 110.0, 2200],  # DIFF (close: 110)
        [10.0, 10.0, 10.0, 100.0, 1000],  # AV_ONLY
        [10.0, 10.0, 10.0, 0.0, 0],       # ZERO
    ]
    av_idx = [
        (pd.Timestamp("2026-07-01"), "MATCH"),
        (pd.Timestamp("2026-07-01"), "DIFF"),
        (pd.Timestamp("2026-07-01"), "AV_ONLY"),
        (pd.Timestamp("2026-07-01"), "ZERO"),
    ]
    av_panel = _make_panel(av_rows, av_idx)
    av_panel_original = av_panel.copy(deep=True)
    
    comp = build_vendor_comparison(yf_panel, av_panel)
    
    # 1. Input panels are not mutated
    pd.testing.assert_frame_equal(yf_panel, yf_panel_original)
    pd.testing.assert_frame_equal(av_panel, av_panel_original)
    
    # 2. Identical close and volume produce zero diffs
    assert comp.loc[(pd.Timestamp("2026-07-01"), "MATCH"), "close_abs_diff"] == 0
    assert comp.loc[(pd.Timestamp("2026-07-01"), "MATCH"), "close_rel_diff"] == 0
    
    # 3. Known values produce exact symmetric rel diff (100 vs 110 = 10 / 105)
    expected_rel_diff = 10.0 / 105.0
    assert math.isclose(comp.loc[(pd.Timestamp("2026-07-01"), "DIFF"), "close_rel_diff"], expected_rel_diff)
    
    # 4. Keys present in only one vendor remain with NaN differences
    assert (pd.Timestamp("2026-07-01"), "YF_ONLY") in comp.index
    assert comp.loc[(pd.Timestamp("2026-07-01"), "YF_ONLY"), "yfinance_observed"] == True
    assert comp.loc[(pd.Timestamp("2026-07-01"), "YF_ONLY"), "alpha_vantage_observed"] == False
    assert pd.isna(comp.loc[(pd.Timestamp("2026-07-01"), "YF_ONLY"), "close_rel_diff"])
    
    # 5. Zero handling doesn't divide by zero
    assert comp.loc[(pd.Timestamp("2026-07-01"), "ZERO"), "close_rel_diff"] == 0.0