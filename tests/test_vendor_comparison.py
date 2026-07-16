import pandas as pd
import pytest
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
    yf_rows = [
        [10.0, 10.0, 10.0, 100.0, 1000],  # MATCH
        [10.0, 10.0, 10.0, 100.0, 2000],  # DIFF
        [10.0, 10.0, 10.0, 100.0, 1000],  # YF_ONLY
        [10.0, 10.0, 10.0, pd.NA, 1000],  # NA_CLOSE (Exists, but NaN close)
        [10.0, 10.0, 10.0, 0.0, 0],       # ZERO
    ]
    yf_idx = [
        (pd.Timestamp("2026-07-01"), "MATCH"),
        (pd.Timestamp("2026-07-01"), "DIFF"),
        (pd.Timestamp("2026-07-01"), "YF_ONLY"),
        (pd.Timestamp("2026-07-01"), "NA_CLOSE"),
        (pd.Timestamp("2026-07-01"), "ZERO"),
    ]
    
    av_rows = [
        [10.0, 10.0, 10.0, 100.0, 1000],  # MATCH
        [10.0, 10.0, 10.0, 110.0, 2200],  # DIFF
        [10.0, 10.0, 10.0, 0.0, 0],       # ZERO
    ]
    av_idx = [
        (pd.Timestamp("2026-07-01"), "MATCH"),
        (pd.Timestamp("2026-07-01"), "DIFF"),
        (pd.Timestamp("2026-07-01"), "ZERO"),
    ]
    
    yf_panel = _make_panel(yf_rows, yf_idx)
    av_panel = _make_panel(av_rows, av_idx)
    
    yf_original = yf_panel.copy(deep=True)
    av_original = av_panel.copy(deep=True)
    
    comp = build_vendor_comparison(yf_panel, av_panel)
    
    # 1. Copies of both inputs remain equal to their original versions
    pd.testing.assert_frame_equal(yf_panel, yf_original)
    pd.testing.assert_frame_equal(av_panel, av_original)
    
    # 2. Identical values produce zero absolute and relative differences
    assert comp.loc[(pd.Timestamp("2026-07-01"), "MATCH"), "close_abs_diff"] == 0.0
    assert comp.loc[(pd.Timestamp("2026-07-01"), "MATCH"), "close_rel_diff"] == 0.0
    
    # 3. 100 versus 110 produces approx(10/105)
    assert comp.loc[(pd.Timestamp("2026-07-01"), "DIFF"), "close_rel_diff"] == pytest.approx(10 / 105)
    
    # 4. Row present only in YF remains in output, observed flag True, AV False, diffs missing
    assert (pd.Timestamp("2026-07-01"), "YF_ONLY") in comp.index
    assert comp.loc[(pd.Timestamp("2026-07-01"), "YF_ONLY"), "yfinance_observed"] == True
    assert comp.loc[(pd.Timestamp("2026-07-01"), "YF_ONLY"), "alpha_vantage_observed"] == False
    assert pd.isna(comp.loc[(pd.Timestamp("2026-07-01"), "YF_ONLY"), "close_abs_diff"])
    assert pd.isna(comp.loc[(pd.Timestamp("2026-07-01"), "YF_ONLY"), "close_rel_diff"])
    
    # 5. Row that exists but has close_raw=pd.NA is still marked as observed
    assert comp.loc[(pd.Timestamp("2026-07-01"), "NA_CLOSE"), "yfinance_observed"] == True
    
    # 6. Two zero values produce relative difference 0.0
    assert comp.loc[(pd.Timestamp("2026-07-01"), "ZERO"), "close_rel_diff"] == 0.0