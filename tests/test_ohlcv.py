import pandas as pd
from cleanbars.normalize import CANONICAL_COLUMNS, CANONICAL_DTYPES, INDEX_NAMES
from cleanbars.validate import build_structural_checks, summarize_checks

def _make_panel(rows, index_tuples):
    df = pd.DataFrame(rows, columns=["open_raw", "high_raw", "low_raw", "close_raw", "volume_raw"]) 

    # Fill in required dummy data for the rest of the schema
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

def test_structural_checks_valid_panel():
    rows = [
        [10.0, 12.0, 9.0, 11.0, 100],
        [11.0, 13.0, 10.0, 12.0, 200]
    ]
    idx = [(pd.Timestamp("2026-07-01"), "AAPL"), (pd.Timestamp("2026-07-02"), "AAPL")]
    panel = _make_panel(rows, idx)

    checks = build_structural_checks(panel)

    # A valid panel should have absolutely no anomalies
    assert not checks.any().any() 
    assert checks.index.equals(panel.index)
    assert tuple(checks.columns) == (
        "nonpositive_price",
        "negative_volume",
        "low_above_body",
        "high_below_body",
        "high_below_low",
    )

def test_structural_checks_flag_anomalies():
    rows = [
        [0.0, 12.0, 9.0, 11.0, 100],     # Row 0: nonpositive_price (open is 0)
        [10.0, 12.0, 9.0, 11.0, -50],    # Row 1: negative_volume
        [10.0, 12.0, 11.5, 11.0, 100],   # Row 2: low_above_body (low 11.5 > open 10.0)
        [10.0, 9.0, 8.0, 11.0, 100],     # Row 3: high_below_body (high 9.0 < close 11.0)
        [10.0, 5.0, 15.0, 11.0, 100],    # Row 4: high_below_low (high 5.0 < low 15.0)
    ]
    idx = [
        (pd.Timestamp("2026-07-01"), "ERR1"),
        (pd.Timestamp("2026-07-02"), "ERR2"),
        (pd.Timestamp("2026-07-03"), "ERR3"),
        (pd.Timestamp("2026-07-04"), "ERR4"),
        (pd.Timestamp("2026-07-05"), "ERR5"),
    ]
    panel = _make_panel(rows, idx)
    checks = build_structural_checks(panel)

    # Verify individual cells trigger correctly
    assert checks.loc[("2026-07-01", "ERR1"), "nonpositive_price"] == True
    assert checks.loc[("2026-07-02", "ERR2"), "negative_volume"] == True
    assert checks.loc[("2026-07-03", "ERR3"), "low_above_body"] == True
    assert checks.loc[("2026-07-04", "ERR4"), "high_below_body"] == True
    assert checks.loc[("2026-07-05", "ERR5"), "high_below_low"] == True

def test_summarize_checks_by_asset():
    rows = [
        [10.0, 12.0, 9.0, 11.0, -100],   # AAPL: negative volume
        [10.0, 12.0, 9.0, 11.0, -50],    # AAPL: negative volume again
        [10.0, 12.0, 9.0, 11.0, 100],    # MSFT: completely valid
        [-5.0, 12.0, 9.0, 11.0, 100],    # TSLA: nonpositive price
    ]
    idx = [
        (pd.Timestamp("2026-07-01"), "AAPL"),
        (pd.Timestamp("2026-07-02"), "AAPL"),
        (pd.Timestamp("2026-07-01"), "MSFT"),
        (pd.Timestamp("2026-07-01"), "TSLA"),
    ]
    panel = _make_panel(rows, idx)
    checks = build_structural_checks(panel)
    summary = summarize_checks(checks)
    
    # AAPL should have exactly 2 negative volume flags and 0 others
    assert summary.loc["AAPL", "negative_volume"] == 2
    assert summary.loc["AAPL", "nonpositive_price"] == 0
    
    # MSFT should be a completely clean slate
    assert summary.loc["MSFT"].sum() == 0
    
    # TSLA should have 1 nonpositive price flag
    assert summary.loc["TSLA", "nonpositive_price"] == 1