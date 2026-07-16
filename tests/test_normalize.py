import pandas as pd
import pytest
from cleanbars.normalize import normalize_yfinance_symbol, normalize_alpha_vantage_symbol, CANONICAL_COLUMNS

# --- YFinance Tests ---

@pytest.fixture
def dummy_yfinance_raw():
    cols = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume", "Adj Close", "Dividends", "Stock Splits"], ["AAPL"]],
        names=["Price", "Ticker"]
    )
    idx = pd.DatetimeIndex(["2026-07-01", "2026-07-02"])
    data = [
        [150.0, 155.0, 149.0, 152.0, 1000000, 152.0, 0.0, 0.0],
        [152.0, 153.0, 148.0, 150.0, 1200000, 150.0, 0.5, 4.0] 
    ]
    return pd.DataFrame(data, index=idx, columns=cols)

def test_normalize_yfinance_symbol_maps_schema(dummy_yfinance_raw):
    retrieved_at = pd.Timestamp.now(tz="UTC")
    normalized = normalize_yfinance_symbol(dummy_yfinance_raw, "AAPL", "AAPL", retrieved_at)
    assert set(normalized.columns) == set(CANONICAL_COLUMNS)
    assert normalized.columns.name is None
    assert normalized.index.names == ("date", "asset_id")
    assert str(normalized.index.get_level_values("date").dtype) == "datetime64[ns]"
    assert (normalized["source"] == "yfinance").all()

def test_normalize_yfinance_symbol_preserves_actions(dummy_yfinance_raw):
    retrieved_at = pd.Timestamp.now(tz="UTC")
    normalized = normalize_yfinance_symbol(dummy_yfinance_raw, "AAPL", "AAPL", retrieved_at)
    assert normalized.loc[(pd.Timestamp("2026-07-01"), "AAPL"), "split_factor"] == 1.0
    assert normalized.loc[(pd.Timestamp("2026-07-02"), "AAPL"), "split_factor"] == 4.0
    assert normalized.loc[(pd.Timestamp("2026-07-02"), "AAPL"), "cash_dividend"] == 0.5

def test_normalize_yfinance_symbol_does_not_mutate_raw(dummy_yfinance_raw):
    raw_copy = dummy_yfinance_raw.copy(deep=True)
    retrieved_at = pd.Timestamp.now(tz="UTC")
    _ = normalize_yfinance_symbol(dummy_yfinance_raw, "AAPL", "AAPL", retrieved_at)
    pd.testing.assert_frame_equal(dummy_yfinance_raw, raw_copy)

def test_normalize_yfinance_symbol_retrieved_at_tz_naive_raises(dummy_yfinance_raw):
    naive_ts = pd.Timestamp("2026-07-01 12:00:00")
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_yfinance_symbol(dummy_yfinance_raw, "AAPL", "AAPL", naive_ts)

def test_normalize_yfinance_symbol_retrieved_at_converts_to_utc(dummy_yfinance_raw):
    est_ts = pd.Timestamp("2026-07-01 12:00:00", tz="America/New_York")
    normalized = normalize_yfinance_symbol(dummy_yfinance_raw, "AAPL", "AAPL", est_ts)
    retrieved_col = normalized["retrieved_at"]
    assert str(retrieved_col.dtype) == "datetime64[ns, UTC]"
    assert str(retrieved_col.iloc[0].tzinfo) in ["UTC", "datetime.timezone.utc", "+00:00", "UTC+00:00"] or retrieved_col.iloc[0].tzinfo == pd.Timestamp.utcnow().tzinfo
    
def test_normalize_yfinance_symbol_rejects_missing_field(dummy_yfinance_raw):
    bad_raw = dummy_yfinance_raw.drop(columns="Volume", level="Price")
    retrieved_at = pd.Timestamp.now(tz="UTC")
    with pytest.raises(ValueError, match="Required field 'Volume' is missing"):
        normalize_yfinance_symbol(bad_raw, "AAPL", "AAPL", retrieved_at)

def test_normalize_yfinance_symbol_rejects_wrong_symbol(dummy_yfinance_raw):
    retrieved_at = pd.Timestamp.now(tz="UTC")
    with pytest.raises(ValueError, match="not found in 'Ticker' level"):
        normalize_yfinance_symbol(dummy_yfinance_raw, "AAPL", "WRONG", retrieved_at)


# --- Alpha Vantage Tests ---

@pytest.fixture
def dummy_alpha_vantage_raw():
    data = {
        "timestamp": ["2026-07-02", "2026-07-01"],
        "open": [100.0, 98.0],
        "high": [105.0, 101.0],
        "low": [99.0, 97.0],
        "close": [104.0, 100.0],
        "volume": [123456, 100000]
    }
    return pd.DataFrame(data)

def test_normalize_alpha_vantage_maps_schema(dummy_alpha_vantage_raw):
    retrieved_at = pd.Timestamp.now(tz="UTC")
    normalized = normalize_alpha_vantage_symbol(dummy_alpha_vantage_raw, "AAPL", "AAPL", retrieved_at)
    assert set(normalized.columns) == set(CANONICAL_COLUMNS)
    assert normalized.columns.name is None
    assert normalized.index.names == ("date", "asset_id")
    assert str(normalized.index.get_level_values("date").dtype) == "datetime64[ns]"
    assert (normalized["source"] == "alpha_vantage").all()

def test_normalize_alpha_vantage_sorts_dates(dummy_alpha_vantage_raw):
    retrieved_at = pd.Timestamp.now(tz="UTC")
    normalized = normalize_alpha_vantage_symbol(dummy_alpha_vantage_raw, "AAPL", "AAPL", retrieved_at)
    dates = normalized.index.get_level_values("date")
    assert dates[0] == pd.Timestamp("2026-07-01")
    assert dates[1] == pd.Timestamp("2026-07-02")
    assert dates.is_monotonic_increasing

def test_normalize_alpha_vantage_marks_unavailable_fields(dummy_alpha_vantage_raw):
    retrieved_at = pd.Timestamp.now(tz="UTC")
    normalized = normalize_alpha_vantage_symbol(dummy_alpha_vantage_raw, "AAPL", "AAPL", retrieved_at)
    missing_fields = ["adj_close", "cash_dividend", "split_factor"]
    for field in missing_fields:
        assert str(normalized[field].dtype) == "Float64"
        assert normalized[field].isna().all()

def test_normalize_alpha_vantage_does_not_mutate_raw(dummy_alpha_vantage_raw):
    raw_copy = dummy_alpha_vantage_raw.copy(deep=True)
    retrieved_at = pd.Timestamp.now(tz="UTC")
    _ = normalize_alpha_vantage_symbol(dummy_alpha_vantage_raw, "AAPL", "AAPL", retrieved_at)
    pd.testing.assert_frame_equal(dummy_alpha_vantage_raw, raw_copy)

def test_normalize_alpha_vantage_rejects_missing_field(dummy_alpha_vantage_raw):
    bad_raw = dummy_alpha_vantage_raw.drop(columns="volume")
    retrieved_at = pd.Timestamp.now(tz="UTC")
    with pytest.raises(ValueError, match="Required field 'volume' is missing"):
        normalize_alpha_vantage_symbol(bad_raw, "AAPL", "AAPL", retrieved_at)

def test_normalize_alpha_vantage_rejects_invalid_timestamp(dummy_alpha_vantage_raw):
    bad_raw = dummy_alpha_vantage_raw.copy()
    bad_raw.loc[0, "timestamp"] = "INVALID_DATE"
    retrieved_at = pd.Timestamp.now(tz="UTC")
    with pytest.raises(ValueError):
        normalize_alpha_vantage_symbol(bad_raw, "AAPL", "AAPL", retrieved_at)

def test_normalize_alpha_vantage_rejects_naive_retrieved_at(dummy_alpha_vantage_raw):
    naive_ts = pd.Timestamp("2026-07-01 12:00:00")
    with pytest.raises(ValueError, match="timezone-aware"):
        normalize_alpha_vantage_symbol(dummy_alpha_vantage_raw, "AAPL", "AAPL", naive_ts)
