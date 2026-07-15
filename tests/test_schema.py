import pandas as pd
import pytest

from cleanbars.normalize import validate_index
from cleanbars.normalize import CANONICAL_COLUMNS, CANONICAL_DTYPES, INDEX_NAMES
from cleanbars.normalize import validate_columns

def test_schema_constants():
    assert isinstance(CANONICAL_COLUMNS, tuple)
    assert len(CANONICAL_COLUMNS) == len(set(CANONICAL_COLUMNS))
    assert tuple(CANONICAL_DTYPES) == CANONICAL_COLUMNS

def test_canonical_dtypes():
    expected = {
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

    assert CANONICAL_DTYPES == expected

def test_index_names():
    assert isinstance(INDEX_NAMES, tuple)
    assert INDEX_NAMES == ("date", "asset_id")
    assert len(INDEX_NAMES) == len(set(INDEX_NAMES))

def test_validate_index_panel():
    index = pd.MultiIndex.from_tuples([("2026-07-01", "AAPL"), ("2026-07-02", "AAPL")],
        names=("date", "asset_id"))
    panel = pd.DataFrame({"close_raw": [150.5, 151.0]}, index=index)
    result = validate_index(panel)
    assert result is None

def test_duplicate_index_panel():
    index = pd.MultiIndex.from_tuples([("2026-07-01", "AAPL"), ("2026-07-01", "AAPL")],
        names=("date", "asset_id"))
    panel = pd.DataFrame({"close_raw": [150.5, 151.0]}, index=index)

    with pytest.raises(ValueError, match="unique"):
        validate_index(panel)

def test_names_index_panel():
    index = pd.MultiIndex.from_tuples([("2026-07-01", "AAPL"), ("2026-07-02", "AAPL")],
        names=("timestamp", "ticker"))
    panel = pd.DataFrame({"close_raw": [150.5, 151.0]}, index=index)
    with pytest.raises(ValueError, match="names"):
        validate_index(panel)

def test_validate_dates():
    index = pd.MultiIndex.from_tuples([("2026-07-02", "AAPL"),("2026-07-01", "AAPL")],
        names=("date", "asset_id"))
    panel = pd.DataFrame({"close_raw": [150.5, 151.0]}, index=index)
    with pytest.raises(ValueError, match="monotonically"):
        validate_index(panel)

def test_validate_multiindex():
    panel = pd.DataFrame({"close_raw": [150.5, 151.0]})
    with pytest.raises(ValueError, match="MultiIndex"):
        validate_index(panel)

####

def test_validate_columns_valid():
    panel = pd.DataFrame(columns=CANONICAL_COLUMNS)
    result = validate_columns(panel)
    assert result is None

def test_validate_columns_missing():
    panel = pd.DataFrame(columns=CANONICAL_COLUMNS[:-1])

    with pytest.raises(ValueError, match="Missing"):
        validate_columns(panel)

def tst_validate_columns_unexpected():
    panel = pd.DataFrame(columns=(*CANONICAL_COLUMNS, "unexpected_field"))
    with pytest.raises(ValueError, match="unexpected"):
        validate_columns(panel)

def test_validate_columns_order():
    panel = pd.DataFrame(columns=tuple(reversed(CANONICAL_COLUMNS)))
    with pytest.raises(ValueError, match="order"):
        validate_columns(panel)


