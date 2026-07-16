import pytest
import pandas as pd
from cleanbars.pipeline import build_yfinance_universe
from cleanbars.normalize import CANONICAL_COLUMNS, CANONICAL_DTYPES, INDEX_NAMES

@pytest.fixture
def base_config():
    return {
        "universe": ["AAPL", "MSFT"],
        "vendors": {"yfinance": {"period": "1d", "interval": "1d", "auto_adjust": True, "actions": True, "prepost": False, "repair": False}},
        "symbol_overrides": {"BRK.B": {"yfinance": "BRK-B"}}
    }

def test_build_yfinance_universe_combines_assets(base_config):
    def mock_dl(sym, sett): 
        return pd.DataFrame({"Open": [10.0], "High": [11.0], "Low": [9.0], "Close": [10.0], "Volume": [100], "Adj Close": [10.0], "Dividends": [0.0], "Stock Splits": [0.0]}, index=pd.to_datetime(["2026-07-01"]))
    
    def mock_norm(raw, aid, vsym, ts):
        # Create dict in strict canonical order
        data = {col: pd.Series([1.0], dtype=CANONICAL_DTYPES.get(col, "Float64")) for col in CANONICAL_COLUMNS}
        # Override specific types for test compatibility
        data["vendor_symbol"] = pd.Series(["A"], dtype="string")
        data["source"] = pd.Series(["yfinance"], dtype="string")
        data["retrieved_at"] = pd.Series([pd.Timestamp("2026-07-01", tz="UTC")], dtype="datetime64[ns, UTC]")
        
        idx = pd.MultiIndex.from_arrays([pd.to_datetime(["2026-07-01"]).as_unit("ns"), [aid]], names=INDEX_NAMES)
        return pd.DataFrame(data, index=idx)
    
    panel, raw = build_yfinance_universe(base_config, pd.Timestamp.now(tz="UTC"), mock_dl, mock_norm)
    assert len(panel) == 2
    assert set(panel.index.get_level_values("asset_id")) == {"AAPL", "MSFT"}

def test_build_yfinance_universe_rejects_naive_timestamp(base_config):
    with pytest.raises(ValueError, match="timezone-aware"):
        build_yfinance_universe(base_config, pd.Timestamp("2026-07-01"), None, None)

def test_build_yfinance_universe_reports_failed_asset(base_config):
    def mock_dl(sym, sett): raise Exception("Network fail")
    with pytest.raises(RuntimeError, match="AAPL"):
        build_yfinance_universe(base_config, pd.Timestamp.now(tz="UTC"), mock_dl, None)
