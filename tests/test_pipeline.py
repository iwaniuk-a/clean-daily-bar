import pytest
import pandas as pd
import copy
import inspect
from cleanbars.pipeline import build_yfinance_universe
from cleanbars.normalize import CANONICAL_COLUMNS, CANONICAL_DTYPES, INDEX_NAMES, validate_panel

@pytest.fixture
def base_config():
    return {
        "universe": ["AAPL", "MSFT", "BRK.B"],
        "vendors": {
            "yfinance": {
                "period": "1d",
                "interval": "1d",
                "auto_adjust": True,
                "actions": True,
                "prepost": False,
                "repair": False
            }
        },
        "symbol_overrides": {"BRK.B": {"yfinance": "BRK-B"}}
    }

def _valid_mock_norm(raw, aid, vsym, ts):
    data = {col: pd.Series([1.0], dtype=CANONICAL_DTYPES.get(col, "Float64")) for col in CANONICAL_COLUMNS}
    data["vendor_symbol"] = pd.Series([vsym], dtype="string")
    data["source"] = pd.Series(["yfinance"], dtype="string")
    data["retrieved_at"] = pd.Series([pd.Timestamp("2026-07-01", tz="UTC")], dtype="datetime64[ns, UTC]")

    day = len(aid)
    dt = pd.to_datetime([f"2026-07-0{day}"]).as_unit("ns")
    if aid == "MSFT":
        dt = pd.to_datetime(["2026-07-08"]).as_unit("ns")

    idx = pd.MultiIndex.from_arrays([dt, [aid]], names=INDEX_NAMES)
    return pd.DataFrame(data, index=idx)

def test_build_yfinance_universe_combines_assets(base_config):
    def mock_dl(sym, sett): return pd.DataFrame({"Dummy": [1]})

    panel, raw = build_yfinance_universe(base_config, pd.Timestamp.now(tz="UTC"), mock_dl, _valid_mock_norm)

    assert len(panel) == 3
    assert set(panel.index.get_level_values("asset_id")) == {"AAPL", "MSFT", "BRK.B"}

    validate_panel(panel)
    assert panel.index.is_unique
    assert panel.index.is_monotonic_increasing

def test_build_yfinance_universe_uses_symbol_override(base_config):
    requested_symbols = []
    def mock_dl(sym, sett):
        requested_symbols.append(sym)
        return pd.DataFrame()

    panel, _ = build_yfinance_universe(base_config, pd.Timestamp.now(tz="UTC"), mock_dl, _valid_mock_norm)

    assert "BRK-B" in requested_symbols
    assert "BRK.B" not in requested_symbols
    assert "BRK.B" in panel.index.get_level_values("asset_id")
    assert "BRK-B" not in panel.index.get_level_values("asset_id")

def test_build_yfinance_universe_uses_shared_retrieval_timestamp(base_config):
    received_timestamps = []

    def mock_dl(sym, sett): return pd.DataFrame()
    def mock_norm(raw, aid, vsym, ts):
        received_timestamps.append(ts)
        return _valid_mock_norm(raw, aid, vsym, ts)

    non_utc_ts = pd.Timestamp("2026-07-17 10:00:00", tz="America/New_York")
    build_yfinance_universe(base_config, non_utc_ts, mock_dl, mock_norm)

    assert all(isinstance(ts, pd.Timestamp) for ts in received_timestamps)
    assert all(str(ts.tzinfo) in ["UTC", "datetime.timezone.utc", "+00:00", "UTC+00:00"] or ts.tzinfo == pd.Timestamp.utcnow().tzinfo for ts in received_timestamps)
    assert len(set(received_timestamps)) == 1
    assert received_timestamps[0] == non_utc_ts.tz_convert("UTC")

def test_build_yfinance_universe_rejects_missing_symbol_overrides(base_config):
    del base_config["symbol_overrides"]

    def mock_dl(sym, sett): raise AssertionError("Should not be called")

    with pytest.raises(KeyError):
        build_yfinance_universe(base_config, pd.Timestamp.now(tz="UTC"), mock_dl, _valid_mock_norm)

def test_build_yfinance_universe_returns_raw_frames(base_config):
    dummy_dfs = {
        "AAPL": pd.DataFrame({"A": [1]}),
        "MSFT": pd.DataFrame({"M": [2]}),
        "BRK.B": pd.DataFrame({"B": [3]})
    }
    def mock_dl(sym, sett):
        canonical = "BRK.B" if sym == "BRK-B" else sym
        return dummy_dfs[canonical]

    _, raw_dict = build_yfinance_universe(base_config, pd.Timestamp.now(tz="UTC"), mock_dl, _valid_mock_norm)

    assert set(raw_dict.keys()) == {"AAPL", "MSFT", "BRK.B"}
    assert id(raw_dict["AAPL"]) == id(dummy_dfs["AAPL"])

def test_build_yfinance_universe_rejects_empty_universe(base_config):
    base_config["universe"] = []
    def mock_dl(sym, sett): raise AssertionError("Should not be called")
    with pytest.raises(ValueError, match="empty"):
        build_yfinance_universe(base_config, pd.Timestamp.now(tz="UTC"), mock_dl, _valid_mock_norm)

def test_build_yfinance_universe_rejects_duplicate_assets(base_config):
    base_config["universe"] = ["AAPL", "AAPL"]
    dl_calls = 0
    def mock_dl(sym, sett):
        nonlocal dl_calls
        dl_calls += 1
        return pd.DataFrame()

    with pytest.raises(ValueError, match="duplicate"):
        build_yfinance_universe(base_config, pd.Timestamp.now(tz="UTC"), mock_dl, _valid_mock_norm)

    assert dl_calls == 0

def test_build_yfinance_universe_does_not_mutate_config(base_config):
    original_config = copy.deepcopy(base_config)
    def mock_dl(sym, sett): return pd.DataFrame()
    build_yfinance_universe(base_config, pd.Timestamp.now(tz="UTC"), mock_dl, _valid_mock_norm)
    assert base_config == original_config

def test_build_yfinance_universe_rejects_naive_timestamp(base_config):
    with pytest.raises(ValueError, match="timezone-aware"):
        build_yfinance_universe(base_config, pd.Timestamp("2026-07-01"), None, None)

def test_build_yfinance_universe_reports_failed_asset(base_config):
    def mock_dl(sym, sett): raise Exception("Network fail")
    with pytest.raises(RuntimeError, match="AAPL"):
        build_yfinance_universe(base_config, pd.Timestamp.now(tz="UTC"), mock_dl, None)

def test_build_yfinance_universe_has_production_defaults():
    sig = inspect.signature(build_yfinance_universe)
    assert sig.parameters["downloader"].default is not inspect.Parameter.empty
    assert sig.parameters["normalizer"].default is not inspect.Parameter.empty
