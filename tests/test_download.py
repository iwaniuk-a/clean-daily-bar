import pandas as pd
import pytest
import yfinance as yf
from unittest.mock import MagicMock
from cleanbars.download import download_yfinance_symbol, download_alpha_vantage_symbol

# --- yfinance tests ---

def test_download_yfinance_symbol_passes_settings(monkeypatch):
    captured_kwargs = {}
    def mock_download(**kwargs):
        captured_kwargs.update(kwargs)
        return pd.DataFrame({"Close": [150.0]})
    monkeypatch.setattr(yf, "download", mock_download)

    test_settings = {
        "period": "5y",
        "interval": "1d",
        "auto_adjust": True,
        "actions": True,
        "prepost": True,
        "repair": True,
    }
    result = download_yfinance_symbol("AAPL", test_settings)
    assert not result.empty
    assert captured_kwargs["tickers"] == "AAPL"
    assert captured_kwargs["period"] == "5y"
    assert captured_kwargs["interval"] == "1d"

def test_download_yfinance_symbol_rejects_empty(monkeypatch):
    def mock_download(**kwargs):
        return pd.DataFrame()
    monkeypatch.setattr(yf, "download", mock_download)

    with pytest.raises(ValueError, match="no data"):
        download_yfinance_symbol("INVALID_TICKER", {
            "period": "max", "interval": "1d", "auto_adjust": False, 
            "actions": True, "prepost": False, "repair": False
        })

def test_download_yfinance_symbol_rejects_missing_setting(monkeypatch):
    incomplete_settings = {
        "period": "max", "interval": "1d", "auto_adjust": False, 
        "actions": True, "prepost": False
    }
    with pytest.raises(KeyError):
        download_yfinance_symbol("AAPL", incomplete_settings)

# --- Alpha Vantage tests ---

@pytest.fixture
def mock_session():
    return MagicMock()

def test_download_alpha_vantage_passes_parameters(mock_session):
    settings = {"function": "TIME_SERIES_DAILY", "outputsize": "compact", "datatype": "csv"}
    
    # Provide a minimal valid CSV so parsing doesn't crash after the request
    mock_session.get.return_value.text = "timestamp,open,high,low,close,volume\n2026-07-15,100.0,105.0,99.0,104.0,123456"
    
    download_alpha_vantage_symbol("AAPL", settings, "SECRET_KEY", mock_session)
    
    args, kwargs = mock_session.get.call_args
    assert kwargs["params"]["symbol"] == "AAPL"
    assert kwargs["params"]["apikey"] == "SECRET_KEY"
    assert kwargs["timeout"] == 30.0

def test_download_alpha_vantage_parses_csv(mock_session):
    csv_data = "timestamp,open,high,low,close,volume\n2026-07-15,100.0,105.0,99.0,104.0,123456\n2026-07-14,98.0,101.0,97.0,100.0,100000"
    mock_session.get.return_value.text = csv_data
    mock_session.get.return_value.status_code = 200
    
    settings = {"function": "TIME_SERIES_DAILY", "outputsize": "compact", "datatype": "csv"}
    df = download_alpha_vantage_symbol("AAPL", settings, "KEY", mock_session)
    
    assert df.shape == (2, 6)
    assert "timestamp" in df.columns
    assert df["close"].iloc[0] == 104.0

def test_download_alpha_vantage_rejects_api_error(mock_session):
    # Simulate an error response from Alpha Vantage
    mock_session.get.return_value.text = '{"Error Message": "Invalid API call"}'
    mock_session.get.return_value.json.return_value = {"Error Message": "Invalid API call"}
    
    # Provide valid settings so it doesn't fail the dictionary lookup beforehand
    settings = {"function": "TIME_SERIES_DAILY", "outputsize": "compact", "datatype": "csv"}
    
    with pytest.raises(RuntimeError, match="Invalid API call"):
        download_alpha_vantage_symbol("AAPL", settings, "KEY", mock_session)

def test_download_alpha_vantage_rejects_blank_key(mock_session):
    with pytest.raises(ValueError, match="API key cannot be blank"):
        # Key check happens before settings, so {} is fine here
        download_alpha_vantage_symbol("AAPL", {}, " ", mock_session)

def test_download_alpha_vantage_rejects_missing_setting(mock_session):
    with pytest.raises(KeyError):
        download_alpha_vantage_symbol("AAPL", {"function": "A", "outputsize": "B"}, "KEY", mock_session)

def test_download_alpha_vantage_rejects_empty_response(mock_session):
    # Simulate an empty response body
    mock_session.get.return_value.text = "   "
    settings = {"function": "TIME_SERIES_DAILY", "outputsize": "compact", "datatype": "csv"}
    
    with pytest.raises(ValueError, match="no data"):
        download_alpha_vantage_symbol("AAPL", settings, "KEY", mock_session)
