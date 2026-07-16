import pandas as pd
import pytest
import yfinance as yf
from cleanbars.download import download_yfinance_symbol

def test_download_yfinance_symbol_passes_settings(monkeypatch):
    captured_kwargs = {}

    # Define a fake download function that intercepts the arguments
    def mock_download(**kwargs):
        captured_kwargs.update(kwargs)
        # Return a minimal non-empty dataframe so it survives the empty check
        return pd.DataFrame({"Close": [150.0]})

    # Replace the real yf.download with our fake one
    monkeypatch.setattr(yf, "download", mock_download)

    # The exact settings we expect to be passed through
    test_settings = {
        "period": "5y",
        "interval": "1d",
        "auto_adjust": True,
        "actions": True,
        "prepost": True,
        "repair": True,
    }

    # Run the function
    result = download_yfinance_symbol("AAPL", test_settings)

    # 1. Assert it returned the data from our mock
    assert not result.empty

    # 2. Assert every parameter was routed perfectly
    assert captured_kwargs["tickers"] == "AAPL"
    assert captured_kwargs["period"] == "5y"
    assert captured_kwargs["interval"] == "1d"
    assert captured_kwargs["auto_adjust"] is True
    assert captured_kwargs["actions"] is True
    assert captured_kwargs["prepost"] is True
    assert captured_kwargs["repair"] is True
    
    # 3. Assert the hardcoded deterministic overrides were applied
    assert captured_kwargs["progress"] is False
    assert captured_kwargs["threads"] is False

def test_download_yfinance_symbol_rejects_empty(monkeypatch):
    # Fake download that returns absolutely nothing
    def mock_download(**kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(yf, "download", mock_download)

    # It must raise a ValueError if the dataframe is empty
    with pytest.raises(ValueError, match="no data"):
        download_yfinance_symbol("INVALID_TICKER", {
            "period": "max",
            "interval": "1d",
            "auto_adjust": False,
            "actions": True,
            "prepost": False,
            "repair": False,
        })

def test_download_yfinance_symbol_rejects_missing_setting(monkeypatch):
    # Omit one critical field ("repair") from the dictionary
    incomplete_settings = {
        "period": "max",
        "interval": "1d",
        "auto_adjust": False,
        "actions": True,
        "prepost": False,
        # "repair": False, <-- Missing
    }

    # Verify that omitting a setting triggers a KeyError
    with pytest.raises(KeyError):
        download_yfinance_symbol("AAPL", incomplete_settings)