import yfinance as yf
import pandas as pd
from io import StringIO

def download_yfinance_symbol(symbol, settings):
    """
    Downloads raw OHLCV data for a single symbol using yfinance.
    Returns a copy of the unflattened vendor dataframe.
    """
    df = yf.download(
        tickers=symbol,
        period=settings["period"],
        interval=settings["interval"],
        auto_adjust=settings["auto_adjust"],
        actions=settings["actions"],
        prepost=settings["prepost"],
        repair=settings["repair"],
        progress=False,
        threads=False,
    )

    if df.empty:
        raise ValueError(f"yfinance returned no data for symbol: {symbol}")
    
    return df.copy()

def download_alpha_vantage_symbol(symbol, settings, api_key, session):
    """
    Downloads raw OHLCV data for a single symbol using Alpha Vantage.
    Expects settings: function, outputsize, datatype.
    Returns a copy of the raw dataframe.
    """
    # Reject blank API keys
    if not api_key or not str(api_key).strip():
        raise ValueError("Alpha Vantage API key cannot be blank.")
        
    url = "https://www.alphavantage.co/query"
    params = {
        "function": settings["function"],
        "symbol": symbol,
        "outputsize": settings["outputsize"],
        "datatype": settings["datatype"],
        "apikey": api_key
    }
    
    # 30-second finite timeout prevents the pipeline from hanging forever
    response = session.get(url, params=params, timeout=30.0)
    response.raise_for_status()
    
    # Alpha Vantage returns JSON for errors even when `datatype=csv` is requested.
    if response.text.strip().startswith("{"):
        payload = response.json()
        for error_key in ["Error Message", "Note", "Information"]:
            if error_key in payload:
                raise RuntimeError(f"Alpha Vantage API message: {payload[error_key]}")

    if not response.text.strip():
        raise ValueError(f"Alpha Vantage returned no data for symbol: {symbol}")
                
    # Parse the successful CSV response
    df = pd.read_csv(StringIO(response.text))
    
    if df.empty:
        raise ValueError(f"Alpha Vantage returned no data for symbol: {symbol}")
        
    return df.copy()

def resolve_vendor_symbol(asset_id, vendor, symbol_overrides):
    """
    Returns the vendor-specific symbol if mapped, otherwise returns asset_id.
    """
    if not asset_id or not str(asset_id).strip():
        raise ValueError("asset_id cannot be blank.")
    if not vendor or not str(vendor).strip():
        raise ValueError("vendor cannot be blank.")
        
    return symbol_overrides.get(asset_id, {}).get(vendor, asset_id)