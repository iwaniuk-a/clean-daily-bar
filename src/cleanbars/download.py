import yfinance as yf


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
