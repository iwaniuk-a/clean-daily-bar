# clean-daily-bar

## Project objective

This project investigates what each row in a daily equity-price download represents before returns, signals, or models are calculated. Daily OHLCV data embed assumptions about market calendars, time zones, corporate-action adjustments, missing sessions, partial trading days, and vendor corrections. The objective is to produce an auditable dataset whose fields, provenance, and limitations are explicit.

## Data and universe

Daily data were collected from yfinance and Alpha Vantage for:

`SPY`, `AAPL`, `MSFT`, `KO`, `NVDA`, `TSLA`, `FIZZ`, `PLTR`, `COIN`, and `BRK.B`.

yfinance provides the full available history, adjusted close, dividends, and split records. `auto_adjust=False` prevents yfinance from applying dividend-based adjustment to OHLC, although Yahoo’s historical OHLC may already be split-adjusted. Therefore, `*_raw` means unchanged by this project, not necessarily an as-traded historical quotation.

The free Alpha Vantage `TIME_SERIES_DAILY` endpoint supplies 100 recent raw OHLCV observations per asset. Adjusted close, dividend, and split fields unavailable from this endpoint are stored as `pd.NA`.

`XNYS` is used as the canonical exchange calendar. Automated gap auditing begins on `1970-01-01` because earlier holiday definitions produced false positives.

## Key findings

- Ten securities were normalized into one canonical `(date, asset_id)` schema
- All automated tests pass
- Neither vendor panel contained structural OHLCV violations
- No missing or unexpected sessions remained within the post-1970 audit period
- Ten potentially incomplete yfinance bars from `2026-07-17` were excluded from processed data
- Nineteen yfinance price moves exceeded the 25% diagnostic threshold and were retained for investigation
- Closing prices reconciled closely across vendors
- Volume differences were larger and clustered on particular dates

## Reproduction

```bash
python3 -m pytest tests/ -q

PYTHONPATH=src python3 scripts/build_validation_artifacts.py

python3 -m jupyter nbconvert \
  --to notebook \
  --execute notebooks/01_data_audit.ipynb \
  --inplace \
  --ExecutePreprocessor.timeout=120
```

The scripts use cached interim data and do not automatically redownload vendor data.

## Limitations
- Alpha Vantage’s free endpoint cannot validate the full yfinance history
- Vendor volume aggregation and historical revisions may differ
- Pre-1970 observations are outside the automated calendar-gap audit
- `asset_id` is project-controlled rather than an institutional permanent identifier
- Daily OHLCV cannot reconstruct spreads, depth, venue, intraday paths, or achievable execution prices
- Future downloads may differ because vendors can revise historical records

Silent historical vendor revisions are the most important reproducibility limitation