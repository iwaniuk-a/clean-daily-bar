## Canonical index

The normalized dataset uses a two-level pandas MultiIndex:

1. `date`
2. `asset_id`

### Index rules

- `date` represents the U.S. regular trading-session date.
- `date` is stored as a timezone-naive calendar date after normalization.
- `asset_id` is the project-controlled canonical identifier.
- Vendor-specific ticker symbols are stored separately and are not used as the primary key.
- Every `(date, asset_id)` pair must be unique.
- The index must be sorted first by `date`, then by `asset_id`.
- Missing trading sessions are not inserted or forward-filled automatically.

## Canonical columns

Each vendor is normalized independently into the following schema.

| Column | pandas dtype | Meaning |
|---|---|---|
| `vendor_symbol` | `string` | Symbol submitted to or returned by the vendor. |
| `open_raw` | `Float64` | First qualifying transaction price for the session, before back-adjustment. |
| `high_raw` | `Float64` | Highest qualifying transaction price for the session, before back-adjustment. |
| `low_raw` | `Float64` | Lowest qualifying transaction price for the session, before back-adjustment. |
| `close_raw` | `Float64` | Final or official session price, before back-adjustment. |
| `volume_raw` | `Int64` | Vendor-reported number of shares traded. |
| `adj_close` | `Float64` | Vendor-adjusted closing price, when available. |
| `cash_dividend` | `Float64` | Cash dividend effective on the session date, when supplied by the vendor. |
| `split_factor` | `Float64` | Stock-split ratio effective on the session date, when supplied by the vendor. |
| `source` | `string` | Vendor identifier, such as `yfinance` or `alpha_vantage`. |
| `retrieved_at` | `datetime64[ns, UTC]` | Timestamp at which the vendor response was retrieved. |

### Vendor price-basis conventions

- `*_raw` denotes the vendor-supplied OHLC field before transformations performed by this project.
- For yfinance/Yahoo, `auto_adjust=False` prevents dividend-based OHLC adjustment by yfinance, but the supplied historical OHLC series is split-adjusted.
- Consequently, old yfinance OHLC values are not necessarily prices that were historically quoted at the exchange.
- `adj_close` adds the vendor's broader adjustment convention, including dividend effects where applicable.
- Alpha Vantage raw OHLCV is retained under its own vendor convention and compared without assuming either vendor is authoritative.
- Execution-price research must not automatically treat historical yfinance OHLC as literal as-traded quotations.

### Missing-value rules

- Use pandas nullable dtypes: `Float64`, `Int64`, `string`, and `boolean` where applicable.
- A field unavailable from a vendor is stored as `pd.NA`, not as zero.
- A genuine zero reported by a vendor remains zero.
- Raw price or volume values are never forward-filled during normalization.
- Alpha Vantage raw daily data will have missing `adj_close`, `cash_dividend`, and `split_factor`.
- Cross-vendor tables are created by joining vendor tables side-by-side on `(date, asset_id)`, with vendor-specific column suffixes.
## Persistence contract

Each vendor-normalized table is saved independently in Parquet format:

- `data/interim/yfinance_daily.parquet`
- `data/interim/alpha_vantage_daily.parquet`

### Storage rules

- Save the `(date, asset_id)` MultiIndex in the Parquet file.
- Use the `pyarrow` engine.
- Use `zstd` compression.
- Do not save the DataFrame with `index=False`.
- Sort the index immediately before writing.
- Validate index uniqueness immediately before writing.
- Reload the saved file once and verify that its index names, dtypes, row count, and column order match the in-memory table.
- Never overwrite files in `data/raw/` during normalization.
- The final cross-vendor comparison table will be saved separately under `data/processed/`.

### Corporate-action encoding

- `cash_dividend` is `0.0` when the vendor reports no cash dividend on that date.
- `split_factor` is `1.0` when no split occurs.
- For yfinance, the vendor's `Stock Splits == 0.0` value is normalized to `1.0`.
- A reported split ratio, such as `4.0` for a four-for-one split, is preserved.
- An action field unavailable from a vendor remains `pd.NA`.

### Calendar reliability boundary

- The canonical session calendar is `XNYS` from `exchange-calendars`.
- Calendar-gap validation is scoped to dates on or after `1970-01-01`.
- The cutoff is a project-specific reliability boundary, not a claim that earlier price observations are invalid.
- Before 1970, the current calendar implementation classified documented holiday-like dates as expected sessions, producing false missing-row flags for KO.
- Pre-1970 observations are preserved but reported as outside the automated calendar-audit scope.
