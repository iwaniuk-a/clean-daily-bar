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