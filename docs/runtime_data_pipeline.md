# Runtime Data Pipeline

## Outputs

- `data/latest/quidax_runtime_2h.csv`
  - Built from Quidax public 2-hour k-line data for `usdtngn` and `btcngn`.
  - Drops the currently open 2-hour candle so inference stays aligned with training on closed bars.
  - Supplies the runtime columns needed by the final model: `open`, `high`, `low`, `close`, `volume`, `btcngn_close`, `btcngn_volume`, `implied_btcusd_quidax`.

- `data/latest/external_daily.csv`
  - Cached daily exogenous features for inference.
  - Runtime prefers this cache and stops calling Yahoo live when the cache exists.

- `runtime/refresh_runtime_data_summary.json`
  - Refresh summary for monitoring and incident review.

## Refresh Command

```bash
. .venv/bin/activate
python scripts/refresh_runtime_data.py
```

## Scheduling

- Recommended cadence: every 2 hours.
- The script can be scheduled by cron, launchd, ECS, or Codex automation.
- The runtime app automatically prefers `quidax_runtime_2h.csv` over the older manual export files.

## Current State

- Quidax market-bar freshness is solved via public k-lines.
- Runtime synthetic carry-forward bars are no longer required once the scheduled bar file is present.
- External daily features are cache-driven at inference time.

## Remaining Source Decision

The refresh script still populates `external_daily.csv` from the existing live provider layer. Today that default provider is Yahoo-backed because the exact production replacements for:

- `brent`
- `dxy`
- `vix`
- `usdngn_official`
- `usdzar`
- `usdghs`
- `usdkes`
- `btcusd_global`

have not yet been locked.

That is the only remaining blocker to being fully off Yahoo without changing feature semantics. If those production-approved source endpoints are supplied, the cache pipeline is already in place and the runtime app will consume the staged file without additional architectural changes.
