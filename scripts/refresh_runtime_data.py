from __future__ import annotations

import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.config import get_settings
from app.services.market_data import ExternalDailyMarketDataService, QuidaxKlineService


def build_runtime_bars(usdtngn: pd.DataFrame, btcngn: pd.DataFrame) -> pd.DataFrame:
    frame = usdtngn.rename(
        columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
    ).join(
        btcngn.rename(
            columns={
                "close": "btcngn_close",
                "volume": "btcngn_volume",
            }
        )[["btcngn_close", "btcngn_volume"]],
        how="left",
    )
    frame["implied_btcusd_quidax"] = frame["btcngn_close"] / frame["close"]

    # Columns preserved for compatibility with the historical export schema.
    optional_zero_cols = [
        "trade_count",
        "buy_volume",
        "sell_volume",
        "buy_count",
        "sell_count",
        "avg_trade_size",
        "max_trade_size",
        "stddev_trade_size",
        "large_trade_count",
        "large_trade_volume",
        "intrabar_price_stddev",
        "intrabar_range",
        "unique_buyers",
        "unique_sellers",
        "btcngn_trade_count",
    ]
    for col in optional_zero_cols:
        frame[col] = 0.0

    ordered_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "trade_count",
        "buy_volume",
        "sell_volume",
        "buy_count",
        "sell_count",
        "avg_trade_size",
        "max_trade_size",
        "stddev_trade_size",
        "large_trade_count",
        "large_trade_volume",
        "intrabar_price_stddev",
        "intrabar_range",
        "unique_buyers",
        "unique_sellers",
        "btcngn_close",
        "btcngn_volume",
        "btcngn_trade_count",
        "implied_btcusd_quidax",
    ]
    return frame[ordered_cols]


def drop_open_bar(frame: pd.DataFrame, *, period_minutes: int) -> pd.DataFrame:
    now = datetime.now(UTC)
    current_bucket_start = pd.Timestamp(now).floor(f"{period_minutes}min")
    return frame.loc[frame.index < current_bucket_start].copy()


def main() -> int:
    settings = get_settings()
    kline_service = QuidaxKlineService(settings)
    external_service = ExternalDailyMarketDataService(settings)

    usdtngn = kline_service.fetch("usdtngn").frame
    btcngn = kline_service.fetch("btcngn").frame

    runtime_bars = build_runtime_bars(usdtngn, btcngn)
    runtime_bars = drop_open_bar(runtime_bars, period_minutes=settings.quidax_kline_period_minutes)
    if runtime_bars.empty:
        raise RuntimeError("No closed 2-hour Quidax bars were available after dropping the open candle.")

    runtime_path = settings.data_dir / settings.runtime_bars_filename
    runtime_bars.reset_index().to_csv(runtime_path, index=False)

    external_start = (runtime_bars.index.min() - timedelta(days=45)).to_pydatetime()
    external_end = datetime.now(UTC) + timedelta(days=1)
    external_fetch = external_service.fetch_live(start=external_start, end=external_end)
    external_service.write_cache(external_fetch.frame)

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "runtime_bars_path": str(runtime_path),
        "runtime_bar_rows": len(runtime_bars),
        "runtime_bar_start": runtime_bars.index.min().isoformat(),
        "runtime_bar_end": runtime_bars.index.max().isoformat(),
        "external_daily_path": str(settings.data_dir / settings.external_daily_filename),
        "external_daily_rows": len(external_fetch.frame),
        "external_daily_end": external_fetch.frame.index.max().isoformat(),
        "sources": [status.model_dump(mode="json") for status in external_fetch.statuses],
    }
    (settings.runtime_dir / "refresh_runtime_data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
