from __future__ import annotations

import unittest
from datetime import UTC

import numpy as np
import pandas as pd

from app.config import get_settings
from app.services.artifacts import ArtifactLoader, ExportLoader
from app.services.features import PublicFeatureBuilder
from app.services.inference import LiveInferenceService
from app.services.market_data import MarketFetchResult, QuidaxMarketSnapshot, QuidaxTicker
from app.schemas import SourceStatus
from scripts.refresh_runtime_data import build_runtime_bars


class LiveInferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = get_settings()
        self.artifacts = ArtifactLoader(self.settings).load()
        self.export = ExportLoader(self.settings).load_latest().tail(self.settings.feature_lookback_bars)

    def test_feature_builder_matches_saved_public_feature_set(self) -> None:
        index = pd.date_range(
            start=(self.export.index.min() - pd.Timedelta(days=10)).date(),
            end=(self.export.index.max() + pd.Timedelta(days=1)).date(),
            freq="D",
            tz=UTC,
        )
        base = np.linspace(100, 110, len(index))
        external = pd.DataFrame(
            {
                "brent": base,
                "dxy": base + 10,
                "vix": np.linspace(14, 20, len(index)),
                "usdzar": np.linspace(17, 19, len(index)),
                "usdngn_official": np.linspace(1450, 1500, len(index)),
                "usdghs": np.linspace(13, 14, len(index)),
                "usdkes": np.linspace(129, 132, len(index)),
                "btcusd_global": np.linspace(60000, 90000, len(index)),
            },
            index=index,
        )
        result = PublicFeatureBuilder().build(
            export_frame=self.export,
            external_daily=external,
            feature_columns=self.artifacts.feature_columns,
        )
        self.assertEqual(list(result.features.columns), self.artifacts.feature_columns)
        self.assertFalse(result.features.iloc[-1].isna().any())

    def test_confidence_scoring_stays_bounded(self) -> None:
        service = LiveInferenceService(self.settings)
        score = service._confidence_score(
            ensemble_pred=0.004,
            component_predictions=[0.0042, 0.0039, 0.0038],
            threshold=0.003,
            holdout_dir_acc=64.8,
            synthetic_bars=0,
        )
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 100.0)

    def test_live_quote_overlay_updates_runtime_bar(self) -> None:
        service = LiveInferenceService(self.settings)
        base_frame = self.export.tail(12).copy()
        last_index = base_frame.index.max()
        live_at = (last_index + pd.Timedelta(hours=2)).to_pydatetime()

        snapshot = QuidaxMarketSnapshot(
            usdtngn=QuidaxTicker(
                market="usdtngn",
                at=live_at,
                buy=1410.0,
                sell=1418.0,
                low=1408.0,
                high=1420.0,
                open=1412.0,
                last=1415.0,
                vol=1000.0,
            ),
            btcngn=QuidaxTicker(
                market="btcngn",
                at=live_at,
                buy=99000000.0,
                sell=99500000.0,
                low=98500000.0,
                high=100000000.0,
                open=98900000.0,
                last=99400000.0,
                vol=1.2,
            ),
            statuses=[],
        )

        runtime_frame, synthetic_bars = service._apply_live_quotes(base_frame, snapshot)
        self.assertEqual(synthetic_bars, 1)
        self.assertEqual(float(runtime_frame.iloc[-1]["close"]), 1415.0)
        self.assertEqual(float(runtime_frame.iloc[-1]["btcngn_close"]), 99400000.0)
        self.assertAlmostEqual(float(runtime_frame.iloc[-1]["implied_btcusd_quidax"]), 99400000.0 / 1415.0)

    def test_runtime_bar_builder_merges_usdtngn_and_btcngn(self) -> None:
        index = pd.date_range("2026-03-10 10:00:00+00:00", periods=3, freq="2h")
        usdtngn = pd.DataFrame(
            {
                "open": [1410.0, 1412.0, 1414.0],
                "high": [1415.0, 1416.0, 1418.0],
                "low": [1409.0, 1411.0, 1413.0],
                "close": [1412.0, 1414.0, 1416.0],
                "volume": [100.0, 110.0, 120.0],
            },
            index=index,
        )
        btcngn = pd.DataFrame(
            {
                "open": [99000000.0, 99100000.0, 99200000.0],
                "high": [99300000.0, 99400000.0, 99500000.0],
                "low": [98900000.0, 99000000.0, 99100000.0],
                "close": [99200000.0, 99300000.0, 99400000.0],
                "volume": [1.0, 1.1, 1.2],
            },
            index=index,
        )

        runtime = build_runtime_bars(usdtngn, btcngn)
        self.assertEqual(float(runtime.iloc[-1]["close"]), 1416.0)
        self.assertEqual(float(runtime.iloc[-1]["btcngn_close"]), 99400000.0)
        self.assertAlmostEqual(float(runtime.iloc[-1]["implied_btcusd_quidax"]), 99400000.0 / 1416.0)

    def test_refresh_exposes_live_last_trade_from_quidax_ticker_last(self) -> None:
        service = LiveInferenceService(self.settings)
        service.export_loader.load_latest = lambda: self.export.copy()
        service.export_loader.latest_export_path = lambda: self.settings.artifacts_dir / "bquxjob_46cfbd3b_19cd830c445.csv"

        index = pd.date_range(
            start=(self.export.index.min() - pd.Timedelta(days=10)).date(),
            end=(self.export.index.max() + pd.Timedelta(days=1)).date(),
            freq="D",
            tz=UTC,
        )
        base = np.linspace(100, 110, len(index))
        external = pd.DataFrame(
            {
                "brent": base,
                "dxy": base + 10,
                "vix": np.linspace(14, 20, len(index)),
                "usdzar": np.linspace(17, 19, len(index)),
                "usdngn_official": np.linspace(1450, 1500, len(index)),
                "usdghs": np.linspace(13, 14, len(index)),
                "usdkes": np.linspace(129, 132, len(index)),
                "btcusd_global": np.linspace(60000, 90000, len(index)),
            },
            index=index,
        )
        service.external_market_data.fetch = lambda start, end: MarketFetchResult(
            frame=external,
            statuses=[SourceStatus(source_id="external_daily_test", status="ok")],
        )

        live_at = (self.export.index.max() + pd.Timedelta(hours=2)).to_pydatetime()
        service.quidax_tickers.fetch = lambda: QuidaxMarketSnapshot(
            usdtngn=QuidaxTicker(
                market="usdtngn",
                at=live_at,
                buy=1406.6,
                sell=1415.08,
                low=1404.21,
                high=1433.59,
                open=1412.61,
                last=1415.08,
                vol=38200.6633,
            ),
            btcngn=QuidaxTicker(
                market="btcngn",
                at=live_at,
                buy=98000000.0,
                sell=99000000.0,
                low=97000000.0,
                high=100000000.0,
                open=98500000.0,
                last=98800000.0,
                vol=1.1,
            ),
            statuses=[],
        )

        snapshot = service.refresh()
        self.assertEqual(snapshot.live_last_trade, 1415.08)
        self.assertEqual(snapshot.live_bid, 1406.6)
        self.assertEqual(snapshot.live_ask, 1415.08)


if __name__ == "__main__":
    unittest.main()
