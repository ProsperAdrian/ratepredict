from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from app.config import Settings
from app.schemas import InferenceSnapshot, MarketBrief, ModelBreakdown, SourceStatus, TopFeature
from app.services.artifacts import ArtifactBundle, ArtifactLoader, ExportLoader
from app.services.features import PublicFeatureBuilder
from app.services.gemini_ai import GeminiAIContextEngine as GeminiBriefService
from app.services.market_data import ExternalDailyMarketDataService, QuidaxMarketSnapshot, QuidaxTickerService


class LiveInferenceService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.artifacts = ArtifactLoader(settings).load()
        self.export_loader = ExportLoader(settings)
        self.external_market_data = ExternalDailyMarketDataService(settings)
        self.quidax_tickers = QuidaxTickerService(settings)
        self.feature_builder = PublicFeatureBuilder()
        self.gemini = GeminiBriefService(settings)
        self._lock = threading.Lock()
        self._latest_snapshot: InferenceSnapshot | None = None
        self._cache_path = self.settings.runtime_dir / "latest_signal.json"

    @property
    def latest_snapshot(self) -> InferenceSnapshot | None:
        return self._latest_snapshot

    def refresh(self) -> InferenceSnapshot:
        with self._lock:
            snapshot = self._run_refresh()
            self._latest_snapshot = snapshot
            self._cache_path.write_text(snapshot.model_dump_json(indent=2))
            return snapshot

    def get_or_refresh(self) -> InferenceSnapshot:
        if self._latest_snapshot is not None:
            return self._latest_snapshot
        if self._cache_path.exists():
            try:
                cached = InferenceSnapshot.model_validate_json(self._cache_path.read_text())
                self._latest_snapshot = cached
                return cached
            except Exception:
                pass
        return self.refresh()

    def _run_refresh(self) -> InferenceSnapshot:
        live_quotes = self.quidax_tickers.fetch()
        latest_path = self.export_loader.latest_export_path()
        export_frame = self.export_loader.load_latest()
        using_runtime_bars = latest_path.name == self.settings.runtime_bars_filename
        if using_runtime_bars:
            runtime_frame = export_frame
            synthetic_bars = 0
        else:
            runtime_frame, synthetic_bars = self._apply_live_quotes(export_frame, live_quotes)
        export_tail = runtime_frame.tail(self.settings.feature_lookback_bars).copy()
        latest_bar_time = export_frame.index.max().to_pydatetime()
        live_bar_time = export_tail.index.max().to_pydatetime()

        start = (export_tail.index.min() - timedelta(days=10)).to_pydatetime()
        end = datetime.now(UTC) + timedelta(days=1)
        market_fetch = self.external_market_data.fetch(start=start, end=end)

        feature_result = self.feature_builder.build(
            export_frame=export_tail,
            external_daily=market_fetch.frame,
            feature_columns=self.artifacts.feature_columns,
        )
        latest_features = feature_result.features.iloc[[-1]]
        transformed = self.artifacts.scaler.transform(latest_features)
        transformed_frame = pd.DataFrame(transformed, index=latest_features.index, columns=self.artifacts.feature_columns)

        xgb_pred = float(self.artifacts.xgb_model.predict(transformed_frame)[0])
        lgbm_pred = float(self.artifacts.lgbm_model.predict(transformed_frame)[0])
        ridge_pred = float(self.artifacts.ridge_model.predict(transformed)[0])
        ensemble_pred = 0.50 * xgb_pred + 0.30 * lgbm_pred + 0.20 * ridge_pred

        signal_anchor_price = float(export_tail["close"].iloc[-1])
        live_last_trade = float(live_quotes.usdtngn.last)
        forecast_price = live_last_trade * (1 + ensemble_pred)

        threshold = float(
            self.settings.price_threshold
            if self.settings.price_threshold is not None
            else self.artifacts.metadata.get("recommended_threshold", 0.0030)
        )
        signal = self._signal_for_prediction(ensemble_pred, threshold)
        confidence_score = self._confidence_score(
            ensemble_pred=ensemble_pred,
            component_predictions=[xgb_pred, lgbm_pred, ridge_pred],
            threshold=threshold,
            holdout_dir_acc=float(self.artifacts.metadata.get("holdout_dir_acc_at_30bps", 60.0)),
            synthetic_bars=synthetic_bars,
        )
        confidence_label = self._confidence_label(confidence_score)
        tradeable = signal != "hold"
        absolute_edge_bps = abs(ensemble_pred) * 10_000

        top_features = self._top_features(latest_features.iloc[0])
        brief = self.gemini.generate(
            signal=signal,
            confidence_label=confidence_label,
            predicted_return_pct=ensemble_pred * 100,
            live_last_trade=live_last_trade,
            forecast_price=forecast_price,
            top_features=[feature.model_dump() for feature in top_features],
            threshold_bps=threshold * 10_000,
        )

        source_statuses = live_quotes.statuses + market_fetch.statuses + [
            SourceStatus(
                source_id="latest_export",
                status="ok",
                latest_timestamp=latest_bar_time,
                message=str(latest_path.name),
            )
        ]
        if synthetic_bars > 0:
            source_statuses.append(
                SourceStatus(
                    source_id="export_gap_fill",
                    status="degraded",
                    latest_timestamp=live_bar_time,
                    message=f"{synthetic_bars} synthetic 2h bars carried forward before live quote overlay",
                )
            )
        live_bar_dt = live_bar_time if live_bar_time.tzinfo else live_bar_time.replace(tzinfo=UTC)
        signal_window_start = live_bar_dt + timedelta(minutes=self.settings.quidax_kline_period_minutes)
        signal_window_end = signal_window_start + timedelta(minutes=self.settings.quidax_kline_period_minutes)
        freshness_anchor = signal_window_start if using_runtime_bars else live_bar_dt
        freshness_minutes = max(0.0, (datetime.now(UTC) - freshness_anchor).total_seconds() / 60.0)

        trade_rationale = self._trade_rationale(
            signal=signal,
            confidence_label=confidence_label,
            absolute_edge_bps=absolute_edge_bps,
            threshold_bps=threshold * 10_000,
            synthetic_bars=synthetic_bars,
        )

        metadata = {
            "training_rows": int(self.artifacts.metadata.get("training_rows", 0)),
            "holdout_dir_acc_pct": float(self.artifacts.metadata.get("holdout_dir_acc", 0.0)),
            "holdout_dir_acc_at_threshold_pct": float(self.artifacts.metadata.get("holdout_dir_acc_at_30bps", 0.0)),
            "holdout_net_pnl_bps": float(self.artifacts.metadata.get("holdout_net_pnl_bps", 0.0)),
            "holdout_trades_at_threshold": int(self.artifacts.metadata.get("holdout_trades_at_30bps", 0)),
            "recommended_threshold_bps": threshold * 10_000,
            "assumed_round_trip_cost_bps": self.settings.assumed_round_trip_cost_bps,
            "live_quote_time": live_quotes.usdtngn.at.isoformat(),
            "live_last_trade": live_quotes.usdtngn.last,
            "signal_anchor_price": signal_anchor_price,
            "synthetic_bars": synthetic_bars,
            "target_type": str(self.artifacts.metadata.get("target_type", "unknown")),
            "signal_window_start": signal_window_start.isoformat(),
            "signal_window_end": signal_window_end.isoformat(),
        }

        return InferenceSnapshot(
            as_of=datetime.now(UTC),
            latest_bar_time=live_bar_time,
            signal_anchor_price=signal_anchor_price,
            live_last_trade=live_last_trade,
            live_bid=float(live_quotes.usdtngn.buy),
            live_ask=float(live_quotes.usdtngn.sell),
            forecast_price=forecast_price,
            predicted_return=ensemble_pred,
            absolute_edge_bps=absolute_edge_bps,
            threshold_bps=threshold * 10_000,
            signal=signal,
            confidence_score=confidence_score,
            confidence_label=confidence_label,
            tradeable=tradeable,
            trade_rationale=trade_rationale,
            data_freshness_minutes=freshness_minutes,
            model_version=str(self.artifacts.metadata.get("model_version", "unknown")),
            model_breakdown=ModelBreakdown(
                xgb=xgb_pred,
                lgbm=lgbm_pred,
                ridge=ridge_pred,
                ensemble=ensemble_pred,
            ),
            top_features=top_features,
            source_statuses=source_statuses,
            market_brief=brief,
            metadata=metadata,
        )

    def _apply_live_quotes(
        self,
        export_frame: pd.DataFrame,
        live_quotes: QuidaxMarketSnapshot,
    ) -> tuple[pd.DataFrame, int]:
        frame = export_frame.copy().sort_index()
        latest_export_time = frame.index.max()
        live_time = pd.Timestamp(live_quotes.usdtngn.at).tz_convert(UTC)
        live_bucket = live_time.floor("2h")
        synthetic_bars = 0

        if live_bucket > latest_export_time:
            missing_index = pd.date_range(
                start=latest_export_time + pd.Timedelta(hours=2),
                end=live_bucket,
                freq="2h",
                tz=UTC,
            )
            if len(missing_index) > 0:
                filler = pd.DataFrame(
                    [frame.iloc[-1].to_dict() for _ in range(len(missing_index))],
                    index=missing_index,
                )
                filler.index.name = frame.index.name
                frame = pd.concat([frame, filler])
                synthetic_bars = len(missing_index)

        target_index = live_bucket if live_bucket in frame.index else frame.index.max()
        current = frame.loc[target_index].copy()
        current["open"] = live_quotes.usdtngn.open
        current["high"] = live_quotes.usdtngn.high
        current["low"] = live_quotes.usdtngn.low
        current["close"] = live_quotes.usdtngn.last
        current["volume"] = live_quotes.usdtngn.vol
        current["btcngn_close"] = live_quotes.btcngn.last
        current["btcngn_volume"] = live_quotes.btcngn.vol
        if live_quotes.usdtngn.last > 0:
            current["implied_btcusd_quidax"] = live_quotes.btcngn.last / live_quotes.usdtngn.last
        frame.loc[target_index, current.index] = current.values
        return frame, synthetic_bars

    def _signal_for_prediction(self, predicted_return: float, threshold: float) -> str:
        if predicted_return >= threshold:
            return "buy_usd"
        if predicted_return <= -threshold:
            return "buy_ngn"
        return "hold"

    def _confidence_score(
        self,
        *,
        ensemble_pred: float,
        component_predictions: list[float],
        threshold: float,
        holdout_dir_acc: float,
        synthetic_bars: int,
    ) -> float:
        disagreement = float(np.std(component_predictions))
        magnitude_score = min(abs(ensemble_pred) / max(threshold, 1e-6), 2.0) / 2.0
        agreement_score = max(0.0, 1.0 - (disagreement / max(abs(ensemble_pred), threshold, 1e-6)))
        historical_score = min(max(holdout_dir_acc / 100.0, 0.0), 1.0)
        staleness_penalty = min(synthetic_bars * 4.0, 30.0)
        score = 100.0 * ((0.45 * magnitude_score) + (0.35 * agreement_score) + (0.20 * historical_score))
        score -= staleness_penalty
        return round(min(max(score, 0.0), 100.0), 1)

    def _confidence_label(self, score: float) -> str:
        if score >= 80:
            return "high"
        if score >= 65:
            return "medium"
        if score >= 50:
            return "guarded"
        return "low"

    def _trade_rationale(
        self,
        *,
        signal: str,
        confidence_label: str,
        absolute_edge_bps: float,
        threshold_bps: float,
        synthetic_bars: int,
    ) -> str:
        staleness_note = (
            f" Runtime used {synthetic_bars} synthetic carry-forward bars because the export lagged the live quote."
            if synthetic_bars > 0
            else ""
        )
        if signal == "hold":
            return (
                f"Model edge is {absolute_edge_bps:.1f} bps, below the {threshold_bps:.1f} bps activation threshold. "
                f"Desk should stay neutral unless non-model information changes the setup.{staleness_note}"
            )
        return (
            f"Model edge is {absolute_edge_bps:.1f} bps versus a {threshold_bps:.1f} bps threshold, "
            f"with {confidence_label} confidence after model-agreement checks.{staleness_note}"
        )

    def _top_features(self, latest_feature_row: pd.Series) -> list[TopFeature]:
        top = self.artifacts.feature_importance.head(8)
        result: list[TopFeature] = []
        for _, row in top.iterrows():
            feature = str(row["feature"])
            result.append(
                TopFeature(
                    name=feature,
                    value=float(latest_feature_row.get(feature, 0.0)),
                    importance=float(row["importance"]),
                )
            )
        return result
