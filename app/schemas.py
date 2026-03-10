from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class SourceStatus(BaseModel):
    source_id: str
    status: str
    latest_timestamp: datetime | None = None
    message: str | None = None


class ModelBreakdown(BaseModel):
    xgb: float
    lgbm: float
    ridge: float
    ensemble: float


class TopFeature(BaseModel):
    name: str
    value: float
    importance: float


class MarketBrief(BaseModel):
    provider: str
    content: str
    generated_at: datetime


class InferenceSnapshot(BaseModel):
    as_of: datetime
    latest_bar_time: datetime
    signal_anchor_price: float
    live_last_trade: float
    live_bid: float
    live_ask: float
    forecast_price: float
    predicted_return: float
    absolute_edge_bps: float
    threshold_bps: float
    signal: str
    confidence_score: float
    confidence_label: str
    tradeable: bool
    trade_rationale: str
    data_freshness_minutes: float
    model_version: str
    model_breakdown: ModelBreakdown
    top_features: list[TopFeature] = Field(default_factory=list)
    source_statuses: list[SourceStatus] = Field(default_factory=list)
    market_brief: MarketBrief
    metadata: dict[str, str | float | int | bool] = Field(default_factory=dict)
