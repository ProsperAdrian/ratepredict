from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Literal

TierClass = Literal["verified", "monitored"]
VerificationStatus = Literal["verified", "monitored", "stale", "manual_review", "failed"]


class OperatingMode(str, Enum):
    """System operating modes derived from current source health."""

    FULL = "A"
    EXTERNAL_DEGRADED = "B"
    INTERNAL_ONLY = "C"
    DEAD = "D"


class Signal(str, Enum):
    """Desk action vocabulary."""

    BUY_USD = "buy_usd"
    BUY_NGN = "buy_ngn"
    HOLD = "hold"


@dataclass(frozen=True)
class SourceHealthV1:
    source_id: str
    tier_class: TierClass
    freshness_sec: int
    integrity_score: float
    verification_status: VerificationStatus
    quality_penalty: float


@dataclass(frozen=True)
class FeatureVectorV3:
    as_of_time: datetime
    mode: OperatingMode
    proprietary_features: dict[str, float]
    public_features: dict[str, float]
    missing_flags: list[str] = field(default_factory=list)
    effective_penalty: float = 0.0


@dataclass(frozen=True)
class PredictionOutputV3:
    forecast_return_2h: float
    confidence_raw: float
    confidence_adjusted: float
    mode: OperatingMode
    external_anchor_present: bool
    model_breakdown: dict[str, float]


@dataclass(frozen=True)
class DecisionRecommendationV3:
    signal: Signal
    target_inventory_split: dict[str, float]
    max_delta: float
    risk_cap: float
    requires_human_approval: bool
    constraint_reason_codes: list[str]


@dataclass(frozen=True)
class AblationReportV1:
    window: str
    baseline_model_metrics: dict[str, float]
    enhanced_model_metrics: dict[str, float]
    feature_family: str
    relative_lift_pct: float
    keep_feature_family: bool


@dataclass(frozen=True)
class EconomicsReportV1:
    baseline_net_pnl: float
    model_net_pnl: float
    incremental_alpha_bps: float
    cost_breakdown: dict[str, float]
    continuation_gate_status: bool
