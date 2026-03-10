from __future__ import annotations

from dataclasses import dataclass
from math import inf
from statistics import mean
from typing import Iterable

from .types import (
    AblationReportV1,
    DecisionRecommendationV3,
    EconomicsReportV1,
    OperatingMode,
    PredictionOutputV3,
    Signal,
    SourceHealthV1,
)

INTERNAL_SOURCE_ID = "quidax_internal"
VERIFIED_EXTERNAL_SOURCE_IDS = {"binance_p2p", "yahoo_finance", "approved_institutional_feed"}
MONITORED_EXTERNAL_SOURCE_IDS = {"abokifx", "fmdq", "cbn_pages"}


@dataclass(frozen=True)
class ProductionThresholds:
    incremental_alpha_hurdle_bps: float = 15.0
    integrity_floor: float = 0.70
    quality_penalty_ceiling: float = 0.75
    degraded_mode_penalty: float = 0.10
    decision_threshold_full: float = 0.0015
    decision_threshold_external_degraded: float = 0.0020
    decision_threshold_internal_only: float = inf
    decision_threshold_dead: float = inf
    confidence_floor_full: float = 60.0
    confidence_floor_external_degraded: float = 65.0
    confidence_floor_internal_only: float = 101.0
    risk_cap_full: float = 0.20
    risk_cap_external_degraded: float = 0.07
    risk_cap_internal_only: float = 0.0
    risk_cap_dead: float = 0.0
    max_delta_full: float = 0.15
    max_delta_external_degraded: float = 0.07
    max_delta_internal_only: float = 0.05
    max_delta_dead: float = 0.0


DEFAULT_THRESHOLDS = ProductionThresholds()


def _freshness_limit(source: SourceHealthV1) -> int:
    if source.source_id == INTERNAL_SOURCE_ID:
        return 2 * 60 * 60
    if source.tier_class == "verified":
        return 4 * 60 * 60
    if source.source_id == "cbn_pages":
        return 7 * 24 * 60 * 60
    if source.source_id == "fmdq":
        return 24 * 60 * 60
    return 8 * 60 * 60


def _is_healthy(
    source: SourceHealthV1,
    thresholds: ProductionThresholds = DEFAULT_THRESHOLDS,
) -> bool:
    return (
        source.verification_status not in {"stale", "failed"}
        and source.integrity_score >= thresholds.integrity_floor
        and source.quality_penalty < thresholds.quality_penalty_ceiling
        and source.freshness_sec <= _freshness_limit(source)
    )


def determine_operating_mode(
    sources: Iterable[SourceHealthV1],
    thresholds: ProductionThresholds = DEFAULT_THRESHOLDS,
) -> tuple[OperatingMode, bool]:
    """Return the current operating mode and whether any usable external anchor exists."""

    sources = list(sources)
    healthy_internal = any(
        source.source_id == INTERNAL_SOURCE_ID and _is_healthy(source, thresholds)
        for source in sources
    )
    healthy_verified_external = any(
        source.source_id in VERIFIED_EXTERNAL_SOURCE_IDS and _is_healthy(source, thresholds)
        for source in sources
    )
    healthy_monitored_external = any(
        source.source_id in MONITORED_EXTERNAL_SOURCE_IDS and _is_healthy(source, thresholds)
        for source in sources
    )
    external_anchor_present = healthy_verified_external or healthy_monitored_external

    if healthy_verified_external and healthy_monitored_external and healthy_internal:
        return OperatingMode.FULL, True
    if external_anchor_present:
        return OperatingMode.EXTERNAL_DEGRADED, True
    if healthy_internal:
        return OperatingMode.INTERNAL_ONLY, False
    return OperatingMode.DEAD, False


def adjust_confidence(
    confidence_raw: float,
    mode: OperatingMode,
    sources: Iterable[SourceHealthV1],
    thresholds: ProductionThresholds = DEFAULT_THRESHOLDS,
) -> float:
    """Apply source-quality and operating-mode penalties to raw confidence."""

    sources = list(sources)
    healthy_sources = [source for source in sources if _is_healthy(source, thresholds)]
    penalty_sources = healthy_sources if healthy_sources else sources
    source_penalty = mean([source.quality_penalty for source in penalty_sources]) if penalty_sources else 0.0
    adjusted = confidence_raw - (source_penalty * 100.0)

    if mode == OperatingMode.EXTERNAL_DEGRADED:
        adjusted -= thresholds.degraded_mode_penalty * 100.0
    elif mode == OperatingMode.INTERNAL_ONLY:
        adjusted = max(0.0, adjusted - 25.0)
    elif mode == OperatingMode.DEAD:
        adjusted = 0.0

    return max(0.0, min(100.0, adjusted))


def calculate_incremental_alpha_bps(
    model_net_pnl: float,
    baseline_net_pnl: float,
    avg_book_notional: float,
) -> float:
    if avg_book_notional <= 0:
        raise ValueError("avg_book_notional must be positive")
    return ((model_net_pnl - baseline_net_pnl) / avg_book_notional) * 10_000


def build_economics_report(
    baseline_net_pnl: float,
    model_net_pnl: float,
    avg_book_notional: float,
    cost_breakdown: dict[str, float],
    thresholds: ProductionThresholds = DEFAULT_THRESHOLDS,
) -> EconomicsReportV1:
    incremental_alpha_bps = calculate_incremental_alpha_bps(
        model_net_pnl=model_net_pnl,
        baseline_net_pnl=baseline_net_pnl,
        avg_book_notional=avg_book_notional,
    )
    return EconomicsReportV1(
        baseline_net_pnl=baseline_net_pnl,
        model_net_pnl=model_net_pnl,
        incremental_alpha_bps=incremental_alpha_bps,
        cost_breakdown=cost_breakdown,
        continuation_gate_status=incremental_alpha_bps >= thresholds.incremental_alpha_hurdle_bps,
    )


def build_ablation_report(
    window: str,
    baseline_model_metrics: dict[str, float],
    enhanced_model_metrics: dict[str, float],
    feature_family: str,
    thresholds: ProductionThresholds = DEFAULT_THRESHOLDS,
) -> AblationReportV1:
    baseline_score = baseline_model_metrics["composite_score"]
    enhanced_score = enhanced_model_metrics["composite_score"]
    if baseline_score == 0:
        raise ValueError("baseline composite_score must be non-zero")
    lift = ((enhanced_score - baseline_score) / abs(baseline_score)) * 100.0
    return AblationReportV1(
        window=window,
        baseline_model_metrics=baseline_model_metrics,
        enhanced_model_metrics=enhanced_model_metrics,
        feature_family=feature_family,
        relative_lift_pct=lift,
        keep_feature_family=lift > 0,
    )


def should_keep_feature_family(
    report: AblationReportV1,
) -> bool:
    return report.keep_feature_family


def passes_economics_gate(
    report: EconomicsReportV1,
    thresholds: ProductionThresholds = DEFAULT_THRESHOLDS,
) -> bool:
    return report.incremental_alpha_bps >= thresholds.incremental_alpha_hurdle_bps


def passes_directional_gate(
    model_accuracy: float,
    baseline_accuracy: float,
    p_value: float,
    minimum_edge: float = 0.0,
    significance_alpha: float = 0.05,
) -> bool:
    return (model_accuracy - baseline_accuracy) > minimum_edge and p_value <= significance_alpha


def passes_composite_promotion_gate(
    directional_gate_passed: bool,
    economics_gate_passed: bool,
    risk_gate_passed: bool,
    shadow_gate_passed: bool,
) -> bool:
    return all(
        [
            directional_gate_passed,
            economics_gate_passed,
            risk_gate_passed,
            shadow_gate_passed,
        ]
    )


def _decision_threshold_for_mode(
    mode: OperatingMode,
    thresholds: ProductionThresholds = DEFAULT_THRESHOLDS,
) -> float:
    if mode == OperatingMode.FULL:
        return thresholds.decision_threshold_full
    if mode == OperatingMode.EXTERNAL_DEGRADED:
        return thresholds.decision_threshold_external_degraded
    if mode == OperatingMode.INTERNAL_ONLY:
        return thresholds.decision_threshold_internal_only
    return thresholds.decision_threshold_dead


def _confidence_floor_for_mode(
    mode: OperatingMode,
    thresholds: ProductionThresholds = DEFAULT_THRESHOLDS,
) -> float:
    if mode == OperatingMode.FULL:
        return thresholds.confidence_floor_full
    if mode == OperatingMode.EXTERNAL_DEGRADED:
        return thresholds.confidence_floor_external_degraded
    if mode == OperatingMode.INTERNAL_ONLY:
        return thresholds.confidence_floor_internal_only
    return 101.0


def _risk_cap_for_mode(
    mode: OperatingMode,
    thresholds: ProductionThresholds = DEFAULT_THRESHOLDS,
) -> float:
    if mode == OperatingMode.FULL:
        return thresholds.risk_cap_full
    if mode == OperatingMode.EXTERNAL_DEGRADED:
        return thresholds.risk_cap_external_degraded
    if mode == OperatingMode.INTERNAL_ONLY:
        return thresholds.risk_cap_internal_only
    return thresholds.risk_cap_dead


def _max_delta_for_mode(
    mode: OperatingMode,
    thresholds: ProductionThresholds = DEFAULT_THRESHOLDS,
) -> float:
    if mode == OperatingMode.FULL:
        return thresholds.max_delta_full
    if mode == OperatingMode.EXTERNAL_DEGRADED:
        return thresholds.max_delta_external_degraded
    if mode == OperatingMode.INTERNAL_ONLY:
        return thresholds.max_delta_internal_only
    return thresholds.max_delta_dead


def make_recommendation(
    forecast_return_2h: float,
    confidence_raw: float,
    sources: Iterable[SourceHealthV1],
    model_breakdown: dict[str, float] | None = None,
    thresholds: ProductionThresholds = DEFAULT_THRESHOLDS,
) -> tuple[PredictionOutputV3, DecisionRecommendationV3]:
    """Build a prediction artifact and the corresponding human-reviewed recommendation."""

    sources = list(sources)
    model_breakdown = model_breakdown or {}
    mode, external_anchor_present = determine_operating_mode(sources, thresholds)
    confidence_adjusted = adjust_confidence(
        confidence_raw=confidence_raw,
        mode=mode,
        sources=sources,
        thresholds=thresholds,
    )
    prediction = PredictionOutputV3(
        forecast_return_2h=forecast_return_2h,
        confidence_raw=confidence_raw,
        confidence_adjusted=confidence_adjusted,
        mode=mode,
        external_anchor_present=external_anchor_present,
        model_breakdown=model_breakdown,
    )

    threshold = _decision_threshold_for_mode(mode, thresholds)
    confidence_floor = _confidence_floor_for_mode(mode, thresholds)
    risk_cap = _risk_cap_for_mode(mode, thresholds)
    max_delta = _max_delta_for_mode(mode, thresholds)
    reasons = ["HUMAN_APPROVAL_REQUIRED"]

    if mode == OperatingMode.EXTERNAL_DEGRADED:
        reasons.append("LIMITED_CONTEXT_MODE")
    elif mode == OperatingMode.INTERNAL_ONLY:
        reasons.extend(["EXTERNAL_ANCHOR_ABSENT", "HOLD_ONLY_WEAK_SIGNAL"])
    elif mode == OperatingMode.DEAD:
        reasons.extend(["MARKET_ANCHOR_UNAVAILABLE", "HOLD_ONLY_MODE"])

    signal = Signal.HOLD
    if mode != OperatingMode.DEAD and confidence_adjusted >= confidence_floor:
        if forecast_return_2h >= threshold:
            signal = Signal.BUY_USD
        elif forecast_return_2h <= -threshold:
            signal = Signal.BUY_NGN

    target_inventory_split = {"usd": 0.50, "ngn": 0.50}
    if signal == Signal.BUY_USD:
        target_inventory_split = {"usd": 0.50 + max_delta, "ngn": 0.50 - max_delta}
    elif signal == Signal.BUY_NGN:
        target_inventory_split = {"usd": 0.50 - max_delta, "ngn": 0.50 + max_delta}

    recommendation = DecisionRecommendationV3(
        signal=signal,
        target_inventory_split=target_inventory_split,
        max_delta=max_delta if signal != Signal.HOLD else 0.0,
        risk_cap=risk_cap,
        requires_human_approval=True,
        constraint_reason_codes=reasons,
    )
    return prediction, recommendation
