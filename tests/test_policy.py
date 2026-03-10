from __future__ import annotations

import unittest

from ratepredict.policy import (
    build_ablation_report,
    build_economics_report,
    calculate_incremental_alpha_bps,
    determine_operating_mode,
    make_recommendation,
    passes_composite_promotion_gate,
)
from ratepredict.types import OperatingMode, Signal, SourceHealthV1


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.internal = SourceHealthV1(
            source_id="quidax_internal",
            tier_class="verified",
            freshness_sec=300,
            integrity_score=0.98,
            verification_status="verified",
            quality_penalty=0.01,
        )
        self.verified_external = SourceHealthV1(
            source_id="binance_p2p",
            tier_class="verified",
            freshness_sec=600,
            integrity_score=0.95,
            verification_status="verified",
            quality_penalty=0.02,
        )
        self.monitored_external = SourceHealthV1(
            source_id="abokifx",
            tier_class="monitored",
            freshness_sec=3_600,
            integrity_score=0.90,
            verification_status="monitored",
            quality_penalty=0.10,
        )

    def test_mode_a_requires_verified_and_monitored_external_sources(self) -> None:
        mode, external_anchor_present = determine_operating_mode(
            [self.internal, self.verified_external, self.monitored_external]
        )
        self.assertEqual(mode, OperatingMode.FULL)
        self.assertTrue(external_anchor_present)

    def test_mode_b_when_any_expected_external_class_is_degraded(self) -> None:
        stale_monitored = SourceHealthV1(
            source_id="abokifx",
            tier_class="monitored",
            freshness_sec=12 * 60 * 60,
            integrity_score=0.90,
            verification_status="stale",
            quality_penalty=0.20,
        )
        mode, external_anchor_present = determine_operating_mode(
            [self.internal, self.verified_external, stale_monitored]
        )
        self.assertEqual(mode, OperatingMode.EXTERNAL_DEGRADED)
        self.assertTrue(external_anchor_present)

    def test_mode_c_when_only_internal_data_is_usable(self) -> None:
        mode, external_anchor_present = determine_operating_mode([self.internal])
        self.assertEqual(mode, OperatingMode.INTERNAL_ONLY)
        self.assertFalse(external_anchor_present)

    def test_mode_b_when_external_anchor_is_healthy_even_if_internal_is_unavailable(self) -> None:
        failed_internal = SourceHealthV1(
            source_id="quidax_internal",
            tier_class="verified",
            freshness_sec=4 * 60 * 60,
            integrity_score=0.20,
            verification_status="failed",
            quality_penalty=0.90,
        )
        mode, external_anchor_present = determine_operating_mode(
            [failed_internal, self.verified_external, self.monitored_external]
        )
        self.assertEqual(mode, OperatingMode.EXTERNAL_DEGRADED)
        self.assertTrue(external_anchor_present)

    def test_internal_only_mode_forces_hold(self) -> None:
        prediction, recommendation = make_recommendation(
            forecast_return_2h=0.010,
            confidence_raw=90.0,
            sources=[self.internal],
        )
        self.assertEqual(prediction.mode, OperatingMode.INTERNAL_ONLY)
        self.assertFalse(prediction.external_anchor_present)
        self.assertEqual(prediction.confidence_adjusted, 64.0)
        self.assertEqual(recommendation.signal, Signal.HOLD)
        self.assertIn("EXTERNAL_ANCHOR_ABSENT", recommendation.constraint_reason_codes)
        self.assertEqual(recommendation.risk_cap, 0.0)
        self.assertEqual(recommendation.max_delta, 0.0)

    def test_external_anchor_with_missing_internal_can_still_trade_conservatively(self) -> None:
        failed_internal = SourceHealthV1(
            source_id="quidax_internal",
            tier_class="verified",
            freshness_sec=9_999,
            integrity_score=0.10,
            verification_status="failed",
            quality_penalty=0.95,
        )
        prediction, recommendation = make_recommendation(
            forecast_return_2h=0.020,
            confidence_raw=99.0,
            sources=[failed_internal, self.verified_external],
        )
        self.assertEqual(prediction.mode, OperatingMode.EXTERNAL_DEGRADED)
        self.assertEqual(recommendation.signal, Signal.BUY_USD)
        self.assertIn("LIMITED_CONTEXT_MODE", recommendation.constraint_reason_codes)
        self.assertEqual(recommendation.risk_cap, 0.07)

    def test_incremental_alpha_formula(self) -> None:
        alpha_bps = calculate_incremental_alpha_bps(
            model_net_pnl=57_500,
            baseline_net_pnl=50_000,
            avg_book_notional=5_000_000,
        )
        self.assertAlmostEqual(alpha_bps, 15.0)

    def test_economics_gate_uses_locked_hurdle(self) -> None:
        report = build_economics_report(
            baseline_net_pnl=50_000,
            model_net_pnl=57_500,
            avg_book_notional=5_000_000,
            cost_breakdown={"slippage": 1_000, "ops": 500},
        )
        self.assertTrue(report.continuation_gate_status)

    def test_feature_family_ablation_keeps_only_positive_contributors(self) -> None:
        report = build_ablation_report(
            window="2026-Q1",
            baseline_model_metrics={"composite_score": 1.00},
            enhanced_model_metrics={"composite_score": 1.05},
            feature_family="internal_microstructure",
        )
        self.assertTrue(report.keep_feature_family)
        self.assertAlmostEqual(report.relative_lift_pct, 5.0)

    def test_composite_promotion_gate_requires_all_controls(self) -> None:
        self.assertFalse(
            passes_composite_promotion_gate(
                directional_gate_passed=True,
                economics_gate_passed=True,
                risk_gate_passed=True,
                shadow_gate_passed=False,
            )
        )
        self.assertTrue(
            passes_composite_promotion_gate(
                directional_gate_passed=True,
                economics_gate_passed=True,
                risk_gate_passed=True,
                shadow_gate_passed=True,
            )
        )


if __name__ == "__main__":
    unittest.main()
