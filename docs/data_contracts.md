# Data Contracts and Public Interfaces

## 1. Contract Principles

All contracts are point-in-time, auditable, and versioned. They exist to prevent hidden leakage, ambiguous source handling, and silent degradation.

The executable definitions live in [`ratepredict/types.py`](../ratepredict/types.py).

## 2. Source Classification

### `SourceHealthV1`

Purpose:

- Describe the operational health of a single upstream source at decision time.

Fields:

- `source_id`: stable identifier such as `quidax_internal`, `binance_p2p`, `abokifx`, or `fmdq`.
- `tier_class`: `verified` or `monitored`.
- `freshness_sec`: age of the last usable record at evaluation time.
- `integrity_score`: normalized source quality score from `0.0` to `1.0`.
- `verification_status`: operational status such as `verified`, `monitored`, `stale`, `manual_review`, or `failed`.
- `quality_penalty`: normalized penalty from `0.0` to `1.0`.

Operational notes:

- Verified sources should normally have lower penalties.
- Monitored sources can still participate in production if they pass freshness and integrity rules.
- Quidax internal feed is mandatory for any directional recommendation.

## 3. Feature Contract

### `FeatureVectorV3`

Purpose:

- Store the feature state used for a single prediction run.

Fields:

- `as_of_time`: model decision timestamp.
- `mode`: operating mode `A`, `B`, `C`, or `D`.
- `proprietary_features`: Quidax-only signals.
- `public_features`: external anchors and context features.
- `missing_flags`: identifiers for unavailable features or sources.
- `effective_penalty`: total penalty carried into policy adjustment.

Operational notes:

- Proprietary and public features are split by design so family-level ablations can be run cleanly.
- Missing flags must be persisted, not dropped silently.
- `mode` must reflect the source-health evaluation used for the run.

## 4. Prediction Contract

### `PredictionOutputV3`

Purpose:

- Represent the pre-decision model output after confidence adjustment inputs are known.

Fields:

- `forecast_return_2h`: expected 2-hour return in decimal form.
- `confidence_raw`: pre-policy confidence score on a `0-100` scale.
- `confidence_adjusted`: post-policy confidence score on a `0-100` scale.
- `mode`: operating mode at prediction time.
- `external_anchor_present`: whether at least one usable non-internal anchor exists.
- `model_breakdown`: component outputs such as champion ensemble members or challenger references.

Operational notes:

- `confidence_adjusted` is the value that decision rules must use.
- `external_anchor_present` prevents silent circularity.

## 5. Recommendation Contract

### `DecisionRecommendationV3`

Purpose:

- Encode the policy-constrained recommendation delivered to the desk.

Fields:

- `signal`: `buy_usd`, `buy_ngn`, or `hold`.
- `target_inventory_split`: recommended USD and NGN target weights.
- `max_delta`: maximum permitted inventory shift for the recommendation.
- `risk_cap`: mode-adjusted risk budget cap.
- `requires_human_approval`: launch default is always `true`.
- `constraint_reason_codes`: machine-readable reasons explaining degradation or blocks.

Operational notes:

- This object is the decision artifact to be audited and approved.
- Any manual override must link back to the originating recommendation.

## 6. Evaluation Contracts

### `AblationReportV1`

Purpose:

- Record whether an added feature family improves the signal enough to justify inclusion.

Fields:

- `window`: evaluation period identifier.
- `baseline_model_metrics`: metrics for the core model before adding the candidate family.
- `enhanced_model_metrics`: metrics after adding the candidate family.
- `feature_family`: name of the family under test, such as `internal_microstructure` or `llm_context`.
- `relative_lift_pct`: percentage lift from the added family.
- `keep_feature_family`: whether the family should remain in the champion feature set.

### `EconomicsReportV1`

Purpose:

- Record whether the system is economically justified.

Fields:

- `baseline_net_pnl`: passive spread-capture baseline net P&L.
- `model_net_pnl`: model-guided net P&L.
- `incremental_alpha_bps`: economic uplift in basis points.
- `cost_breakdown`: slippage, fees, operational cost, and other deductions.
- `continuation_gate_status`: whether the `>= 15 bps` hurdle is met.
