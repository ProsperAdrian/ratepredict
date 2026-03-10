# Acceptance Gatebook

## 1. Purpose

This gatebook defines the formal pass or fail rules for promotion, continuation, downgrade, and stop review. A model is not "good" because it looks interesting in backtests. It is good only if it clears every required gate under governed conditions.

## 2. Promotion Gates

### Directional Gate

Pass conditions:

- Beats the naive directional baseline.
- Holds up across walk-forward windows.
- Does not collapse in any key regime.
- Meets the chosen significance test threshold.

Fail action:

- Challenger remains out of production influence.

### Economics Gate

Pass conditions:

- Three-month rolling `incremental_alpha_bps >= 15`.
- Net result remains positive after slippage, spread leakage, fees, and operating costs.
- Improvement is material relative to passive spread-capture baseline.

Fail action:

- Model cannot be promoted or continued without redesign review.

### Proprietary Edge Gate

Pass conditions:

- Quarterly feature-family ablations are performed on the same evaluation windows as the core model.
- Every included optional family shows measurable value in composite score, calibration, or economics.

Fail action:

- Remove the failing family from the production feature set or enter redesign review if the signal collapses without it.

### Risk Gate

Pass conditions:

- No unresolved inventory-limit breaches.
- No unresolved drawdown-limit breaches.
- Deterministic operating-mode transitions verified.
- No silent circularity violations.

Fail action:

- Model may remain shadow-only or be removed from controlled influence.

### Shadow Gate

Pass conditions:

- At least `12` consecutive weeks of shadow operation.
- Composite gate remains passing through the shadow window.
- Audit trail and approval workflow operate without gaps.

Fail action:

- No live inventory influence.

## 3. Continuation Gates

The live champion continues only if all of the following remain true:

- Economics gate remains passing on a rolling three-month basis.
- Included feature families remain justified on the latest quarterly ablation.
- Risk gate remains passing.
- Governance artifacts remain complete and current.

Any failure moves the system into controlled downgrade, shadow-only mode, redesign review, or retirement.

## 4. Test-to-Gate Mapping

- Unit tests validate mode transitions, circularity controls, and formula correctness.
- Walk-forward experiments validate directional and regime behavior.
- Shadow trading validates economics, risk, and operational readiness.
- Incident drills validate rollback, degraded-mode correctness, and source-failure handling.

## 5. Ownership

- Trading signs off on recommendation usability and override integrity.
- Data signs off on source health, manual verification, and point-in-time correctness.
- ML signs off on challenger evidence, ablation results, and regime behavior.
- Engineering signs off on deployment, rollback, monitoring, and audit durability.
