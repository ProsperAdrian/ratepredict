# Model Governance Standard

## 1. Governance Objective

This standard governs when models may be trained, promoted, kept live, downgraded, or retired. The default stance is conservative because the system influences OTC inventory and therefore real balance-sheet risk.

## 2. Model Roles

- `Champion`: the currently approved production ensemble.
- `Challenger`: any candidate model, including attention-LSTM, seeking promotion.
- `Baseline`: passive spread-capture and naive directional baselines used to prove value.

Attention-LSTM is challenger-only until it passes all gates. Architectural sophistication is not a promotion criterion.

## 3. Mandatory Gates

### Directional Gate

The model must:

- Beat the naive directional baseline.
- Do so across walk-forward windows.
- Avoid regime collapse in trending, mean-reverting, and stressed conditions.
- Show statistical significance on the chosen evaluation test.

### Economics Gate

The model must:

- Deliver three-month rolling `incremental_alpha_bps >= 15`.
- Remain positive after cost deductions.
- Beat passive spread-capture baseline sufficiently to justify operational complexity.

### Feature Utility Review

Quarterly ablation must prove that each included optional feature family deserves to stay:

- Compare the core external-signal model with and without each optional family.
- Keep a family only if it improves composite score, confidence calibration, or cost-adjusted economics.
- Remove families that add operational complexity without measurable benefit.

### Risk Gate

The model must:

- Respect all inventory, drawdown, and mode-specific caps.
- Produce no unresolved policy breaches.
- Demonstrate deterministic degraded-mode behavior.

### Shadow Gate

The model must:

- Complete at least `12` weeks of shadow trading.
- Maintain passing status throughout the shadow period.
- Produce stable audit records and approval workflows.

## 4. Promotion Rules

A challenger becomes champion only if every mandatory gate passes. Composite failure on any single gate blocks promotion.

Promotion package must include:

- Model artifact version.
- Feature-set version.
- Training and validation windows.
- Economics report.
- Feature-family ablation report.
- Regime breakdown.
- Known failure modes.
- Rollback plan.

## 5. Continuation Rules

The live champion remains approved only while:

- Economics hurdle remains above `15 bps`.
- Included feature families continue to justify themselves in quarterly ablations.
- Data and policy incidents stay within tolerance.
- Override behavior does not show unexplained systematic disagreement with the model.

If continuation rules fail, the champion enters one of:

- Controlled downgrade.
- Shadow-only mode.
- Redesign review.
- Retirement.

## 6. Retraining Rules

Retraining is allowed only under governed conditions:

- Scheduled retraining on approved cadence.
- Emergency retraining after documented performance deterioration.
- No direct production deployment without fresh gate evaluation.

Every retrain must preserve:

- Point-in-time dataset lineage.
- Reproducible feature definitions.
- Comparable backtest methodology.

## 7. Audit Requirements

Every production decision cycle must emit immutable records for:

- Source health.
- Operating mode.
- Feature availability.
- Prediction output.
- Recommendation output.
- Human approval or override.
- Production deployment and rollback events.

Audit gaps are a production incident, not a documentation issue.
