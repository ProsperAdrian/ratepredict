# Operating Modes and Runbooks

## 1. Mode Definitions

### Mode A: Full

Entry conditions:

- At least one healthy Verified external source.
- At least one healthy Monitored external source.
- Quidax internal context healthy.

Policy:

- No base confidence penalty.
- Standard directional thresholds.
- Standard risk cap and max inventory delta.

### Mode B: External Degraded

Entry conditions:

- At least one healthy non-internal external anchor present.
- Either internal context is unavailable or one expected source class is stale, inconsistent, or unavailable.

Policy:

- Confidence penalty applied.
- Tighter risk cap.
- Smaller max inventory delta.
- Recommendation reason codes must identify degradation.

### Mode C: Internal-Only

Entry conditions:

- No healthy non-internal external anchor available.
- Internal context may or may not be available.

Policy:

- No normal directional trade recommendation.
- Recommendation must explicitly state anchor absence and low market confidence.
- `HOLD` is the default action.

### Mode D: Dead Mode

Entry conditions:

- Critical market anchors unavailable or failed integrity checks.

Policy:

- Output `HOLD` only.
- Zero risk cap.
- Trigger incident and manual desk review.

## 2. Manual Verification Protocol for Monitored Sources

Monitored sources remain part of production because the market does not provide clean APIs for every critical anchor. To control that risk:

- Freshness rules run automatically.
- Integrity scoring compares current values to recent historical ranges and alternate anchors.
- Stale or inconsistent monitored sources are flagged for manual review.
- Manual review outcomes are recorded in the audit trail.
- Unverified monitored data cannot silently recover back into Mode A.

## 3. Incident Runbooks

### External Source Degradation

Actions:

- Evaluate health state and enter Mode B if at least one anchor remains.
- Apply confidence and exposure penalties.
- Open a source-quality incident if degradation persists beyond the defined threshold.

### External Anchor Loss

Actions:

- Enter Mode C immediately.
- Restrict outputs to non-tradable or `HOLD` state.
- Attach `EXTERNAL_ANCHOR_ABSENT` to every recommendation.

### Internal Feed Failure

Actions:

- If external anchors remain healthy, enter Mode B with constrained tradeability.
- If external anchors are also unhealthy, enter Mode D.
- Escalate to trading, data, and engineering owners.

### Data Poisoning or Integrity Breach

Actions:

- Quarantine the affected source.
- Recompute mode using the remaining healthy sources.
- Preserve all raw and derived evidence for audit.
- If the issue affects the internal feed, remain in Mode D until cleared.

## 4. Runbook Ownership

The production owner set must be explicit:

- Trading owner for decision approval.
- Data owner for source contracts and manual verification.
- ML owner for model health and challenger governance.
- Engineering owner for deployment, alerting, and rollback.
