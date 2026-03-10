# World-Class Production Methodology v3 for USD/NGN 2-Hour Decision Intelligence

## 1. Purpose

This system is a Quidax trading intelligence platform for one question: can we generate the strongest possible 2-hour directional signal on the relevant NGN/USD market anchor, and should the OTC desk trade on that signal at a high enough confidence level to justify risk?

It is not built on an assumption that Quidax's internal data is itself a moat. Quidax internal data may improve execution quality, selectivity, or confidence, but the primary signal may come from external price anchors, cross-venue dislocations, and broader market context. Internal data is therefore treated as an optional feature family that must earn its place empirically.

## 2. Business Objective and Continuation Economics

The system exists only if it adds measurable economics beyond the desk's passive spread-capture baseline.

The formal continuation formula is:

```text
incremental_alpha_bps =
  (model_net_pnl - passive_baseline_net_pnl) / avg_book_notional * 10,000
```

Production continuation requires:

- Three-month rolling `incremental_alpha_bps >= 15`.
- Net P&L measured after spread leakage, slippage, fees, and operational costs.
- No unresolved model-risk breaches, policy breaches, or drawdown breaches.

If the economics gate fails for a quarter, the system enters redesign review. If leadership concludes that the expected alpha does not justify engineering and operational cost, the program stops.

## 3. Signal Thesis and Feature Strategy

The system's first priority is predictive strength and tradeability, not proving that Quidax data is uniquely special. Feature families are included only if they improve the probability that the model correctly predicts market direction and the economics of trading that prediction.

Priority external signal families:

- Cross-venue NGN/USD anchor prices and spreads.
- Cross-venue dislocations and premium z-scores.
- Short-horizon momentum and reversal signals on the anchor.
- Regional FX and risk proxies.
- Event and calendar context where point-in-time safe.

Optional enrichment families:

- Quidax internal microstructure and inventory features.
- LLM-derived news and event features.
- Slow macro or policy context.

Quarterly family ablation is mandatory:

- Train the core external-signal model.
- Add one feature family at a time on the same walk-forward windows.
- Keep a family only if it improves composite signal quality, calibrated confidence, or cost-adjusted economics.
- Remove any family that adds noise, instability, or operational risk without measurable benefit.

## 4. Architecture

The production architecture has six layers:

1. Data acquisition and point-in-time storage.
2. Data quality control and source health classification.
3. Feature generation with explicit separation of proprietary and public features.
4. Champion/challenger model layer.
5. Policy engine for confidence adjustment, operating mode selection, and recommendation constraints.
6. Decision audit, monitoring, approvals, and governance.

Every 2-hour production cycle must produce a full audit record covering source health, feature availability, model outputs, adjusted confidence, recommended action, policy constraints, and human override status.

## 5. Data Strategy

### 5.1 Source Classes

Production uses two Tier-1 classes rather than a single rigid class:

- `Tier-1 Verified`: contractual or API-backed feeds with stable automation and explicit operational ownership.
- `Tier-1 Monitored`: scraped or semi-structured sources that are operationally controlled through freshness rules, integrity checks, and manual verification protocols.

Examples:

- Verified: Quidax internal OTC feed, Binance P2P, approved institutional vendors.
- Monitored: AbokiFX, FMDQ pages, CBN pages where direct APIs are not available.

Monitored sources remain usable in production because the market requires them, but they are penalized for reliability risk.

### 5.2 Anchor Control

The platform must not infer tradable market direction from Quidax internal activity alone. At least one healthy non-internal external anchor is required for any normal directional recommendation.

Permitted external anchors include:

- Binance P2P.
- Approved institutional feeds.
- Monitored anchor sources such as AbokiFX or FMDQ if they pass health checks.

When no external anchor is healthy, the system automatically degrades to `HOLD`.

### 5.3 Point-in-Time Requirements

Every source record must store:

- Event time.
- Ingest time.
- Freshness in seconds.
- Integrity score.
- Verification status.
- Source class.
- Quality penalty.

No feature may use future information or revised data without storing the effective timestamp that was known at decision time.

## 6. Operating Modes

The platform uses a deterministic degradation hierarchy:

- `Mode A (Full)`: healthy external anchors across Verified and Monitored classes, plus healthy internal context. Normal confidence and standard risk caps.
- `Mode B (Tradable but constrained)`: at least one healthy external anchor is available, but either one expected source class is degraded or internal context is missing. Apply confidence penalty and tighter risk caps.
- `Mode C (Signal weak)`: internal context may exist, but no healthy non-internal external anchor exists. Do not issue normal directional trade recommendations.
- `Mode D (Dead mode)`: critical market anchors are unavailable and the system cannot form a reliable market view. Output `HOLD` only and raise an incident.

Mode C minimum viable context:

- Internal execution context if available.
- Time and calendar features.
- Explicit reason codes showing that market-direction confidence is not tradable.

## 7. Modeling Strategy

### 7.1 Champion and Challenger

The production champion remains tree-based:

- XGBoost.
- LightGBM.
- Ridge regression anchor.

Attention-LSTM remains a challenger model until it passes the same promotion gates as the champion. It does not enter production simply because it is sophisticated.

### 7.2 Validation

Model validation must be purged and walk-forward:

- Rolling train, validate, and test windows with embargo to reduce leakage.
- Regime-segmented evaluation.
- Event-window evaluation around policy shocks and liquidity stress.
- Cost-adjusted simulation, not raw hit-rate only.

Headline metrics:

- Directional accuracy versus naive baseline.
- Cost-adjusted net P&L.
- Incremental alpha in basis points.
- Maximum drawdown.
- Policy breach count.
- Regime stability.

### 7.3 Promotion Logic

Promotion is composite and conservative:

- Directional gate must pass.
- Economics gate must pass.
- Risk gate must pass.
- Shadow trading gate must pass.

No model is promoted if any gate fails.

## 8. Decision Policy

The platform is human in the loop at launch. It produces constrained recommendations, not autonomous inventory changes.

Decision policy requirements:

- Human approval required for every inventory recommendation.
- All overrides must be logged with reason codes.
- Mode-driven confidence penalties must be applied before decision thresholds are checked.
- External-anchor absence must block normal directional behavior.
- Dead mode must always return `HOLD`.

Mode-specific risk posture:

- Mode A: normal risk cap.
- Mode B: tighter cap and smaller maximum inventory delta.
- Mode C: no normal directional trade recommendation.
- Mode D: no directional recommendation.

## 9. Promotion, Continuation, and Stop Rules

Promotion to controlled live influence requires:

- Minimum `12` weeks of shadow trading.
- Composite gate pass across walk-forward windows and live shadow observation.
- No unremediated data or governance incidents.

Continuation requires:

- Three-month rolling incremental net alpha at or above `15 bps`.
- Quarterly feature-family review showing that included families continue to add value.
- No material policy-control failure.

Stop or redesign review is triggered by:

- Economics failure.
- Repeated external-anchor failure or circularity breach.
- Persistent feature-family instability that degrades model quality.
- Material audit or override integrity failure.

## 10. Operational Standard

This system is production infrastructure, not a dashboard demo. The production standard therefore requires:

- Immutable audit events for every prediction and approval.
- Versioned model artifacts and feature definitions.
- Champion/challenger deployment controls.
- Rollback procedures and incident playbooks.
- Deterministic degradation behavior.
- Explicit ownership for data, models, and desk decisions.

This methodology should be treated as the operating contract between leadership, engineering, data science, and the OTC desk.
