# Quidax USD/NGN Decision Intelligence

This repository implements the production specification for a world-class USD/NGN 2-hour decision intelligence system for the Quidax OTC desk.

It contains two layers:

- Executable domain contracts and policy logic in [`ratepredict`](./ratepredict).
- Production-grade methodology, governance, data, and runbook documents in [`docs`](./docs).

## Repository Structure

- [`docs/world_class_production_methodology_v3.md`](./docs/world_class_production_methodology_v3.md): End-to-end methodology and operating model.
- [`docs/training_blueprint.md`](./docs/training_blueprint.md): Training objectives, required datapoints, dataset design, and environment guidance.
- [`docs/runtime_data_pipeline.md`](./docs/runtime_data_pipeline.md): Scheduled runtime bar refresh, cached external data flow, and remaining source decisions.
- [`docs/data_contracts.md`](./docs/data_contracts.md): Point-in-time data contracts and interface semantics.
- [`docs/acceptance_gatebook.md`](./docs/acceptance_gatebook.md): Formal pass/fail gates for promotion, continuation, and stop-review.
- [`docs/model_governance_standard.md`](./docs/model_governance_standard.md): Promotion, continuation, challenger, and audit controls.
- [`docs/operating_modes_and_runbooks.md`](./docs/operating_modes_and_runbooks.md): Degradation hierarchy, verification workflows, and incident actions.
- [`app/main.py`](./app/main.py): FastAPI desk app for live inference and signal review.
- [`artifacts/`](./artifacts): Trained model files, metadata, feature importances, and the latest BigQuery export.
- [`ratepredict/types.py`](./ratepredict/types.py): Public types that operationalize the specification.
- [`ratepredict/policy.py`](./ratepredict/policy.py): Deterministic policy engine defaults for modes, penalties, gates, and recommendations.
- [`tests/test_policy.py`](./tests/test_policy.py): Acceptance tests for degradation logic, circularity controls, and economics gates.

## Locked Production Defaults

- Launch posture: human in the loop.
- Risk posture: capital preservation.
- Primary objective: predict the next 2-hour move of the Quidax USDT/NGN close, because that is the OTC execution price.
- Cross-venue and macro series remain exogenous features, not target components.
- Economic continuation gate: `+15 bps` monthly incremental net alpha versus passive spread-capture baseline.
- Included feature families must justify themselves in ablations; any family that does not improve signal quality or economics should be removed.
- Live `usdtngn` and `btcngn` quotes are fetched directly from the Quidax public ticker endpoints.

## Verification

Run:

```bash
. .venv/bin/activate
python -m unittest discover -s tests
uvicorn app.main:app --reload
```
