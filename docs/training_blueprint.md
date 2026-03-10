# Training Blueprint

## 1. What You Are Actually Training

The first production champion should be a supervised model that scores the next 2-hour directional move in the best available NGN/USD market anchor, while evaluation is done on both direction and net economics.

Use this split:

- Primary training target: 2-hour forward return of the selected market anchor, typically a quality-controlled consensus built from Binance P2P, AbokiFX, FMDQ, and any approved institutional feed.
- Secondary evaluation targets: directional hit rate, cost-adjusted net P&L, incremental alpha in basis points, and drawdown.
- Challenger research targets: regime classifier and attention-LSTM only after the champion pipeline is stable.

Important constraint:

- At a 2-hour desk cadence, you only get roughly `1,200` to `1,600` training rows per year.
- Two years of history gives you only about `2,500` to `3,200` supervised rows.
- That is enough for disciplined tree models.
- That is not enough reason to start with a deep learning stack as the champion.

## 2. Training Row Design

Each row should represent one decision timestamp.

Recommended row grain:

- Every 2 hours during official OTC decision windows.
- One row per decision timestamp in UTC plus WAT-local trading metadata.
- Every field must be point-in-time correct as of that timestamp.

Every row needs:

- `decision_time_utc`
- `decision_time_wat`
- `session_id`
- `day_of_week`
- `hour_of_day`
- `is_month_end`
- `is_holiday_window`
- `mode_at_decision`
- `external_anchor_present`

## 3. Raw Datapoints You Need

These are raw inputs to collect before feature engineering. They are split into `required` and `strongly recommended`.

### 3.1 Recommended: Quidax Internal Data

This is optional signal enrichment and execution context. It is useful only if it measurably improves trading outcomes.

- OTC bid at timestamp.
- OTC ask at timestamp.
- OTC executable mid.
- OTC quoted spread in absolute NGN.
- OTC quoted spread in basis points.
- Inventory position in USD.
- Inventory position in NGN.
- Inventory deviation from neutral target.
- Client buy volume over trailing 2h.
- Client sell volume over trailing 2h.
- Trade count over trailing 2h.
- Average trade size over trailing 2h.
- Largest trade size over trailing 2h.
- Large-trade count over trailing 2h.
- Time since last trade.
- Time since last large trade.
- Client concentration HHI over trailing 24h.
- Top-1 client share over trailing 24h.
- Top-3 client share over trailing 24h.
- BTC/NGN internal price.
- USDT/NGN internal price if available.
- Quote revision count over trailing 2h.
- Quote skew or internal mark-up adjustment if tracked.
- Trade direction labels from actual desk flow.

Strongly recommended internal extensions:

- RFQ count and RFQ acceptance rate.
- Client segment mix by regulated internal taxonomy.
- Cancelled quote count.
- Filled versus quoted notional ratio.
- Desk override history and reason code frequency.

### 3.2 Required: External Anchor Data

This is the core signal layer. You need at least one high-integrity non-internal anchor for any normal directional recommendation.

- Binance P2P NGN buy price.
- Binance P2P NGN sell price.
- Binance P2P mid.
- Binance P2P spread.
- AbokiFX buy price.
- AbokiFX sell price.
- AbokiFX mid.
- AbokiFX spread.
- FMDQ or NAFEM official rate.
- FMDQ turnover or liquidity proxy if available.
- Approved institutional vendor quote if Quidax can license one.

For every external source record, also collect:

- `event_time`
- `ingest_time`
- `source_id`
- `tier_class`
- `verification_status`
- `freshness_sec`
- `integrity_score`
- `quality_penalty`

### 3.3 Strongly Recommended: Market Context

These are not the moat, but they help the model understand when microstructure signals should or should not be trusted.

- DXY level.
- DXY 1d and 7d changes.
- Brent crude level.
- Brent 1d and 3d changes.
- VIX level.
- BTC/USD spot price.
- USDT/USD deviation from parity if available.
- USD/ZAR spot and changes.
- USD/GHS spot and changes.
- USD/KES spot and changes.

### 3.4 Recommended: Calendar and Macro Context

- Latest CBN reserves level.
- Week-over-week change in reserves.
- Latest MPC decision date.
- MPC decision surprise flag.
- FAAC proximity.
- Month-end flag.
- Salary-week flag if useful internally.
- School-fee season flag.
- Hajj-season flag if useful for demand bursts.

These are slower-moving variables. They can still be used, but only with proper effective timestamps and staleness flags.

### 3.5 Optional but Valuable: Event and News Context

Only use these if you can version and timestamp them properly.

- Headline text.
- Headline source.
- Headline publish time.
- Topic tags.
- LLM sentiment score.
- LLM event-magnitude score.
- LLM urgency flag.

Do not train on unversioned or manually edited LLM outputs.

## 4. Labels You Need

You need more than one label.

### Primary supervised label

- `target_return_2h = (anchor_price_t_plus_2h - anchor_price_t) / anchor_price_t`

### Derived directional label

- `target_direction_2h = sign(target_return_2h)`

### Economic labels for evaluation

- `pnl_if_buy_usd`
- `pnl_if_buy_ngn`
- `pnl_if_hold`
- `spread_capture_bps`
- `slippage_bps`
- `inventory_drawdown`

### Governance labels

- `regime_label`
- `mode_label`
- `source_health_breach_flag`
- `policy_breach_flag`

The model can be trained on return or direction, but the platform must always evaluate on economics as well. Internal data is not automatically retained; it must prove usefulness in ablations.

## 5. Where To Store and Train It

### Development and research

Do early research on a secure Quidax-controlled machine:

- Your laptop if it has secure access and the data volume is still manageable.
- A private EC2 instance or internal VM if the data is sensitive or larger.

Recommended storage:

- Raw data in Postgres or TimescaleDB if you want transactional and time-series access.
- Curated point-in-time datasets as Parquet files in secure object storage.
- Experiment tracking in MLflow.

### Production retraining

Do not rely on Codex itself as the training environment.

Use:

- A scheduled Python training job on Quidax infrastructure.
- Preferably a containerized job on EC2, ECS, Kubernetes, or an internal VM.
- Model artifacts stored in versioned object storage.
- Metrics and lineage stored in a registry or metadata store.

Why:

- Codex can help write and run training code locally.
- Codex is not your production scheduler, artifact registry, data perimeter, or long-running compute platform.

## 6. Do You Need To Train Outside Codex?

Yes for production. Not necessarily for development.

Practical answer:

- You can build the code with Codex.
- You can run local experiments from this repo with Codex helping you.
- You should train and retrain the real production model on Quidax-controlled infrastructure outside the chat environment.

For the current champion models:

- XGBoost, LightGBM, and Ridge can be trained on CPU.
- A normal secure VM or EC2 instance is enough.

For the LSTM challenger:

- Use a separate GPU-enabled environment only if the simpler champion already works and the data volume justifies it.

## 7. Recommended Training Workflow

1. Build raw ingestion first.
2. Build an external anchor constructor with source health and penalties.
3. Build a point-in-time dataset builder.
4. Create one training table with one row per decision timestamp.
5. Train Ridge, XGBoost, and LightGBM on the same dataset.
6. Run purged walk-forward validation.
7. Run family-level ablations, including internal-data and LLM-data add-ons.
8. Run cost-adjusted backtest.
9. Produce `AblationReportV1` and `EconomicsReportV1`.
10. Only then consider shadow deployment.

## 8. What Not To Do

- Do not start with deep learning as the champion.
- Do not train on data that was manually corrected without versioning.
- Do not use revised official rates as if they were known earlier.
- Do not mix data collected at different timestamps without explicit freshness fields.
- Do not optimize for directional accuracy alone.
- Do not let notebooks become the production training system.
