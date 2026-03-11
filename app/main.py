from __future__ import annotations

import csv
import logging
import pickle
import sys
import threading
import time
from datetime import UTC, datetime, timedelta, timezone
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings, get_settings, reload_settings
from app.macro_calendar import get_month_events as _get_month_events, get_categories as get_macro_categories, EVENTS as MACRO_EVENTS
from app.schemas import InferenceSnapshot
from app.services.artifacts import ArtifactLoader, ExportLoader
from app.services.gemini_ai import GeminiAIContextEngine
from app.services.features import PublicFeatureBuilder
from app.services.market_data import ExternalDailyMarketDataService, QuidaxTickerService
from app.services.news_aggregator import NewsAggregatorService, format_news_for_prompt, format_news_summary_stats

WAT = timezone(timedelta(hours=1))

st.set_page_config(
    page_title="Quidax OTC Rate Intelligence",
    page_icon="\u20a6",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom desk UI
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;700&display=swap');
:root {
    --bg: #f5f1e8;
    --panel: #fffdf8;
    --panel-strong: #ffffff;
    --ink: #16202b;
    --muted: #5f6b76;
    --line: #ddd4c3;
    --brand: #123b63;
    --brand-soft: #dce7f2;
    --up: #13795b;
    --up-soft: #dff3e9;
    --down: #b4492b;
    --down-soft: #f9e3dc;
    --amber: #a56a00;
    --amber-soft: #f8edcf;
    --shadow: 0 20px 40px rgba(31, 42, 55, 0.06);
}
html, body, [class*="stApp"], p, span, div, input, textarea, select, button, label, td, th, li {
    font-family: 'IBM Plex Sans', sans-serif !important;
}
/* Preserve Material Icons used by Streamlit for sidebar toggle etc. */
[data-testid="stSidebarCollapseButton"] span,
[data-testid="stSidebarNavToggle"] span,
button[kind="headerNoPadding"] span,
header[data-testid="stHeader"] button span,
[data-testid="collapsedControl"] span,
[data-testid="collapsedControl"] button span,
[class*="material"],
.e1fb0mya1 span {
    font-family: 'Material Symbols Rounded', 'Material Icons', sans-serif !important;
}
h1,h2,h3,h4,h5,h6 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
    letter-spacing: -0.03em;
    color: var(--ink);
}
#MainMenu, footer, .stDeployButton { display: none; }
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 2.35rem !important;
    box-shadow: none !important;
    border-bottom: none !important;
    backdrop-filter: none !important;
}
[data-testid="stHeader"] > div {
    height: 2.35rem !important;
    background: transparent !important;
    box-shadow: none !important;
    border-bottom: none !important;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at top left, rgba(18, 59, 99, 0.08), transparent 28%),
        linear-gradient(180deg, #fbf8f1 0%, var(--bg) 100%);
}
/* Ensure sidebar toggle button is always visible */
button[data-testid="stSidebarCollapseButton"],
button[data-testid="stSidebarNavToggle"],
header[data-testid="stHeader"] button { z-index: 999 !important; }

/* Leave enough room below Streamlit's top chrome */
.stMainBlockContainer, .block-container {
    padding-top: 2.45rem !important;
    max-width: 100%;
}
div[data-testid="stVerticalBlock"] > div:first-child > .desk-banner {
    margin-top: 0 !important;
}
section[data-testid="stSidebar"] > div:first-child { padding-top: 2rem !important; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f8f5ee 0%, #f2ece0 100%);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    border-radius: 16px !important;
    font-weight: 600 !important;
    background: #111111 !important;
    color: #ffffff !important;
    border: none !important;
    min-height: 3rem !important;
    box-shadow: none !important;
}
[data-testid="stMetric"] {
    border: 1px solid var(--line);
    border-radius: 20px;
    padding: 16px 18px;
    background: var(--panel);
    box-shadow: var(--shadow);
}
[data-testid="stMetricLabel"] p {
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted) !important;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    border-bottom: none !important;
    margin-bottom: 10px;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}
.stTabs,
.stTabs > div,
.stTabs [role="tablist"],
.stTabs > div > div {
    background: transparent !important;
    background-color: transparent !important;
}
.stTabs [data-baseweb="tab"] {
    padding: 10px 18px;
    font-size: 0.84rem;
    font-weight: 600;
    color: var(--muted);
    border: 1px solid var(--line);
    background: rgba(255,255,255,0.6);
    border-radius: 999px;
    box-shadow: none !important;
}
.stTabs [aria-selected="true"] {
    color: #ffffff !important;
    border: 1px solid #111111 !important;
    background: #111111 !important;
    box-shadow: none !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-list"]::after,
.stTabs [data-baseweb="tab"]::after,
.stTabs [aria-selected="true"]::after {
    display: none !important;
    border-bottom: none !important;
    box-shadow: none !important;
}
.top-section-tabs-anchor {
    height: 0;
}
div[data-testid="stVerticalBlock"] > div:has(.top-section-tabs-anchor) + div .stTabs [data-baseweb="tab-list"] {
    position: sticky;
    top: 2.95rem;
    z-index: 80;
    padding: 0.5rem 0 0.9rem;
    margin-bottom: 0.6rem;
    background: linear-gradient(180deg, rgba(251, 248, 241, 0.96) 0%, rgba(245, 241, 232, 0.92) 100%) !important;
    backdrop-filter: blur(10px);
    border-bottom: none !important;
    box-shadow: none !important;
}
div[data-testid="stVerticalBlock"] > div:has(.top-section-tabs-anchor) + div .stTabs [data-baseweb="tab-border"] {
    display: none !important;
}
div[data-testid="stVerticalBlock"] > div:has(.top-section-tabs-anchor) + div .stTabs [data-baseweb="tab-list"]::before,
div[data-testid="stVerticalBlock"] > div:has(.top-section-tabs-anchor) + div .stTabs [data-baseweb="tab-list"]::after {
    display: none !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 20px;
    overflow: hidden;
    box-shadow: var(--shadow);
    background: var(--bg);
}
[data-testid="stDataFrame"] iframe {
    background: var(--bg) !important;
}
/* Smaller metric values so text doesn't overflow */
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 1.3rem;
}
.stTextArea [data-baseweb="textarea"],
.stTextArea [data-baseweb="base-input"] {
    border-color: var(--line) !important;
    box-shadow: none !important;
    background: rgba(0,0,0,0) !important;
}
.stTextArea [data-baseweb="textarea"]:focus-within,
.stTextArea [data-baseweb="base-input"]:focus-within {
    border-color: var(--line) !important;
    box-shadow: none !important;
}
.stTextArea textarea {
    border-radius: 18px !important;
    border-color: transparent !important;
    background: rgba(0,0,0,0) !important;
    box-shadow: none !important;
}
.stTextArea textarea:focus {
    border-color: transparent !important;
    box-shadow: none !important;
    outline: none !important;
}
hr { border-color: var(--line) !important; }
.desk-banner,
.desk-card,
.desk-ai-card,
.desk-shell {
    border: 1px solid var(--line);
    background: var(--panel);
    box-shadow: var(--shadow);
}
.desk-banner {
    position: relative;
    overflow: hidden;
    border-radius: 28px;
    padding: 28px 30px;
    margin-bottom: 18px;
}
.desk-banner::after {
    content: "";
    position: absolute;
    inset: auto -40px -70px auto;
    width: 280px;
    height: 280px;
    background: radial-gradient(circle, rgba(18,59,99,0.12), transparent 70%);
}
.desk-kicker {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}
.desk-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.45rem;
    line-height: 1.02;
    color: var(--ink);
    margin-bottom: 8px;
}
.desk-subtitle {
    color: var(--muted);
    max-width: 760px;
    font-size: 0.96rem;
}
.desk-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 7px 12px;
    border-radius: 999px;
    border: 1px solid var(--line);
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--ink);
    background: rgba(255,255,255,0.78);
    margin-right: 8px;
}
.desk-card {
    border-radius: 24px;
    padding: 22px 24px;
    margin-bottom: 16px;
}
.desk-card h3,
.desk-ai-card h3 {
    margin: 0 0 4px 0;
    font-size: 1.02rem;
}
.desk-card-label {
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 8px;
}
.desk-card-value {
    font-family: 'Space Grotesk', sans-serif;
    color: var(--ink);
    font-size: 2.3rem;
    line-height: 1;
}
.desk-card-meta {
    color: var(--muted);
    margin-top: 10px;
    font-size: 0.88rem;
}
.desk-card-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-top: 18px;
}
.desk-micro {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.72);
}
.desk-micro-label {
    font-size: 0.69rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    margin-bottom: 6px;
}
.desk-micro-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    color: var(--ink);
}
.desk-signal-up { background: linear-gradient(180deg, #fbfffc 0%, #eef8f2 100%); }
.desk-signal-down { background: linear-gradient(180deg, #fffaf8 0%, #fdf0eb 100%); }
.desk-signal-neutral { background: linear-gradient(180deg, #fffefc 0%, #f5f1e8 100%); }
.desk-ai-card {
    border-radius: 24px;
    padding: 22px 24px;
    margin-bottom: 16px;
}
.desk-ai-grid {
    display: grid;
    grid-template-columns: 1.1fr 0.9fr;
    gap: 16px;
    align-items: start;
}
.desk-ai-grid > :only-child {
    grid-column: 1 / -1;
}
.desk-driver {
    border-top: 1px solid var(--line);
    padding: 14px 0;
}
.desk-driver:first-child { border-top: none; padding-top: 0; }
.desk-driver-head {
    display: flex;
    justify-content: space-between;
    gap: 16px;
    font-weight: 600;
    color: var(--ink);
    margin-bottom: 4px;
}
.desk-driver-detail {
    color: var(--muted);
    font-size: 0.9rem;
    line-height: 1.45;
}
.desk-driver-score-up { color: var(--up); }
.desk-driver-score-down { color: var(--down); }
.desk-driver-score-flat { color: var(--muted); }
.desk-math {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
}
.desk-math-step {
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 14px 16px;
    background: rgba(255,255,255,0.72);
}
.desk-signal-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
}
.desk-signal-table th {
    padding: 10px 14px;
    text-align: left;
    color: var(--muted);
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    border-bottom: 1px solid var(--line);
}
.desk-signal-table td {
    padding: 9px 14px;
    color: var(--ink);
    border-bottom: 1px solid var(--line);
}
.desk-signal-table tr:last-child td { border-bottom: none; }
.desk-input-table {
    width: 100%;
    border-collapse: collapse;
}
.desk-input-table tr { border-top: 1px solid var(--line); }
.desk-input-table tr:first-child { border-top: none; }
.desk-input-table td {
    padding: 14px 18px;
    vertical-align: top;
    color: var(--ink);
}
.desk-input-table td:first-child {
    width: 42%;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-size: 0.75rem;
    font-weight: 700;
    border-right: 1px solid var(--line);
}
.desk-input-table td:last-child {
    font-size: 0.96rem;
    font-weight: 400;
}
/* Keep slider value and tick bar always visible */
[data-testid="stSliderThumbValue"],
[data-testid="stSliderTickBar"] {
    opacity: 1 !important;
    visibility: visible !important;
}
/* Keep sidebar toggle button always visible */
[data-testid="stSidebarCollapseButton"],
[data-testid="stExpandSidebarButton"] {
    display: flex !important;
    opacity: 1 !important;
    visibility: visible !important;
}
/* News feed panel */
.desk-news-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
}
.desk-news-badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.desk-news-badge-live { background: var(--up-soft); color: var(--up); }
.desk-news-badge-cat { background: var(--brand-soft); color: var(--brand); }
.desk-news-badge-warn { background: var(--amber-soft); color: var(--amber); }
.desk-news-item {
    padding: 10px 0;
    border-bottom: 1px solid var(--line);
}
.desk-news-item:last-child { border-bottom: none; }
.desk-news-title {
    font-weight: 600;
    font-size: 0.88rem;
    color: var(--ink);
    line-height: 1.35;
}
.desk-news-meta {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 3px;
}
.desk-news-relevance {
    display: inline-block;
    width: 40px;
    height: 4px;
    border-radius: 2px;
    background: var(--line);
    margin-left: 6px;
    vertical-align: middle;
    position: relative;
    overflow: hidden;
}
.desk-news-relevance-fill {
    position: absolute;
    left: 0; top: 0; bottom: 0;
    border-radius: 2px;
    background: var(--up);
}
.desk-source-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 6px;
}
.desk-source-chip {
    display: flex;
    align-items: center;
    gap: 6px;
    padding: 6px 10px;
    border-radius: 10px;
    font-size: 0.72rem;
    border: 1px solid var(--line);
    background: var(--panel);
}
.desk-source-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    flex-shrink: 0;
}
.desk-source-dot-ok { background: var(--up); }
.desk-source-dot-fail { background: var(--down); }
@media (max-width: 1080px) {
    .desk-card-row,
    .desk-math,
    .desk-ai-grid {
        grid-template-columns: 1fr;
    }
}
</style>
""", unsafe_allow_html=True)


# ===================================================================
# HELPERS
# ===================================================================

SIGNAL_LOG_PATH = PROJECT_ROOT / "app" / "signal_log.csv"
SETTINGS_STATE_VERSION = 3
SIGNAL_LOG_COLUMNS = [
    "datetime", "signal", "forecast_price", "current_price",
    "predicted_return", "actual_price_2h", "result", "pnl_bps",
    "confidence", "ai_sentiment", "ai_magnitude",
]


def compute_confidence(
    xgb_pred: float,
    lgbm_pred: float,
    ridge_pred: float,
    adjusted_forecast: float,
    threshold: float,
    signal: str,
    ai_event_magnitude: float,
    spread: float,
    live_last: float,
    data_age_minutes: float,
) -> tuple[str, int, dict]:
    """
    Production confidence scorer.
    Returns (label, score_0_100, breakdown_dict).

    Components
    ----------
    1. Model agreement  (0-30)  – std-dev of the 3 model predictions
    2. Signal magnitude (0-30)  – how far past the threshold
    3. Data freshness   (0-15)  – minutes since last real bar
    4. Spread health    (0-10)  – bid-ask tightness relative to price
    5. AI event risk    (0 to -15) – Gemini event_magnitude penalty
    6. Historical edge  (0-15)  – rolling accuracy from signal log
    """
    breakdown: dict[str, float] = {}
    preds = [xgb_pred, lgbm_pred, ridge_pred]

    # 1. Model agreement — low std = high agreement
    pred_std = float(np.std(preds))
    # Normalise: std < 0.0005 → perfect (30), std > 0.004 → zero
    agreement = max(0.0, min(1.0, 1.0 - (pred_std - 0.0005) / 0.0035))
    agreement_pts = round(agreement * 30)
    breakdown["model_agreement"] = agreement_pts

    # 2. Signal magnitude — how many multiples of threshold
    magnitude = abs(adjusted_forecast) / max(threshold, 1e-6)
    if signal == "HOLD":
        mag_pts = 0
    else:
        # 1x threshold → 10 pts, 2x → 20, 3x+ → 30
        mag_pts = round(min(1.0, (magnitude - 1.0) / 2.0) * 30)
    breakdown["signal_magnitude"] = mag_pts

    # 3. Data freshness — 0 min → 15, >120 min → 0
    freshness = max(0.0, min(1.0, 1.0 - data_age_minutes / 120.0))
    freshness_pts = round(freshness * 15)
    breakdown["data_freshness"] = freshness_pts

    # 4. Spread health — spread as bps of price; <50 bps → 10, >200 bps → 0
    spread_bps = (spread / max(live_last, 1.0)) * 10_000
    spread_score = max(0.0, min(1.0, 1.0 - (spread_bps - 50) / 150.0))
    spread_pts = round(spread_score * 10)
    breakdown["spread_health"] = spread_pts

    # 5. AI event risk penalty
    # event_magnitude 0-1; above 0.3 starts penalising
    if ai_event_magnitude > 0.3:
        penalty = min(15.0, (ai_event_magnitude - 0.3) / 0.7 * 15.0)
    else:
        penalty = 0.0
    event_pts = -round(penalty)
    breakdown["ai_event_risk"] = event_pts

    # 6. Historical edge — rolling accuracy from signal log
    hist_pts = 0
    try:
        log = get_signal_log()
        evaluated = log[log["result"].isin(["correct", "wrong"])]
        if len(evaluated) >= 5:
            recent = evaluated.tail(20)
            accuracy = (recent["result"] == "correct").mean()
            # 50% (random) → 0, 70%+ → 15
            hist_pts = round(max(0.0, min(1.0, (accuracy - 0.5) / 0.2)) * 15)
    except Exception:
        pass
    breakdown["historical_edge"] = hist_pts

    raw_score = (
        agreement_pts + mag_pts + freshness_pts
        + spread_pts + event_pts + hist_pts
    )
    score = max(0, min(100, raw_score))
    breakdown["total"] = score

    if signal == "HOLD":
        label = ""
    elif score >= 65:
        label = "HIGH"
    elif score >= 40:
        label = "MEDIUM"
    else:
        label = "LOW"

    return label, score, breakdown


def get_signal_log() -> pd.DataFrame:
    if SIGNAL_LOG_PATH.exists():
        try:
            df = pd.read_csv(SIGNAL_LOG_PATH)
            df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601")
            return df
        except Exception:
            pass
    return pd.DataFrame(columns=SIGNAL_LOG_COLUMNS)


def append_signal_log(row: dict) -> None:
    file_exists = SIGNAL_LOG_PATH.exists()
    with open(SIGNAL_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SIGNAL_LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def update_signal_outcomes(signal_log: pd.DataFrame, current_price: float) -> pd.DataFrame:
    if signal_log.empty:
        return signal_log
    now = datetime.now(UTC)
    updated = False
    for idx, row in signal_log.iterrows():
        if pd.notna(row.get("actual_price_2h")):
            continue
        signal_time = pd.to_datetime(row["datetime"], format="ISO8601")
        if signal_time.tzinfo is None:
            signal_time = signal_time.replace(tzinfo=UTC)
        if (now - signal_time).total_seconds() >= 7200:
            signal_log.at[idx, "actual_price_2h"] = current_price
            entry_price = float(row["current_price"])
            actual_return = (current_price - entry_price) / entry_price
            signal = row["signal"]
            if signal == "HOLD":
                signal_log.at[idx, "result"] = "--"
                signal_log.at[idx, "pnl_bps"] = 0.0
            else:
                was_up = signal == "UP"
                rate_went_up = actual_return > 0
                correct = was_up == rate_went_up
                signal_log.at[idx, "result"] = "correct" if correct else "wrong"
                pnl = abs(actual_return) * 10000 if correct else -abs(actual_return) * 10000
                signal_log.at[idx, "pnl_bps"] = round(pnl, 1)
            updated = True
    if updated:
        signal_log.to_csv(SIGNAL_LOG_PATH, index=False)
    return signal_log


# ---- Background outcome evaluator ----
# A single daemon thread that wakes up, checks for signals that have passed
# their 2h window, fetches the live price from Quidax, and records the outcome.

_outcome_thread_lock = threading.Lock()
_outcome_thread_started = False
_logger = logging.getLogger("outcome_evaluator")


def _outcome_evaluator_loop(settings_runtime_dir: Path, settings_dict: dict):
    """Runs in a background daemon thread. Never returns."""
    settings = get_settings()

    while True:
        try:
            _evaluate_pending_outcomes(settings)
        except Exception as e:
            _logger.warning("Outcome evaluator error: %s", e)
        # Sleep then check again. Adaptive: short sleep if pending, long if idle.
        sleep_seconds = _seconds_until_next_eval()
        time.sleep(max(30, min(sleep_seconds, 300)))


def _seconds_until_next_eval() -> float:
    """Return seconds until the earliest pending signal hits 2h."""
    try:
        if not SIGNAL_LOG_PATH.exists():
            return 300
        df = pd.read_csv(SIGNAL_LOG_PATH)
        df["datetime"] = pd.to_datetime(df["datetime"], format="ISO8601")
        # Find rows with no actual_price_2h
        pending = df[df["actual_price_2h"].isna() | (df["actual_price_2h"] == "")]
        if pending.empty:
            return 300
        now = datetime.now(UTC)
        earliest_due = None
        for _, row in pending.iterrows():
            t = pd.to_datetime(row["datetime"], format="ISO8601")
            if t.tzinfo is None:
                t = t.tz_localize(UTC)
            due_at = t + timedelta(hours=2)
            secs = (due_at - now).total_seconds()
            if earliest_due is None or secs < earliest_due:
                earliest_due = secs
        if earliest_due is not None and earliest_due > 0:
            return earliest_due + 10  # 10s buffer past the 2h mark
        return 30  # Already overdue, check soon
    except Exception:
        return 120


def _evaluate_pending_outcomes(settings):
    """Fetch live price and evaluate all signals past their 2h window."""
    if not SIGNAL_LOG_PATH.exists():
        return
    signal_log = pd.read_csv(SIGNAL_LOG_PATH)
    signal_log["datetime"] = pd.to_datetime(signal_log["datetime"], format="ISO8601")
    now = datetime.now(UTC)
    pending_mask = signal_log["actual_price_2h"].isna() | (signal_log["actual_price_2h"] == "")
    if not pending_mask.any():
        return

    # Check if any are actually past 2h
    due_rows = []
    for idx in signal_log[pending_mask].index:
        t = pd.to_datetime(signal_log.at[idx, "datetime"], format="ISO8601")
        if t.tzinfo is None:
            t = t.tz_localize(UTC)
        if (now - t).total_seconds() >= 7200:
            due_rows.append(idx)

    if not due_rows:
        return

    # Fetch live price once for all due signals
    try:
        quidax = QuidaxTickerService(settings)
        snapshot = quidax.fetch()
        live_price = float(snapshot.usdtngn.last)
    except Exception as e:
        _logger.warning("Could not fetch live price for outcome eval: %s", e)
        return

    for idx in due_rows:
        signal_log.at[idx, "actual_price_2h"] = live_price
        entry_price = float(signal_log.at[idx, "current_price"])
        actual_return = (live_price - entry_price) / entry_price
        signal = signal_log.at[idx, "signal"]
        if signal == "HOLD":
            signal_log.at[idx, "result"] = "--"
            signal_log.at[idx, "pnl_bps"] = 0.0
        else:
            was_up = signal == "UP"
            rate_went_up = actual_return > 0
            correct = was_up == rate_went_up
            signal_log.at[idx, "result"] = "correct" if correct else "wrong"
            pnl = abs(actual_return) * 10000 if correct else -abs(actual_return) * 10000
            signal_log.at[idx, "pnl_bps"] = round(pnl, 1)

    signal_log.to_csv(SIGNAL_LOG_PATH, index=False)
    _logger.info("Evaluated %d signal outcomes at price %.2f", len(due_rows), live_price)


def start_outcome_evaluator(settings: Settings):
    """Start the background outcome evaluator (once per process)."""
    global _outcome_thread_started
    with _outcome_thread_lock:
        if _outcome_thread_started:
            return
        _outcome_thread_started = True
    # Pass settings as dict so it's pickle-safe for the thread
    settings_dict = settings.model_dump() if hasattr(settings, "model_dump") else {}
    t = threading.Thread(
        target=_outcome_evaluator_loop,
        args=(settings.runtime_dir, settings_dict),
        daemon=True,
        name="outcome-evaluator",
    )
    t.start()
    _logger.info("Outcome evaluator thread started")


def fmt(value: float) -> str:
    return f"\u20a6{value:,.2f}"


def fmt_return_pct(value: float) -> str:
    return f"{value * 100:+.3f}%"


def fmt_bps(value: float) -> str:
    return f"{value:+.1f} bps"


def fmt_plain_pct(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"



# ===================================================================
# SESSION STATE
# ===================================================================

def init_services():
    env_path = PROJECT_ROOT / ".env"
    env_mtime = env_path.stat().st_mtime if env_path.exists() else None
    should_reload_settings = (
        "settings" not in st.session_state
        or st.session_state.get("settings_env_mtime") != env_mtime
        or st.session_state.get("settings_state_version") != SETTINGS_STATE_VERSION
    )

    if should_reload_settings:
        st.session_state.settings = reload_settings()
        st.session_state.settings_env_mtime = env_mtime
        st.session_state.settings_state_version = SETTINGS_STATE_VERSION

    if "artifacts" not in st.session_state or should_reload_settings:
        st.session_state.artifacts = ArtifactLoader(st.session_state.settings).load()

    # Start background outcome evaluator (once per Streamlit server process)
    start_outcome_evaluator(st.session_state.settings)
    if "market_notes" not in st.session_state:
        st.session_state.market_notes = ""
    if "notes_history" not in st.session_state:
        st.session_state.notes_history = []
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = None
    if "snapshot" not in st.session_state:
        st.session_state.snapshot = None
    if "ai_result" not in st.session_state:
        st.session_state.ai_result = None
    if "news_service" not in st.session_state:
        st.session_state.news_service = NewsAggregatorService(st.session_state.settings)
    if "news_digest" not in st.session_state:
        st.session_state.news_digest = None


# ===================================================================
# PREDICTION ENGINE (unchanged logic)
# ===================================================================

def run_prediction(settings: Settings, market_notes: str = "") -> dict:
    artifacts = st.session_state.artifacts
    quidax_tickers = QuidaxTickerService(settings)
    external_market = ExternalDailyMarketDataService(settings)
    feature_builder = PublicFeatureBuilder()
    export_loader = ExportLoader(settings)

    live_quotes = quidax_tickers.fetch()
    export_frame = export_loader.load_latest()
    latest_path = export_loader.latest_export_path()
    using_runtime_bars = latest_path.name == settings.runtime_bars_filename

    if using_runtime_bars:
        runtime_frame = export_frame
        synthetic_bars = 0
    else:
        runtime_frame, synthetic_bars = _apply_live_quotes(export_frame, live_quotes, settings)

    export_tail = runtime_frame.tail(settings.feature_lookback_bars).copy()
    start = (export_tail.index.min() - timedelta(days=10)).to_pydatetime()
    end = datetime.now(UTC) + timedelta(days=1)
    market_fetch = external_market.fetch(start=start, end=end)

    feature_result = feature_builder.build(
        export_frame=export_tail,
        external_daily=market_fetch.frame,
        feature_columns=artifacts.feature_columns,
    )
    latest_features = feature_result.features.iloc[[-1]]
    transformed = artifacts.scaler.transform(latest_features)
    transformed_frame = pd.DataFrame(
        transformed, index=latest_features.index, columns=artifacts.feature_columns
    )

    xgb_pred = float(artifacts.xgb_model.predict(transformed_frame)[0])
    lgbm_pred = float(artifacts.lgbm_model.predict(transformed_frame)[0])
    ridge_pred = float(artifacts.ridge_model.predict(transformed)[0])
    raw_forecast = 0.50 * xgb_pred + 0.30 * lgbm_pred + 0.20 * ridge_pred

    live_last = float(live_quotes.usdtngn.last)
    live_bid = float(live_quotes.usdtngn.buy)
    live_ask = float(live_quotes.usdtngn.sell)

    merged = feature_result.merged
    brent_val = float(merged["brent"].iloc[-1]) if "brent" in merged.columns else None
    dxy_val = float(merged["dxy"].iloc[-1]) if "dxy" in merged.columns else None
    vix_val = float(merged["vix"].iloc[-1]) if "vix" in merged.columns else None
    usdghs_val = float(merged["usdghs"].iloc[-1]) if "usdghs" in merged.columns else None
    usdngn_official_val = float(merged["usdngn_official"].iloc[-1]) if "usdngn_official" in merged.columns else None

    btc_premium_pct = None
    if "btc_premium" in latest_features.columns:
        btc_premium_pct = float(latest_features["btc_premium"].iloc[0]) * 100

    close_series = export_tail["close"]
    change_2h_pct = float(close_series.pct_change(1).iloc[-1]) * 100 if len(close_series) > 1 else 0.0
    change_8h_pct = float(close_series.pct_change(4).iloc[-1]) * 100 if len(close_series) > 4 else 0.0
    change_24h_pct = float(close_series.pct_change(12).iloc[-1]) * 100 if len(close_series) > 12 else 0.0

    # Fetch live news for AI context
    news_text = ""
    news_stats: dict = {}
    if settings.news_enabled:
        try:
            news_svc = st.session_state.get("news_service") or NewsAggregatorService(settings)
            news_digest = news_svc.fetch()
            st.session_state.news_digest = news_digest
            news_text = format_news_for_prompt(news_digest)
            news_stats = format_news_summary_stats(news_digest)
        except Exception:
            news_text = ""
            news_stats = {"total": 0, "error": "News fetch failed"}

    # Official-to-parallel spread (calculated before AI call)
    official_parallel_spread = None
    if usdngn_official_val and usdngn_official_val > 0:
        official_parallel_spread = ((live_last - usdngn_official_val) / usdngn_official_val) * 100

    gemini_engine = GeminiAIContextEngine(settings)
    ai_result = gemini_engine.generate(
        current_rate=live_last,
        bid=live_bid,
        ask=live_ask,
        change_2h_pct=change_2h_pct, change_8h_pct=change_8h_pct, change_24h_pct=change_24h_pct,
        brent=brent_val, dxy=dxy_val, vix=vix_val,
        btc_premium_pct=btc_premium_pct,
        usdngn_official=usdngn_official_val,
        official_parallel_spread_pct=official_parallel_spread,
        usdghs=usdghs_val,
        market_notes=market_notes,
        news_headlines=news_text,
    )

    ai_adjustment_return = ai_result.sentiment_score * 0.001
    adjusted_forecast = raw_forecast + ai_adjustment_return

    threshold = float(
        settings.price_threshold
        if settings.price_threshold is not None
        else artifacts.metadata.get("recommended_threshold", 0.003)
    )

    if adjusted_forecast >= threshold:
        signal = "UP"
    elif adjusted_forecast <= -threshold:
        signal = "DOWN"
    else:
        signal = "HOLD"

    forecast_price = live_last * (1 + adjusted_forecast)

    # Data age: minutes since last real bar
    latest_bar_time = export_tail.index.max()
    if latest_bar_time.tzinfo is None:
        latest_bar_time = latest_bar_time.tz_localize(UTC)
    data_age_minutes = (datetime.now(UTC) - latest_bar_time.to_pydatetime()).total_seconds() / 60.0

    confidence, confidence_score, confidence_breakdown = compute_confidence(
        xgb_pred=xgb_pred,
        lgbm_pred=lgbm_pred,
        ridge_pred=ridge_pred,
        adjusted_forecast=adjusted_forecast,
        threshold=threshold,
        signal=signal,
        ai_event_magnitude=ai_result.event_magnitude,
        spread=live_ask - live_bid,
        live_last=live_last,
        data_age_minutes=data_age_minutes,
    )

    price_24h_ago = float(close_series.iloc[-13]) if len(close_series) > 12 else live_last

    chart_bars = min(84, len(export_tail))
    chart_data = export_tail.tail(chart_bars)[["close"]].copy()
    chart_data.index = (
        chart_data.index.tz_convert(WAT)
        if chart_data.index.tzinfo
        else chart_data.index.tz_localize(UTC).tz_convert(WAT)
    )

    brent_week_change = dxy_week_change = ghs_day_change = None

    if "brent" in merged.columns and len(merged) > 84:
        brent_old = float(merged["brent"].iloc[-85]) if merged["brent"].iloc[-85] > 0 else None
        if brent_old and brent_val:
            brent_week_change = ((brent_val - brent_old) / brent_old) * 100

    if "dxy" in merged.columns and len(merged) > 84:
        dxy_old = float(merged["dxy"].iloc[-85]) if merged["dxy"].iloc[-85] > 0 else None
        if dxy_old and dxy_val:
            dxy_week_change = ((dxy_val - dxy_old) / dxy_old) * 100

    if "usdghs" in merged.columns and len(merged) > 12:
        ghs_old = float(merged["usdghs"].iloc[-13])
        if ghs_old > 0 and usdghs_val:
            ghs_day_change = ((usdghs_val - ghs_old) / ghs_old) * 100

    official_spread = None
    if usdngn_official_val and usdngn_official_val > 0:
        official_spread = ((live_last - usdngn_official_val) / usdngn_official_val) * 100

    source_statuses = {}
    for s in live_quotes.statuses:
        source_statuses[s.source_id] = s.status
    for s in market_fetch.statuses:
        source_statuses[s.source_id] = s.status

    return {
        "live_last": live_last, "live_bid": live_bid, "live_ask": live_ask,
        "spread": live_ask - live_bid, "signal": signal, "confidence": confidence,
        "confidence_score": confidence_score, "confidence_breakdown": confidence_breakdown,
        "raw_forecast": raw_forecast, "adjusted_forecast": adjusted_forecast,
        "ai_adjustment_return": ai_adjustment_return,
        "ai_adjustment_bps": ai_adjustment_return * 10000,
        "forecast_price": forecast_price, "threshold": threshold,
        "change_24h_pct": change_24h_pct, "price_24h_ago": price_24h_ago,
        "ai_result": ai_result, "chart_data": chart_data,
        "brent": brent_val, "brent_week_change": brent_week_change,
        "dxy": dxy_val, "dxy_week_change": dxy_week_change,
        "vix": vix_val, "btc_premium_pct": btc_premium_pct,
        "official_spread": official_spread, "ghs_day_change": ghs_day_change,
        "source_statuses": source_statuses, "timestamp": datetime.now(WAT),
        "ai_inputs": {
            "current_rate": live_last,
            "bid": live_bid,
            "ask": live_ask,
            "spread_bps": ((live_ask - live_bid) / max(live_last, 1)) * 10_000,
            "change_2h_pct": change_2h_pct,
            "change_8h_pct": change_8h_pct,
            "change_24h_pct": change_24h_pct,
            "brent": brent_val,
            "dxy": dxy_val,
            "vix": vix_val,
            "btc_premium_pct": btc_premium_pct,
            "usdngn_official": usdngn_official_val,
            "official_parallel_spread_pct": official_parallel_spread,
            "usdghs": usdghs_val,
            "market_notes": market_notes.strip(),
            "news_headlines_count": news_stats.get("total", 0),
            "news_high_relevance": news_stats.get("high_relevance", 0),
            "news_sources_ok": news_stats.get("sources_ok", 0),
            "news_sources_total": news_stats.get("sources_total", 0),
            "news_by_category": news_stats.get("by_category", {}),
        },
    }


def _apply_live_quotes(export_frame, live_quotes, settings):
    frame = export_frame.copy().sort_index()
    latest_export_time = frame.index.max()
    live_time = pd.Timestamp(live_quotes.usdtngn.at).tz_convert(UTC)
    live_bucket = live_time.floor("2h")
    synthetic_bars = 0
    if live_bucket > latest_export_time:
        missing_index = pd.date_range(
            start=latest_export_time + pd.Timedelta(hours=2),
            end=live_bucket, freq="2h", tz=UTC,
        )
        if len(missing_index) > 0:
            filler = pd.DataFrame(
                [frame.iloc[-1].to_dict() for _ in range(len(missing_index))],
                index=missing_index,
            )
            filler.index.name = frame.index.name
            frame = pd.concat([frame, filler])
            synthetic_bars = len(missing_index)
    target_index = live_bucket if live_bucket in frame.index else frame.index.max()
    current = frame.loc[target_index].copy()
    current["open"] = live_quotes.usdtngn.open
    current["high"] = live_quotes.usdtngn.high
    current["low"] = live_quotes.usdtngn.low
    current["close"] = live_quotes.usdtngn.last
    current["volume"] = live_quotes.usdtngn.vol
    current["btcngn_close"] = live_quotes.btcngn.last
    current["btcngn_volume"] = live_quotes.btcngn.vol
    if live_quotes.usdtngn.last > 0:
        current["implied_btcusd_quidax"] = live_quotes.btcngn.last / live_quotes.usdtngn.last
    frame.loc[target_index, current.index] = current.values
    return frame, synthetic_bars


# ===================================================================
# SIDEBAR
# ===================================================================

def render_sidebar():
    with st.sidebar:
        if st.button("Run Prediction", use_container_width=True, type="primary"):
            with st.spinner("Running prediction..."):
                try:
                    result = run_prediction(
                        st.session_state.settings,
                        market_notes=st.session_state.market_notes,
                    )
                    st.session_state.snapshot = result
                    st.session_state.ai_result = result["ai_result"]
                    st.session_state.last_refresh = datetime.now(WAT)
                    _log_signal(result)
                    # Persist full result for dashboard restore on reload
                    try:
                        cache_pkl = st.session_state.settings.runtime_dir / "latest_result.pkl"
                        cache_pkl.write_bytes(pickle.dumps(result))
                    except Exception:
                        pass
                    st.rerun()
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        if st.session_state.last_refresh:
            st.caption(f"Last run: {st.session_state.last_refresh.strftime('%H:%M WAT')}")

        st.divider()

        # Threshold
        st.caption("SIGNAL THRESHOLD")
        threshold_pct = st.slider(
            "Threshold (%)",
            min_value=0.10,
            max_value=1.00,
            value=0.30,
            step=0.05,
            format="%.2f%%",
            label_visibility="collapsed",
        )
        if threshold_pct != 0.30:
            st.session_state.settings.price_threshold = threshold_pct / 100
        else:
            st.session_state.settings.price_threshold = None

        st.divider()

        # Data health
        st.caption("DATA SOURCES")
        if st.session_state.snapshot:
            for src, status in st.session_state.snapshot["source_statuses"].items():
                icon = "\U0001f7e2" if status == "ok" else ("\U0001f7e1" if status == "degraded" else "\U0001f534")
                st.markdown(f"{icon} {src}")
            ai_prov = st.session_state.snapshot["ai_result"].provider
            icon = "\U0001f7e1" if "fallback" in ai_prov else "\U0001f7e2"
            st.markdown(f"{icon} Gemini AI")
        else:
            st.caption("Run a prediction to see status")

        st.divider()

        st.caption("MODEL")
        st.markdown("XGB + LGBM + Ridge")
        st.caption("42 features \u00b7 2h horizon")


# ===================================================================
# DASHBOARD SECTIONS
# ===================================================================

def render_header(result: dict):
    ai = result["ai_result"]
    ai_status = "AI overlay live" if not ai.provider.startswith("fallback") else "AI overlay offline"
    st.markdown(
        f"""
<div class="desk-banner">
    <div class="desk-kicker">Internal Desk Dashboard</div>
    <div class="desk-title">Quidax USD/NGN Intelligence</div>
    <div class="desk-subtitle">
        Live quote, ensemble forecast, and a AI overlay for human intervention.
    </div>
    <div style="margin-top:16px;">
        <span class="desk-pill">{escape(result["timestamp"].strftime("%H:%M WAT"))}</span>
        <span class="desk-pill">{escape(ai_status)}</span>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_rate_and_signal(result: dict):
    signal = result["signal"]
    signal_map = {
        "UP": {
            "tone": "desk-signal-up",
            "label": "Rate likely moving up",
            "action": "Desk bias: lean long USD inventory and widen less aggressively into client demand.",
        },
        "DOWN": {
            "tone": "desk-signal-down",
            "label": "Rate likely moving down",
            "action": "Desk bias: lean long NGN inventory and avoid chasing USD inventory higher.",
        },
        "HOLD": {
            "tone": "desk-signal-neutral",
            "label": "No material move expected",
            "action": "Desk bias: quote normally and use standard spread discipline.",
        },
    }
    signal_ui = signal_map[signal]
    st.markdown(
        f"""
<div class="desk-card {signal_ui["tone"]}" style="margin-bottom:56px;">
    <div class="desk-card-label">Spot + Signal</div>
    <div style="display:flex;justify-content:space-between;gap:24px;align-items:flex-start;flex-wrap:wrap;">
        <div>
            <h3>{escape(signal_ui["label"])}</h3>
            <div class="desk-card-value">{fmt(result["live_last"])}</div>
            <div class="desk-card-meta">
                Bid {fmt(result["live_bid"])} · Ask {fmt(result["live_ask"])} · Spread {fmt(result["spread"])}
            </div>
        </div>
        <div style="max-width:420px;">
            <div class="desk-card-label">Desk Read</div>
            <div style="font-size:1rem;color:var(--ink);font-weight:600;">{escape(signal_ui["action"])}</div>
            <div class="desk-card-meta">
                24h move {fmt_plain_pct(result["change_24h_pct"])} · Confidence {escape(result["confidence"] or "monitor")} ({result.get("confidence_score", 0)}%)
            </div>
        </div>
    </div>
    <div class="desk-card-row">
        <div class="desk-micro">
            <div class="desk-micro-label">Raw Model Return</div>
            <div class="desk-micro-value">{fmt_return_pct(result["raw_forecast"])}</div>
        </div>
        <div class="desk-micro">
            <div class="desk-micro-label">AI Overlay</div>
            <div class="desk-micro-value">{fmt_bps(result["ai_adjustment_bps"])}</div>
        </div>
        <div class="desk-micro">
            <div class="desk-micro-label">Adjusted Return</div>
            <div class="desk-micro-value">{fmt_return_pct(result["adjusted_forecast"])}</div>
        </div>
        <div class="desk-micro">
            <div class="desk-micro-label">2h Forecast Price</div>
            <div class="desk-micro-value">{fmt(result["forecast_price"])}</div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_ai(result: dict):
    ai = result["ai_result"]

    if ai.provider.startswith("fallback"):
        st.info(ai.narrative)
        return

    driver_html = []
    for driver in ai.drivers:
        score_class = "desk-driver-score-flat"
        if driver.score > 0.05:
            score_class = "desk-driver-score-up"
        elif driver.score < -0.05:
            score_class = "desk-driver-score-down"
        driver_html.append(
            f"""
<div class="desk-driver">
    <div class="desk-driver-head">
        <span>{escape(driver.label)}</span>
        <span class="{score_class}">{fmt_bps(driver.score * 10)}</span>
    </div>
    <div class="desk-driver-detail">{escape(driver.detail)}</div>
</div>
"""
        )

    inputs = result["ai_inputs"]
    notes_text = inputs["market_notes"] or "None supplied"
    news_count = inputs.get("news_headlines_count", 0)
    news_hi_rel = inputs.get("news_high_relevance", 0)
    news_src_ok = inputs.get("news_sources_ok", 0)
    news_src_total = inputs.get("news_sources_total", 0)
    news_by_cat = inputs.get("news_by_category", {})

    # -- AI Assessment card --------------------------------------------------
    st.markdown(
        f"""
<div class="desk-ai-grid">
    <div class="desk-ai-card">
        <div class="desk-card-label">AI Assessment</div>
        <h3>Overlay narrative</h3>
        <div style="color:var(--muted);line-height:1.6;margin-bottom:18px;">{escape(ai.narrative)}</div>
        <div class="desk-math">
            <div class="desk-math-step">
                <div class="desk-micro-label">Overlay Score</div>
                <div class="desk-micro-value">{ai.sentiment_score:+.2f}</div>
            </div>
            <div class="desk-math-step">
                <div class="desk-micro-label">Applied to Model</div>
                <div class="desk-micro-value">{fmt_bps(result["ai_adjustment_bps"])}</div>
            </div>
            <div class="desk-math-step">
                <div class="desk-micro-label">Event Magnitude</div>
                <div class="desk-micro-value">{ai.event_magnitude:.2f}</div>
            </div>
            <div class="desk-math-step">
                <div class="desk-micro-label">Threshold</div>
                <div class="desk-micro-value">{fmt_return_pct(result["threshold"])}</div>
            </div>
        </div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # -- Driver Lines card ---------------------------------------------------
    st.markdown(
        f"""
<div class="desk-ai-card">
    <div class="desk-card-label">Driver Lines</div>
    <h3>Component scores used by the AI overlay</h3>
    <div class="desk-card-meta" style="margin-bottom:12px;">
        Positive scores push USD/NGN up. Negative scores push USD/NGN down.
    </div>
    {''.join(driver_html) or '<div class="desk-driver-detail">No driver lines returned.</div>'}
</div>
""",
        unsafe_allow_html=True,
    )

    # -- Full Input Audit Trail (complete transparency) ---------------------
    cat_labels = {
        "cbn_policy": "CBN Policy", "naira_forex": "Naira/FX",
        "oil": "Oil", "global_macro": "Global Macro",
        "nigeria_economy": "NG Economy",
    }

    news_row = ""
    if news_count > 0:
        cat_summary = " / ".join(f"{cat_labels.get(c, c)} {n}" for c, n in sorted(news_by_cat.items(), key=lambda x: x[1], reverse=True))
        news_row = f'<tr><td>Live News</td><td>{news_count} headlines from {news_src_ok}/{news_src_total} sources ({news_hi_rel} high-relevance) — {cat_summary}</td></tr>'
    else:
        news_row = '<tr><td>Live News</td><td>No news data available</td></tr>'

    # Format new fields for display
    bid_val = inputs.get("bid")
    ask_val = inputs.get("ask")
    spread_bps_val = inputs.get("spread_bps")
    official_val = inputs.get("usdngn_official")
    gap_val = inputs.get("official_parallel_spread_pct")
    ghs_val = inputs.get("usdghs")

    spread_display = f"{fmt(bid_val)} / {fmt(ask_val)}" if bid_val and ask_val else "Unavailable"
    if spread_bps_val is not None:
        spread_display += f" ({spread_bps_val:.0f} bps)"
    official_display = f"NGN {official_val:,.2f}" if official_val else "Unavailable"
    gap_display = f"{gap_val:+.2f}%" if gap_val is not None else "Unavailable"
    ghs_display = f"{ghs_val:.2f}" if ghs_val else "Unavailable"

    st.markdown(
        f"""
<div class="desk-ai-card">
    <div class="desk-card-label">Full Input Audit Trail</div>
    <h3>Every input sent to Gemini AI</h3>
    <div class="desk-card-meta" style="margin-bottom:8px;">
        100% transparency — the exact data the AI used. The ML model forecast is NOT shown to Gemini.
        Both signals are independent.
    </div>
    <table class="desk-input-table">
        <tr><td colspan="2" style="font-weight:700;color:var(--brand);font-size:0.72rem;letter-spacing:0.1em;padding:10px 18px;">SPOT MARKET</td></tr>
        <tr><td>Current Rate</td><td>{fmt(inputs["current_rate"])}</td></tr>
        <tr><td>Bid / Ask (Spread)</td><td>{spread_display}</td></tr>
        <tr><td colspan="2" style="font-weight:700;color:var(--brand);font-size:0.72rem;letter-spacing:0.1em;padding:10px 18px;">PRICE MOMENTUM</td></tr>
        <tr><td>2h Change</td><td>{fmt_plain_pct(inputs["change_2h_pct"])}</td></tr>
        <tr><td>8h Change</td><td>{fmt_plain_pct(inputs["change_8h_pct"])}</td></tr>
        <tr><td>24h Change</td><td>{fmt_plain_pct(inputs["change_24h_pct"])}</td></tr>
        <tr><td colspan="2" style="font-weight:700;color:var(--brand);font-size:0.72rem;letter-spacing:0.1em;padding:10px 18px;">MACRO OBSERVABLES</td></tr>
        <tr><td>Brent Crude</td><td>{'$' + format(inputs["brent"], '.2f') if inputs["brent"] is not None else 'Unavailable'}</td></tr>
        <tr><td>Dollar Index (DXY)</td><td>{format(inputs["dxy"], '.2f') if inputs["dxy"] is not None else 'Unavailable'}</td></tr>
        <tr><td>VIX (Fear Index)</td><td>{format(inputs["vix"], '.2f') if inputs["vix"] is not None else 'Unavailable'}</td></tr>
        <tr><td colspan="2" style="font-weight:700;color:var(--brand);font-size:0.72rem;letter-spacing:0.1em;padding:10px 18px;">NIGERIAN FX STRUCTURE</td></tr>
        <tr><td>Official CBN Rate</td><td>{official_display}</td></tr>
        <tr><td>Official-Parallel Gap</td><td>{gap_display}</td></tr>
        <tr><td>BTC Premium (Quidax)</td><td>{fmt_plain_pct(inputs["btc_premium_pct"]) if inputs["btc_premium_pct"] is not None else 'Unavailable'}</td></tr>
        <tr><td colspan="2" style="font-weight:700;color:var(--brand);font-size:0.72rem;letter-spacing:0.1em;padding:10px 18px;">AFRICAN PEER CURRENCIES</td></tr>
        <tr><td>USD/GHS (Ghana)</td><td>{ghs_display}</td></tr>
        <tr><td colspan="2" style="font-weight:700;color:var(--brand);font-size:0.72rem;letter-spacing:0.1em;padding:10px 18px;">HUMAN + NEWS INTELLIGENCE</td></tr>
        <tr><td>Desk Notes</td><td>{escape(notes_text)}</td></tr>
        {news_row}
    </table>
</div>
""",
        unsafe_allow_html=True,
    )

    # -- Live News Feed with source health -----------------------------------
    _render_news_feed()


def _render_news_feed():
    """Show the live news headlines the AI is reading, with full source transparency."""
    digest = st.session_state.get("news_digest")
    if not digest:
        return

    ok_count = sum(1 for s in digest.source_statuses if s.ok)
    total_count = len(digest.source_statuses)
    fetched_str = digest.fetched_at.strftime("%b %d, %H:%M UTC")

    category_labels = {
        "cbn_policy": "CBN Policy", "naira_forex": "Naira/FX",
        "oil": "Oil", "global_macro": "Global Macro",
        "nigeria_economy": "NG Economy",
    }

    # Source health chips HTML
    source_chips = []
    for s in sorted(digest.source_statuses, key=lambda x: (not x.ok, x.name)):
        dot_class = "desk-source-dot-ok" if s.ok else "desk-source-dot-fail"
        latency = f" ({s.latency_ms:.0f}ms)" if s.latency_ms else ""
        count_str = f" — {s.item_count}" if s.ok and s.item_count else ""
        chip_title = escape(s.error) if s.error else ""
        source_chips.append(
            f'<div class="desk-source-chip" title="{chip_title}">'
            f'<span class="desk-source-dot {dot_class}"></span>'
            f'<span>{escape(s.name[:28])}{count_str}{latency}</span></div>'
        )

    # News items HTML (grouped by category)
    news_items_html = []
    categories_order = ["cbn_policy", "naira_forex", "oil", "global_macro", "nigeria_economy"]
    items_by_cat: dict[str, list] = {c: [] for c in categories_order}
    for item in (digest.items or [])[:30]:
        cat = item.category if item.category in items_by_cat else "nigeria_economy"
        items_by_cat[cat].append(item)

    for cat in categories_order:
        cat_items = items_by_cat[cat]
        if not cat_items:
            continue
        cat_label = category_labels.get(cat, cat)
        news_items_html.append(
            f'<div style="margin-top:14px;margin-bottom:6px;">'
            f'<span class="desk-news-badge desk-news-badge-cat">{cat_label} ({len(cat_items)})</span></div>'
        )
        for item in cat_items:
            ts = item.published.strftime("%b %d %H:%M")
            rel_pct = int(item.relevance * 100)
            rel_color = "var(--up)" if item.relevance >= 0.7 else ("var(--amber)" if item.relevance >= 0.4 else "var(--line)")
            news_items_html.append(
                f'<div class="desk-news-item">'
                f'<div class="desk-news-title">{escape(item.title)}</div>'
                f'<div class="desk-news-meta">{escape(item.source)} &bull; {ts}'
                f' <span class="desk-news-relevance"><span class="desk-news-relevance-fill" '
                f'style="width:{rel_pct}%;background:{rel_color}"></span></span>'
                f' {rel_pct}% relevant'
                + (f' &bull; {escape(item.summary[:100])}' if item.summary else '')
                + '</div></div>'
            )

    if not digest.items:
        st.markdown(
            f"""
<div class="desk-ai-card">
    <div class="desk-card-label">Live News Feed</div>
    <h3>No headlines available</h3>
    <div class="desk-card-meta">News sources were queried but returned no recent headlines.</div>
    <div style="margin-top:12px;">
        <div class="desk-source-grid">{''.join(source_chips)}</div>
    </div>
</div>
""",
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
<div class="desk-ai-card">
    <div class="desk-card-label">Live News Feed</div>
    <div class="desk-news-header">
        <h3 style="margin:0;">Headlines the AI is reading</h3>
        <span class="desk-news-badge desk-news-badge-live">LIVE</span>
    </div>
    <div class="desk-card-meta" style="margin-bottom:4px;">
        {len(digest.items)} headlines from {ok_count}/{total_count} sources &bull;
        Fetched {fetched_str} &bull; Cache: 15 min
    </div>
    {''.join(news_items_html)}
</div>
""",
        unsafe_allow_html=True,
    )

    # Source health panel
    failed = [s for s in digest.source_statuses if not s.ok]
    failed_note = ""
    if failed:
        failed_names = ", ".join(s.name[:25] for s in failed)
        failed_note = (
            f'<div class="desk-card-meta" style="margin-top:10px;color:var(--amber);">'
            f'Failed sources ({len(failed)}): {escape(failed_names)}</div>'
        )

    st.markdown(
        f"""
<div class="desk-ai-card">
    <div class="desk-card-label">Source Health</div>
    <h3>News source status</h3>
    <div class="desk-card-meta" style="margin-bottom:10px;">
        {ok_count} of {total_count} sources responded successfully.
        Each source shows item count and response time.
    </div>
    <div class="desk-source-grid">
        {''.join(source_chips)}
    </div>
    {failed_note}
</div>
""",
        unsafe_allow_html=True,
    )


def render_chart(result: dict):
    chart_data = result["chart_data"]
    signal_log = get_signal_log()
    y_min = float(chart_data["close"].min())
    y_max = float(chart_data["close"].max())
    span = max(y_max - y_min, max(y_max * 0.0025, 1.0))
    y_padding = span * 0.35

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=chart_data.index, y=chart_data["close"],
        mode="lines", line=dict(color="#111111", width=2.2),
        hovertemplate="\u20a6%{y:,.2f}<extra></extra>",
    ))

    if not signal_log.empty:
        for _, row in signal_log.iterrows():
            sig_time = pd.to_datetime(row["datetime"], format="ISO8601")
            if sig_time.tzinfo is None:
                sig_time = sig_time.replace(tzinfo=UTC)
            sig_time_wat = sig_time.astimezone(WAT)
            if sig_time_wat < chart_data.index.min() or sig_time_wat > chart_data.index.max():
                continue
            sig = row["signal"]
            res = row.get("result", "")
            mc = "#1e8e3e" if sig == "UP" else ("#d93025" if sig == "DOWN" else "#9aa0a6")
            ms = "triangle-up" if sig == "UP" else ("triangle-down" if sig == "DOWN" else "circle")
            ann = " \u2713" if res == "correct" else (" \u2717" if res == "wrong" else "")
            fig.add_trace(go.Scatter(
                x=[sig_time_wat], y=[float(row["current_price"])],
                mode="markers+text",
                marker=dict(color=mc, size=10, symbol=ms),
                text=[ann], textposition="top center",
                textfont=dict(size=9, color=mc), showlegend=False,
                hovertemplate=f"{sig}{ann}<br>\u20a6%{{y:,.2f}}<extra></extra>",
            ))

    fig.update_layout(
        height=320, margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            tickformat="%b %d\n%H:%M",
            tickfont=dict(size=11, color="#5f6368"),
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.06)",
            zeroline=False,
            tickformat=",.0f",
            tickprefix="\u20a6",
            tickfont=dict(size=11, color="#5f6368"),
            range=[y_min - y_padding, y_max + y_padding],
        ),
        hovermode="x unified", showlegend=False,
        font=dict(family="Inter, sans-serif", color="#202124"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_drivers(result: dict):
    cards = [
        ("Brent crude", f"${result['brent']:.2f}" if result.get("brent") else "N/A", fmt_plain_pct(result.get("brent_week_change")) + " wk" if result.get("brent_week_change") is not None else "No weekly read"),
        ("Dollar index", f"{result['dxy']:.2f}" if result.get("dxy") else "N/A", fmt_plain_pct(result.get("dxy_week_change")) + " wk" if result.get("dxy_week_change") is not None else "No weekly read"),
        ("VIX", f"{result['vix']:.2f}" if result.get("vix") else "N/A", "Calm" if result.get("vix") and result["vix"] < 20 else "Elevated" if result.get("vix") and result["vix"] < 30 else "Risk-off" if result.get("vix") else "No read"),
        ("BTC premium", fmt_plain_pct(result.get("btc_premium_pct")) if result.get("btc_premium_pct") is not None else "N/A", "Quidax vs global"),
        ("Official spread", fmt_plain_pct(result.get("official_spread")) if result.get("official_spread") is not None else "N/A", "Parallel over official"),
        ("Ghana cedi", fmt_plain_pct(result.get("ghs_day_change")) if result.get("ghs_day_change") is not None else "N/A", "1 day regional FX move"),
    ]
    cols = st.columns(3, gap="medium")
    for idx, (label, value, sub) in enumerate(cards):
        with cols[idx % 3]:
            st.markdown(
                f"""
<div class="desk-card">
    <div class="desk-card-label">{escape(label)}</div>
    <div class="desk-micro-value" style="font-size:1.7rem;">{escape(value)}</div>
    <div class="desk-card-meta">{escape(sub)}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def render_history():
    signal_log = get_signal_log()
    if signal_log.empty:
        st.caption("No signals recorded yet.")
        return

    display = signal_log.tail(30).copy().sort_values("datetime", ascending=False)

    # Summary stats
    evaluated = display[display["result"].isin(["correct", "wrong"])]
    directional = display[display["signal"] != "HOLD"]
    holds = display[display["signal"] == "HOLD"]
    pending = directional[~directional["result"].isin(["correct", "wrong", "--"])]
    parts = []
    if not evaluated.empty:
        correct = (evaluated["result"] == "correct").sum()
        parts.append(f"**{correct}/{len(evaluated)}** evaluated")
    if not pending.empty:
        parts.append(f"**{len(pending)}** directional pending")
    parts.append(f"**{len(display)}** total ({len(holds)} HOLD)")
    st.markdown(" · ".join(parts))

    df = display[["datetime", "signal", "confidence", "forecast_price", "current_price", "actual_price_2h", "result", "pnl_bps"]].copy()
    df.columns = ["Time", "Signal", "Conf", "Forecast", "Entry", "Actual", "Result", "PnL"]
    df["Time"] = pd.to_datetime(df["Time"]).dt.strftime("%b %d %H:%M")
    df["Forecast"] = df["Forecast"].apply(lambda x: f"\u20a6{x:,.2f}" if pd.notna(x) else "--")
    df["Entry"] = df["Entry"].apply(lambda x: f"\u20a6{x:,.2f}" if pd.notna(x) else "--")
    df["Actual"] = df["Actual"].apply(lambda x: f"\u20a6{x:,.2f}" if pd.notna(x) else "pending")
    df["Conf"] = df["Conf"].fillna("").apply(lambda x: x if x else "--")
    df["Result"] = df["Result"].apply(lambda x: "\u2713" if x == "correct" else ("\u2717" if x == "wrong" else "--" if x == "--" else "pending"))
    df["PnL"] = df["PnL"].apply(lambda x: f"{x:+.1f}" if pd.notna(x) and x != 0 else "--")

    # Render as HTML table to avoid iframe white background
    rows_html = ""
    for _, r in df.iterrows():
        cells = "".join(f"<td>{escape(str(v))}</td>" for v in r)
        rows_html += f"<tr>{cells}</tr>"
    headers = "".join(f"<th>{escape(c)}</th>" for c in df.columns)
    st.markdown(
        f"""<div class="desk-card" style="padding:0;max-height:420px;overflow-y:auto;">
<table class="desk-signal-table">
<thead style="position:sticky;top:0;background:var(--panel);z-index:1;"><tr>{headers}</tr></thead>
<tbody>{rows_html}</tbody>
</table></div>""",
        unsafe_allow_html=True,
    )


def render_performance():
    signal_log = get_signal_log()
    if signal_log.empty:
        st.caption("No signals logged yet.")
        return

    now = datetime.now(UTC)
    signal_log["datetime"] = pd.to_datetime(signal_log["datetime"], format="ISO8601")
    this_month = signal_log[signal_log["datetime"].dt.month == now.month]
    directional = this_month[this_month["signal"] != "HOLD"]
    evaluated = directional[directional["result"].isin(["correct", "wrong"])]
    pending = directional[~directional["result"].isin(["correct", "wrong", "--"])]

    total = len(this_month)
    n_dir = len(directional)
    n_eval = len(evaluated)
    n_pending = len(pending)
    correct = (evaluated["result"] == "correct").sum() if n_eval > 0 else 0
    acc = (correct / n_eval * 100) if n_eval > 0 else 0
    pnl = evaluated["pnl_bps"].sum() if n_eval > 0 else 0

    # Summary caption first
    if n_eval > 0:
        st.caption(f"Model: {acc:.0f}% vs Random: 50% \u2014 {acc - 50:+.0f}pp edge")
    elif n_pending > 0:
        st.caption(f"{n_pending} directional signal(s) pending 2h outcome evaluation.")
    elif n_dir == 0:
        st.caption("All signals have been HOLD. Directional trades tracked when UP/DOWN signals fire.")

    # 2x2 metric grid
    acc_val = f"{acc:.0f}%" if n_eval > 0 else "awaiting"
    pnl_val = f"{pnl:+.0f}" if n_eval > 0 else "awaiting"
    st.markdown(
        f"""
<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">
    <div class="desk-card" style="padding:14px 18px;">
        <div class="desk-card-label">Total Signals</div>
        <div style="font-size:1.3rem;font-weight:700;color:var(--ink);">{total}</div>
    </div>
    <div class="desk-card" style="padding:14px 18px;">
        <div class="desk-card-label">Directional</div>
        <div style="font-size:1.3rem;font-weight:700;color:var(--ink);">{n_dir} ({n_pending} pending)</div>
    </div>
    <div class="desk-card" style="padding:14px 18px;">
        <div class="desk-card-label">Accuracy</div>
        <div style="font-size:1.3rem;font-weight:700;color:var(--ink);">{acc_val}</div>
    </div>
    <div class="desk-card" style="padding:14px 18px;">
        <div class="desk-card-label">PnL (bps)</div>
        <div style="font-size:1.3rem;font-weight:700;color:var(--ink);">{pnl_val}</div>
    </div>
</div>""",
        unsafe_allow_html=True,
    )


def render_notes():
    st.markdown(
        """
<div class="desk-card" style="margin-bottom:12px;">
    <div class="desk-card-label">Desk Notes</div>
    <h3>Operator context for the next AI pass</h3>
    <div class="desk-card-meta">Use this for fresh headlines, client flow, or intervention rumors you want Gemini to factor in.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    notes = st.text_area(
        "Market notes",
        value=st.session_state.market_notes,
        height=80,
        placeholder='e.g. "Heard CBN may intervene" or "Large client $2M quote"',
        label_visibility="collapsed",
    )
    if notes != st.session_state.market_notes:
        st.session_state.market_notes = notes
    if st.session_state.notes_history:
        with st.expander("Previous notes"):
            for note in reversed(st.session_state.notes_history[-10:]):
                st.caption(f"{note['time']} -- {note['text']}")


def _log_signal(result: dict):
    ai = result["ai_result"]
    append_signal_log({
        "datetime": datetime.now(UTC).isoformat(),
        "signal": result["signal"],
        "forecast_price": round(result["forecast_price"], 2),
        "current_price": round(result["live_last"], 2),
        "predicted_return": round(result["adjusted_forecast"], 6),
        "actual_price_2h": "", "result": "", "pnl_bps": "",
        "confidence": result["confidence"],
        "ai_sentiment": round(ai.sentiment_score, 3),
        "ai_magnitude": round(ai.event_magnitude, 3),
    })
    if st.session_state.market_notes.strip():
        st.session_state.notes_history.append({
            "time": datetime.now(WAT).strftime("%H:%M %b %d"),
            "text": st.session_state.market_notes.strip(),
        })


def _render_event_card(evt: dict) -> str:
    """Return HTML for a single event card."""
    mag = evt["magnitude"]
    if "LARGE" in mag.upper() or "VERY" in mag.upper():
        mag_color = "#c62828"
    elif "MEDIUM" in mag.upper():
        mag_color = "#e65100"
    else:
        mag_color = "#546e7a"
    return (
        '<div class="desk-card" style="margin-bottom:10px;padding:16px 20px;">'
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">'
        f'<span style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;'
        f'color:var(--muted);background:rgba(0,0,0,0.05);padding:2px 8px;border-radius:99px;">'
        f'{escape(evt["category"])}</span>'
        f'<span style="font-size:0.68rem;font-weight:700;letter-spacing:0.08em;text-transform:uppercase;'
        f'color:{mag_color};background:rgba(0,0,0,0.05);padding:2px 8px;border-radius:99px;">'
        f'{escape(mag)}</span></div>'
        f'<div style="font-size:1.05rem;font-weight:700;color:var(--ink);margin-bottom:4px;">'
        f'{escape(evt["event"])}</div>'
        f'<div style="font-size:0.82rem;color:var(--muted);margin-bottom:6px;">'
        f'{escape(evt["frequency"])} · {escape(evt["timing"])}</div>'
        f'<div style="font-size:0.84rem;color:var(--ink);margin-bottom:4px;">'
        f'<b>Direction:</b> {escape(evt["direction"])}</div>'
        f'<div style="font-size:0.84rem;color:var(--muted);margin-bottom:4px;">'
        f'<b>Impact window:</b> {escape(evt["impact_window"])}</div>'
        f'<div style="font-size:0.82rem;color:var(--muted);">'
        f'{escape(evt["mechanism"])}</div></div>'
    )


def render_macro_calendar():
    import calendar as cal_mod
    import json
    from datetime import date
    import streamlit.components.v1 as components

    today = date.today()

    # --- session state ---
    if "cal_year" not in st.session_state:
        st.session_state.cal_year = today.year
    if "cal_month" not in st.session_state:
        st.session_state.cal_month = today.month
    if "cal_selected_day" not in st.session_state:
        st.session_state.cal_selected_day = 0

    year = st.session_state.cal_year
    month = st.session_state.cal_month
    selected_day = st.session_state.cal_selected_day

    # --- header ---
    st.markdown(
        """
<div class="desk-card" style="margin-bottom:16px;">
    <div class="desk-card-label">Macro Event Calendar</div>
    <div style="font-size:0.95rem;color:var(--ink);">
        Recurring events that move USD/NGN. Click any day to view its events.
    </div>
    <div class="desk-card-meta" style="margin-top:6px;">
        <b>UP</b> = Naira weakens (rate rises) · <b>DOWN</b> = Naira strengthens (rate falls)
    </div>
</div>
""",
        unsafe_allow_html=True,
    )

    @st.fragment
    def _render_macro_calendar_panel():
        year = st.session_state.cal_year
        month = st.session_state.cal_month
        selected_day = st.session_state.cal_selected_day

        nav_prev, nav_title, nav_next = st.columns([1, 4, 1])
        with nav_prev:
            if st.button("◀", key="cal_prev", use_container_width=True):
                if month == 1:
                    st.session_state.cal_year = year - 1
                    st.session_state.cal_month = 12
                else:
                    st.session_state.cal_month = month - 1
                st.session_state.cal_selected_day = 0
                st.rerun(scope="fragment")
        with nav_title:
            st.markdown(
                f"<div style='text-align:center;font-size:1.25rem;font-weight:700;padding:6px 0;'>"
                f"{cal_mod.month_name[month]} {year}</div>",
                unsafe_allow_html=True,
            )
        with nav_next:
            if st.button("▶", key="cal_next", use_container_width=True):
                if month == 12:
                    st.session_state.cal_year = year + 1
                    st.session_state.cal_month = 1
                else:
                    st.session_state.cal_month = month + 1
                st.session_state.cal_selected_day = 0
                st.rerun(scope="fragment")

        year = st.session_state.cal_year
        month = st.session_state.cal_month
        month_events = _get_month_events(year, month)
        event_days = {d: evts for d, evts in month_events.items() if d > 0}
        seasonal = month_events.get(0, [])
        is_today_month = month == today.month and year == today.year
        max_day = cal_mod.monthrange(year, month)[1]
        weekday_columns: list[dict[str, object]] = []
        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for weekday_idx, day_name in enumerate(day_names):
            days_for_weekday = []
            for day in range(1, max_day + 1):
                if date(year, month, day).weekday() == weekday_idx:
                    days_for_weekday.append({"day": day, "count": len(event_days.get(day, []))})
            weekday_columns.append({"name": day_name, "days": days_for_weekday})

        payload = {
            "month_abbr": cal_mod.month_abbr[month],
            "today_day": today.day if is_today_month else 0,
            "weekday_columns": weekday_columns,
            "day_cards": {
                str(day): "".join(_render_event_card(evt) for evt in events)
                for day, events in event_days.items()
            },
            "day_counts": {str(day): len(events) for day, events in event_days.items()},
            "seasonal_cards": "".join(_render_event_card(evt) for evt in seasonal),
            "seasonal_names": ", ".join(escape(evt["event"]) for evt in seasonal),
        }

        calendar_html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <style>
    :root {{
      --bg: transparent;
      --panel: #fffdf8;
      --ink: #16202b;
      --muted: #5f6b76;
      --line: #ddd4c3;
      --brand: #123b63;
      --accent: #b4492b;
    }}
    * {{
      box-sizing: border-box;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: 'IBM Plex Sans', sans-serif;
    }}
    .macro-cal {{
      padding-top: 0.65rem;
      padding-bottom: 1.5rem;
    }}
    .macro-grid {{
      display: grid;
      grid-template-columns: repeat(7, minmax(0, 1fr));
      gap: 0.8rem;
      align-items: start;
      margin-bottom: 1.85rem;
    }}
    .macro-col {{
      background: rgba(255, 253, 248, 0.55);
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 0.9rem 0.55rem 1.35rem;
      min-height: 100%;
    }}
    .macro-col-head {{
      text-align: center;
      font-size: 0.68rem;
      font-weight: 700;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: #7a7a7a;
      margin-bottom: 0.55rem;
    }}
    .macro-day {{
      position: relative;
      width: 100%;
      height: 3.35rem;
      margin: 0 0 0.9rem;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: var(--panel);
      color: var(--ink);
      font: inherit;
      font-size: 1rem;
      font-weight: 700;
      cursor: pointer;
      transition: background 120ms ease, border-color 120ms ease, color 120ms ease;
    }}
    .macro-day:hover {{
      background: #f8f4ec;
      border-color: #b8ac97;
    }}
    .macro-day.is-active {{
      background: #16202b;
      border-color: #16202b;
      color: #fff;
    }}
    .macro-day-count {{
      position: absolute;
      top: 0.38rem;
      right: 0.38rem;
      width: 1.15rem;
      height: 1.15rem;
      border-radius: 999px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      background: var(--accent);
      color: #fff;
      font-size: 0.64rem;
      font-weight: 700;
      line-height: 1;
      box-shadow: 0 0 0 2px var(--panel);
    }}
    .macro-day.is-active .macro-day-count {{
      box-shadow: 0 0 0 2px #16202b;
    }}
    .macro-banner {{
      font-size: 0.8rem;
      color: var(--muted);
      padding: 0.95rem 0 1.4rem;
    }}
    .macro-detail {{
      margin-top: 0.5rem;
    }}
    .macro-detail-title {{
      font-size: 1.05rem;
      font-weight: 700;
      margin: 0 0 0.5rem;
      color: var(--ink);
    }}
    .macro-empty {{
      font-size: 0.9rem;
      color: var(--muted);
      padding: 0.75rem 0;
      text-align: center;
    }}
    .desk-card {{
      margin-bottom: 10px;
      padding: 16px 20px;
      border-radius: 18px;
      background: var(--panel);
      border: 1px solid var(--line);
      box-shadow: 0 20px 40px rgba(31, 42, 55, 0.06);
    }}
    @media (max-width: 960px) {{
      .macro-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
    }}
    @media (max-width: 640px) {{
      .macro-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="macro-cal">
    <div id="macro-grid" class="macro-grid"></div>
    <div id="macro-banner" class="macro-banner"></div>
    <div id="macro-detail" class="macro-detail"></div>
  </div>
  <script>
    const payload = {json.dumps(payload)};
    let selectedDay = 0;

    function resizeFrame() {{
      const frame = window.frameElement;
      if (!frame) return;
      frame.style.height = `${{document.documentElement.scrollHeight + 8}}px`;
    }}

    function detailTitle(day, count) {{
      const suffix = count === 1 ? "event" : "events";
      return `${{payload.month_abbr}} ${{day}} - ${{count}} ${{suffix}}`;
    }}

    function renderDetails() {{
      const detail = document.getElementById("macro-detail");
      const banner = document.getElementById("macro-banner");
      banner.innerHTML = payload.seasonal_names
        ? `<b>Active all month:</b> ${{payload.seasonal_names}}`
        : "";

      if (selectedDay) {{
        const cards = payload.day_cards[String(selectedDay)] || "";
        const count = payload.day_counts[String(selectedDay)] || 0;
        if (cards) {{
          detail.innerHTML = `
            <div class="macro-detail-title">${{detailTitle(selectedDay, count)}}</div>
            ${{cards}}
          `;
        }} else {{
          detail.innerHTML = `
            <div class="macro-empty">No scheduled events on ${{payload.month_abbr}} ${{selectedDay}}.</div>
          `;
        }}
      }} else if (payload.seasonal_cards) {{
        detail.innerHTML = `
          <div class="macro-detail-title">Seasonal / All-Month Events</div>
          ${{payload.seasonal_cards}}
        `;
      }} else {{
        detail.innerHTML = "";
      }}

      requestAnimationFrame(resizeFrame);
    }}

    function renderGrid() {{
      const grid = document.getElementById("macro-grid");
      grid.innerHTML = payload.weekday_columns.map((col) => `
        <div class="macro-col">
          <div class="macro-col-head">${{col.name}}</div>
          ${{col.days.map((dayItem) => {{
            const isActive = selectedDay
              ? selectedDay === dayItem.day
              : payload.today_day === dayItem.day;
            return `
              <button
                type="button"
                class="macro-day${{isActive ? " is-active" : ""}}"
                data-day="${{dayItem.day}}"
              >
                <span>${{dayItem.day}}</span>
                ${{dayItem.count > 0 ? `<span class="macro-day-count">${{dayItem.count}}</span>` : ""}}
              </button>
            `;
          }}).join("")}}
        </div>
      `).join("");

      grid.querySelectorAll(".macro-day").forEach((button) => {{
        button.addEventListener("click", () => {{
          const nextDay = Number(button.dataset.day);
          selectedDay = selectedDay === nextDay ? 0 : nextDay;
          renderGrid();
          renderDetails();
        }});
      }});

      requestAnimationFrame(resizeFrame);
    }}

    const resizeObserver = new ResizeObserver(() => resizeFrame());
    resizeObserver.observe(document.body);
    window.addEventListener("resize", resizeFrame);
    window.addEventListener("load", () => {{
      renderGrid();
      renderDetails();
      resizeFrame();
      setTimeout(resizeFrame, 60);
      setTimeout(resizeFrame, 180);
    }});
  </script>
</body>
</html>
"""

        components.html(calendar_html, height=1500, scrolling=False)

    _render_macro_calendar_panel()


# ===================================================================
# METHODOLOGY
# ===================================================================

def render_methodology():
    st.markdown("""
<style>
.method-toc a { color: var(--ink, #2c2c2c); text-decoration: none; }
.method-toc a:hover { text-decoration: underline; }
.method-toc { background: transparent; border-radius: 8px; padding: 18px 24px; margin-bottom: 28px; }
.method-section { margin-bottom: 32px; }
.method-table { width: 100%; border-collapse: collapse; margin: 12px 0 18px 0; font-size: 0.92rem; }
.method-table th { background: transparent; text-align: left; padding: 8px 12px; border-bottom: 2px solid #d4cfc7; font-weight: 600; }
.method-table td { padding: 8px 12px; border-bottom: 1px solid #e8e4dd; }
.method-table tr:last-child td { border-bottom: none; }
</style>

<div class="method-toc">
<strong>Contents</strong><br>
<a href="#what-is-this">1. What Is This?</a><br>
<a href="#data">2. What Data Did We Use?</a><br>
<a href="#features">3. Features Built</a><br>
<a href="#training">4. How We Trained the Model</a><br>
<a href="#findings">5. Findings</a><br>
<a href="#live-system">6. How the Live System Works</a><br>
<a href="#macro-calendar">7. Macro Calendar</a><br>
<a href="#the-application">8. The Application</a>
</div>
""", unsafe_allow_html=True)

    # ── 1. What Is This? ──
    st.markdown('<div id="what-is-this" class="method-section">', unsafe_allow_html=True)
    st.markdown("### 1. What Is This?")
    st.markdown("""
This system predicts the direction of the USDT/NGN exchange rate on Quidax every 2 hours. It tells the OTC team whether
the rate is likely to go **up**, go **down**, or **stay flat** over the next 2 hours. The goal is to help the team
position inventory ahead of moves and improve trading profitability.

It does not give an exact future price. It gives a directional signal with a confidence level, roughly 12 to 14 times
per month, and stays silent the rest of the time.

The system combines two things: a **machine learning model** trained on 6 years of historical trading data, and an
**AI market assessment** that reads the latest news, oil prices, and macro events to adjust the forecast. The two work
together to produce one unified signal.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 2. What Data Did We Use? ──
    st.markdown('<div id="data" class="method-section">', unsafe_allow_html=True)
    st.markdown("### 2. What Data Did We Use?")

    st.markdown("#### 2.1 Internal Data (Quidax Exchange)")
    st.markdown("""
We pulled USDT/NGN trades on the Quidax exchange from January 2020 to March 2026. This included buy and sell, with
the price, volume, buyer, seller, and timestamp.

We grouped these individual trades into 2-hour windows (called bars). Each bar contains the opening, highest, lowest,
and closing price, the total volume traded, how much was buying vs selling, the number of unique traders involved,
the size distribution of trades, and price volatility within the bar.

**Total: 26,252 bars** spanning 6 years and 2 months.

We also pulled BTC/NGN trades from the same exchange. Dividing the BTC/NGN price by the USDT/NGN price gives us
what BTC costs in dollar terms on Quidax. Comparing this to the global BTC/USD price tells us whether there is a
premium or discount on Quidax \u2014 a signal of dollar scarcity in the Nigerian market.
""")

    st.markdown("#### 2.2 External Data (Yahoo Finance)")
    st.markdown("""
<table class="method-table">
<tr><th>Source</th><th>What It Tells Us</th></tr>
<tr><td>Brent Crude Oil (BZ=F)</td><td>Nigeria\u2019s main export earner. When oil falls, USD supply falls, and the naira weakens.</td></tr>
<tr><td>US Dollar Index (DXY)</td><td>Global dollar strength. A stronger dollar puts pressure on all emerging market currencies.</td></tr>
<tr><td>VIX (Fear Index)</td><td>When global markets are scared, money flows out of Africa and into safe havens.</td></tr>
<tr><td>USD/ZAR (South African Rand)</td><td>Regional benchmark. When the Rand weakens, NGN often follows.</td></tr>
<tr><td>USD/GHS (Ghana Cedi)</td><td>West African peer. GHS and NGN share similar capital flow dynamics.</td></tr>
<tr><td>USD/KES (Kenya Shilling)</td><td>East African peer. Provides a broader continental FX context.</td></tr>
<tr><td>BTC/USD (Global)</td><td>Needed to calculate the BTC premium on Quidax.</td></tr>
<tr><td>USD/NGN Official Rate</td><td>The interbank/official rate. The gap between this and the Quidax rate is a market dislocation signal.</td></tr>
</table>

These external sources update daily. We carried each daily value forward into the 2-hour bars using a standard
forward-fill method, which means each bar inherits the most recent daily close.
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 3. Features Built ──
    st.markdown('<div id="features" class="method-section">', unsafe_allow_html=True)
    st.markdown("### 3. Features Built")
    st.markdown("""
Features are the inputs the model looks at to make its prediction. We built **42 features** from the data above,
organised into 6 categories:

**3.1 Momentum and Trend** \u2014 *what direction is the rate moving?*
- Price change over the last 2 hours, 8 hours, 24 hours, 3 days, and 7 days
- 5-day vs 20-day moving average crossover
- RSI (Relative Strength Index): whether the rate is overbought or oversold
- Brent oil and DXY dollar index momentum over 3 and 7 days

**3.2 Cross-Venue Discrepancy** \u2014 *is our price out of line with other markets?*
- BTC Premium: (Quidax implied BTC/USD) / (global BTC/USD) \u2212 1. A premium above 3\u20135% signals acute dollar scarcity.
- BTC premium smoothed over 24 hours, its standard deviation, and its z-score
- Parallel vs Official Spread and its rate of change over 24 hours

**3.3 Volatility** \u2014 *how much is the rate swinging?*
- Realised volatility over 24 hours and 7 days, and the ratio between them
- Average True Range (ATR): the typical 2-hour price swing
- VIX level and 7-day change

**3.4 Calendar and Seasonality** \u2014 *does time of day or month matter?*
- Hour of day and day of week encoded as sine/cosine waves
- Month of year (to capture seasonal patterns like school fee season, FAAC, and Hajj)
- Month-end flag (day \u2265 25) and weekend-adjacent flag

**3.5 Cross-Regional African FX** \u2014 *are other African currencies moving?*
- USD/ZAR, USD/GHS, and USD/KES: 1-day and 7-day rate of change
- Africa FX Composite: the average of all three regional 1-day changes

**3.6 Volume Dynamics**
- Volume change over 2 hours and 24 hours
- Volume relative to its 5-day moving average
""")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 4. How We Trained the Model ──
    st.markdown('<div id="training" class="method-section">', unsafe_allow_html=True)
    st.markdown("### 4. How We Trained the Model")

    st.markdown("""
**What the model predicts:** the percentage change in the USDT/NGN closing price from now to 2 hours from now.
For example, if the current rate is \u20a61,415 and the model predicts +0.35%, that implies a rate of approximately
\u20a61,420 in 2 hours.
""")

    st.markdown("#### 4.1 The Ensemble (Three Models Working Together)")
    st.markdown("""
<table class="method-table">
<tr><th>Model</th><th>Weight</th><th>Why</th></tr>
<tr><td>XGBoost</td><td>50%</td><td>A tree-based model good at finding non-linear patterns. The main workhorse.</td></tr>
<tr><td>LightGBM</td><td>30%</td><td>Similar to XGBoost but builds trees differently. Provides diversity.</td></tr>
<tr><td>Ridge Regression</td><td>20%</td><td>A simple linear model. Acts as a stabiliser when the tree models disagree.</td></tr>
</table>

**Final prediction:** Forecast = (0.50 \u00d7 XGBoost) + (0.30 \u00d7 LightGBM) + (0.20 \u00d7 Ridge)

Using three models reduces the risk of any single model making a bad call.
""", unsafe_allow_html=True)

    st.markdown("#### 4.2 Recency Weighting")
    st.markdown("""
The model was trained on data from 2020 to 2026, but the market in 2020\u20132021 behaved very differently from today.
To ensure the model prioritises recent market behaviour, we applied **exponential decay weighting with a 180-day
half-life**:
- Data from the last 6 months has full weight (1.0)
- Data from 1 year ago has half weight (0.5)
- Data from 2020 has minimal weight (~0.05)
""")

    st.markdown("#### 4.3 Validation Method")
    st.markdown("""
We tested the model using **walk-forward cross-validation** with 5 time-ordered folds. For each fold: train on
everything before it, skip a 24-hour gap (to prevent information leakage), then test on the unseen chunk.

We also held out the final 15% of data (approximately April 2025 to March 2026) as a completely untouched holdout set.
Hyperparameters were tuned using **Optuna** (50 trials) optimising for profit at the 0.30% production threshold.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 5. Findings ──
    st.markdown('<div id="findings" class="method-section">', unsafe_allow_html=True)
    st.markdown("### 5. Findings")

    st.markdown("#### 5.1 Cross-Validation Results (5 Folds)")
    st.markdown("""
<table class="method-table">
<tr><th>Fold</th><th>Period</th><th>Accuracy (All)</th><th>Accuracy (@30bps)</th><th>Net PnL (bps)</th><th>Trades</th></tr>
<tr><td>1</td><td>Oct 2023 \u2013 Apr 2024</td><td>55.7%</td><td>62.5%</td><td>+34,545</td><td>811</td></tr>
<tr><td>2</td><td>Apr 2024 \u2013 Sep 2024</td><td>61.2%</td><td>63.7%</td><td>+6,385</td><td>466</td></tr>
<tr><td>3</td><td>Sep 2024 \u2013 Mar 2025</td><td>56.0%</td><td>67.7%</td><td>+1,613</td><td>96</td></tr>
<tr><td>4</td><td>Mar 2025 \u2013 Sep 2025</td><td>58.4%</td><td>50.0%</td><td>+1,130</td><td>22</td></tr>
<tr><td>5</td><td>Sep 2025 \u2013 Mar 2026</td><td>56.1%</td><td>72.7%</td><td>+112</td><td>11</td></tr>
</table>

All 5 folds were profitable. The number of tradeable signals decreased sharply in recent folds as the market became
calmer \u2014 the model correctly generated fewer signals.
""", unsafe_allow_html=True)

    st.markdown("#### 5.2 Holdout Results (April 2025 \u2013 March 2026)")
    st.markdown("""
This is the ultimate test. The model was evaluated on 11 months of data it never saw during training:

<table class="method-table">
<tr><th>Metric</th><th>Value</th></tr>
<tr><td>Overall directional accuracy</td><td>57.3%</td></tr>
<tr><td>Accuracy at production threshold (0.30%)</td><td>76.9%</td></tr>
<tr><td>Net profit after costs</td><td>+1,797 basis points</td></tr>
<tr><td>Trades at 0.30% threshold</td><td>26</td></tr>
<tr><td>Model edge over random</td><td>+7.3 percentage points</td></tr>
</table>

During the holdout period, the rate went up 49.9% of the time and down 46.5% of the time \u2014 essentially a coin flip.
The model scored 57.3%, a **7.3 percentage point edge** over always predicting up.
""", unsafe_allow_html=True)

    st.markdown("#### 5.3 What Drives the Predictions?")
    st.markdown("""
<table class="method-table">
<tr><th>Category</th><th>Share of Model Importance</th></tr>
<tr><td>Momentum and Trend</td><td>38.3% \u2014 recent price action is the strongest signal</td></tr>
<tr><td>Volatility</td><td>23.8% \u2014 the model uses volatility to decide when to trade</td></tr>
<tr><td>Cross-Venue Discrepancy</td><td>18.0% \u2014 BTC premium and parallel-official spread</td></tr>
<tr><td>Regional African FX</td><td>10.4% \u2014 Ghana cedi is a top-5 predictor</td></tr>
<tr><td>Calendar and Seasonality</td><td>9.5% \u2014 time of day and day of week matter</td></tr>
</table>

**Key finding:** the Ghana cedi (USD/GHS) is the 2nd and 5th most important individual feature.
NGN and GHS move together because they share West African capital flow dynamics.
""", unsafe_allow_html=True)

    st.markdown("#### 5.4 Longer-Term Predictions")
    st.markdown("""
<table class="method-table">
<tr><th>Horizon</th><th>Accuracy</th><th>Edge Over Random</th></tr>
<tr><td>2 hours</td><td>52.3%</td><td>+2.4% (real edge)</td></tr>
<tr><td>8 hours</td><td>50.6%</td><td>+0.5% (noise)</td></tr>
<tr><td>24 hours</td><td>48.0%</td><td>\u22120.1% (no edge)</td></tr>
<tr><td>3 days</td><td>44.9%</td><td>+0.2% (noise)</td></tr>
<tr><td>7 days</td><td>44.2%</td><td>+0.4% (noise)</td></tr>
</table>

The model has predictive power only at the 2-hour horizon. Beyond that, it performs no better than random.
""", unsafe_allow_html=True)

    st.markdown("#### 5.5 Limitations")
    st.markdown("""
- The model generates roughly **12\u201314 high-conviction signals per month**. The rest of the time it says nothing useful.
- In calm, range-bound markets, the model produces very few signals (as few as 3 per quarter).
- The model **cannot predict** policy shocks, CBN interventions, or sudden geopolitical events.
- Performance will degrade over time as market conditions change. The model will need periodic retraining.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 6. How the Live System Works ──
    st.markdown('<div id="live-system" class="method-section">', unsafe_allow_html=True)
    st.markdown("### 6. How the Live System Works")

    st.markdown("#### Step 1: Fetch Live Quidax Prices")
    st.markdown("""
The system calls two Quidax API endpoints (no API key required) for USDT/NGN and BTC/NGN tickers. Each returns the
current bid, ask, last traded price, 24-hour high/low/open, and volume. The system uses the \u2018last\u2019 field as the
current price (this matches the \u2018close\u2019 field the model was trained on).
""")

    st.markdown("#### Step 2: Fetch External Market Data")
    st.markdown("""
The system fetches daily prices from Yahoo Finance for the same 8 external indicators used during training (Brent, DXY,
VIX, USD/ZAR, USD/GHS, USD/KES, BTC/USD, and the official USD/NGN rate). Weekend and holiday gaps are handled by
forward-filling the last known value, exactly as in training.
""")

    st.markdown("#### Step 3: Scrape Live News")
    st.markdown("""
The system runs a live news aggregator that scrapes **20 sources concurrently** \u2014 a component the ML model cannot see:
- **Google News RSS:** 11 targeted queries covering naira/forex, CBN policy, oil prices, OPEC, Nigeria economy,
  Federal Reserve, and emerging market currencies
- **Nigerian financial media:** Nairametrics, BusinessDay NG, Vanguard, ThisDay Live, Premium Times, and Punch NG
- **CBN Press Releases:** scraped directly from cbn.gov.ng, filtered for policy-relevant keywords
- **International sources:** OilPrice.com RSS and IMF News RSS

Each headline is relevance-scored (0.0 to 1.0), auto-categorised into 5 categories (CBN/Monetary Policy, Naira/Forex,
Oil/Commodities, Global Macro, Nigeria Economy), de-duplicated using Jaccard similarity (>55% = duplicate), and numbered
with headline IDs (H1, H2, H3\u2026) so the AI can cite specific headlines.
""")

    st.markdown("#### Step 4: Compute Features and Run the ML Model")
    st.markdown("""
Using the live data plus a rolling buffer of recent bars, the system computes the exact same 42 features used during
training, in the same order. These are normalised using the same scaler from training, then fed into the three models:

**Raw Forecast = (0.50 \u00d7 XGBoost) + (0.30 \u00d7 LightGBM) + (0.20 \u00d7 Ridge)**
""")

    st.markdown("#### Step 5: AI Market Assessment (Gemini)")
    st.markdown("""
The system calls Google Gemini to produce an **independent** market assessment. The critical design decision:
**the AI does not see the ML model\u2019s forecast**. It is explicitly told: *\u201cYou have NO access to the ML model\u2019s
forecast. Your job is to assess what the quantitative model might be MISSING.\u201d*

The AI receives current prices, recent price changes, macro data, BTC premium, official-parallel spread, desk notes
from the sales team, and the 25 most relevant live news headlines. It returns:
- **Sentiment Score** (\u22121.0 to +1.0): overall directional lean
- **Event Magnitude** (0.0 to 1.0): how uncertain the environment is
- **Narrative:** 3\u20135 sentences explaining the situation with cited headline IDs
- **Drivers:** 3\u20136 individual forces acting on the rate, each scored with evidence
""")

    st.markdown("#### Step 6: Combine ML and AI")
    st.markdown("""
The AI sentiment adjusts the raw ML forecast:

**Adjusted Forecast = Raw ML Forecast + (AI Sentiment Score \u00d7 0.001)**

This is deliberately a **small nudge, not an override**. The AI can contribute at most 10 bps to the forecast, while the
production threshold is 30 bps. The AI can tip a borderline HOLD into a signal if the news supports it, or dampen a weak
signal if the news contradicts it \u2014 but it cannot single-handedly generate a strong signal.
""")

    st.markdown("#### Step 7: Generate the Signal")
    st.markdown("""
The adjusted forecast is compared to the production threshold (default **0.30%**):
- **+0.30% or higher** \u2192 Signal is **UP**. Rate expected to rise.
- **\u22120.30% or lower** \u2192 Signal is **DOWN**. Rate expected to fall.
- **Between \u22120.30% and +0.30%** \u2192 Signal is **HOLD**. No significant move expected.

A **confidence score** (0\u2013100) is computed from 6 factors: model agreement (how closely the three models align),
signal magnitude (how far past the threshold), data freshness, spread health, AI event risk, and historical track record.
This maps to HIGH, MEDIUM, or LOW for display.
""")

    st.markdown("#### Step 8: Log, Track, and Auto-Evaluate")
    st.markdown("""
Every signal is logged with: timestamp, signal direction, forecast price, current price, predicted return, confidence,
AI sentiment, and AI event magnitude.

When a signal becomes 2 hours old, the system fetches the actual price and records whether the prediction was correct
or wrong, along with the profit or loss in basis points.
""")
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 7. Macro Calendar ──
    st.markdown('<div id="macro-calendar" class="method-section">', unsafe_allow_html=True)
    st.markdown("### 7. Macro Calendar")
    st.markdown("""
The system includes a built-in calendar of recurring events known to affect the USD/NGN rate:

<table class="method-table">
<tr><th>Event</th><th>Frequency</th><th>Typical Impact</th></tr>
<tr><td>CBN Monetary Policy Committee (MPC)</td><td>6x/year</td><td>LARGE. Hike = naira strengthens.</td></tr>
<tr><td>CBN FX Interventions / SMIS Auctions</td><td>Weekly</td><td>MEDIUM-LARGE. Large auction = naira strengthens.</td></tr>
<tr><td>Nigerian CPI / Inflation</td><td>Monthly (~15th)</td><td>SMALL-MEDIUM. High inflation erodes naira purchasing power.</td></tr>
<tr><td>FAAC Monthly Distribution</td><td>Monthly (~20th\u201325th)</td><td>SMALL-MEDIUM. Injects naira liquidity.</td></tr>
<tr><td>Company Income Tax / PPT Deadlines</td><td>Quarterly/Annual</td><td>SMALL-MEDIUM. Multinationals convert USD to NGN for tax.</td></tr>
<tr><td>US Federal Reserve Decision</td><td>8x/year</td><td>MEDIUM. Rate hike = dollar strengthens = naira weakens.</td></tr>
<tr><td>US Non-Farm Payrolls</td><td>Monthly (1st Friday)</td><td>SMALL-MEDIUM. Strong jobs = Fed hawkish = dollar up.</td></tr>
<tr><td>OPEC+ Production Decision</td><td>Monthly/Quarterly</td><td>MEDIUM. Output cut = oil up = more USD inflow to Nigeria.</td></tr>
<tr><td>School Fee Season</td><td>Aug\u2013Sep, Jan</td><td>SMALL. Dollar demand spike for overseas tuition.</td></tr>
<tr><td>Hajj Season</td><td>Varies (Islamic calendar)</td><td>SMALL. Dollar demand for pilgrimage travel.</td></tr>
</table>
""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── 8. The Application ──
    st.markdown('<div id="the-application" class="method-section">', unsafe_allow_html=True)
    st.markdown("### 8. The Application")
    st.markdown("""
The dashboard is designed for non-technical users. Key elements:

- **Current rate** in large text with bid, ask, spread, and 24-hour change
- **Signal banner**: green for UP, red for DOWN, grey for HOLD \u2014 with a plain-English action recommendation
- **AI market narrative** explaining the current situation with cited evidence
- **AI drivers**: 3\u20136 individual forces acting on the rate, each scored with evidence
- **Sentiment bar** from bearish (red) through neutral (grey) to bullish (green)
- **Market notes input** where the team can type intelligence (e.g. \u201cLarge client asking for $2M quote\u201d) \u2014 included in the next AI assessment
- **Market drivers panel** showing oil, DXY, VIX, Ghana cedi, BTC premium, and official-parallel spread
- **7-day price chart** with historical 2-hour bars
- **Signal history table** showing past signals with outcomes, accuracy, and PnL
- **Performance summary** with monthly accuracy and comparison to random (50%)
- **Macro calendar** listing upcoming events that could affect the rate
- **Data health indicators** showing green/yellow/red status for each data source
""")
    st.markdown('</div>', unsafe_allow_html=True)


# ===================================================================
# MAIN
# ===================================================================

def main():
    init_services()
    render_sidebar()

    result = st.session_state.snapshot

    if result is None:
        # Try to restore the full dashboard from the last prediction
        cache_pkl = st.session_state.settings.runtime_dir / "latest_result.pkl"
        if cache_pkl.exists():
            try:
                result = pickle.loads(cache_pkl.read_bytes())
                st.session_state.snapshot = result
                st.session_state.ai_result = result["ai_result"]
                st.session_state.last_refresh = result.get("timestamp")
            except Exception:
                result = None

    # Update outcomes
    if result is not None:
        signal_log = get_signal_log()
        if not signal_log.empty:
            update_signal_outcomes(signal_log, result["live_last"])

    st.markdown('<div class="top-section-tabs-anchor"></div>', unsafe_allow_html=True)
    top_prediction, top_macro, top_method = st.tabs(["Prediction", "Macro Calendar", "Methodology"])

    with top_prediction:
        if result is None:
            st.markdown(
                """
<div class="desk-banner">
    <div class="desk-kicker">Ready</div>
    <div class="desk-title">Generate a live desk read</div>
    <div class="desk-subtitle">
        Run a prediction from the sidebar to populate the dashboard with spot pricing, ensemble output,
        AI overlay drivers, and signal tracking.
    </div>
</div>
""",
                unsafe_allow_html=True,
            )
        else:
            render_header(result)
            render_rate_and_signal(result)

            tab_ai, tab_review, tab_chart, tab_drivers, tab_notes = st.tabs([
                "AI Overlay", "Signal Review", "Price Tape", "Market Context", "Desk Notes"
            ])
            with tab_ai:
                render_ai(result)
            with tab_review:
                # Confidence breakdown
                bd = result.get("confidence_breakdown", {})
                if bd:
                    st.markdown("##### Confidence Breakdown")
                    cb1, cb2, cb3, cb4, cb5, cb6 = st.columns(6)
                    cb1.metric("Agreement", f"{bd.get('model_agreement', 0)}/30")
                    cb2.metric("Magnitude", f"{bd.get('signal_magnitude', 0)}/30")
                    cb3.metric("Freshness", f"{bd.get('data_freshness', 0)}/15")
                    cb4.metric("Spread", f"{bd.get('spread_health', 0)}/10")
                    cb5.metric("Event Risk", f"{bd.get('ai_event_risk', 0)}")
                    cb6.metric("Track Record", f"{bd.get('historical_edge', 0)}/15")
                    st.caption(
                        f"Composite score: **{bd.get('total', 0)}**/100 → "
                        f"**{result.get('confidence') or 'monitor'}**"
                    )
                    st.divider()
                col_l, col_r = st.columns(2)
                with col_l:
                    st.markdown("##### Signal History")
                    render_history()
                with col_r:
                    st.markdown("##### This Month")
                    render_performance()
            with tab_chart:
                render_chart(result)
            with tab_drivers:
                render_drivers(result)
            with tab_notes:
                render_notes()

    with top_macro:
        render_macro_calendar()

    with top_method:
        render_methodology()


if __name__ == "__main__":
    main()
else:
    main()
