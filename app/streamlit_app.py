from __future__ import annotations

import csv
import os
import sys
import time
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import Settings, get_settings
from app.schemas import InferenceSnapshot
from app.services.artifacts import ArtifactLoader, ExportLoader
from app.services.gemini_ai import GeminiAIContextEngine
from app.services.features import PublicFeatureBuilder
from app.services.market_data import ExternalDailyMarketDataService, QuidaxTickerService

# ---------------------------------------------------------------------------
# WAT timezone (West Africa Time = UTC+1)
# ---------------------------------------------------------------------------
WAT = timezone(timedelta(hours=1))

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Quidax OTC Rate Intelligence",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS -- white & black theme, clean sans-serif
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

html, body, [class*="stApp"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #ffffff;
    color: #111111;
}

/* Hide default Streamlit chrome */
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Override streamlit element styling */
h1, h2, h3, h4 { font-family: 'Inter', sans-serif; }

/* Rate display */
.rate-massive {
    font-size: 3.8rem;
    font-weight: 900;
    letter-spacing: -0.02em;
    line-height: 1.1;
    color: #111111;
    margin: 0;
    padding: 0;
}
.rate-sub {
    font-size: 0.95rem;
    color: #666666;
    margin: 4px 0 0 0;
}
.rate-change-up { color: #16a34a; font-weight: 600; }
.rate-change-down { color: #dc2626; font-weight: 600; }
.rate-timestamp {
    font-size: 0.8rem;
    color: #999999;
    margin-top: 6px;
}

/* Signal banners */
.signal-banner {
    border-radius: 16px;
    padding: 28px 32px;
    margin: 8px 0;
    text-align: center;
}
.signal-up {
    background: #f0fdf4;
    border: 2px solid #16a34a;
}
.signal-down {
    background: #fef2f2;
    border: 2px solid #dc2626;
}
.signal-hold {
    background: #f5f5f5;
    border: 2px solid #d4d4d4;
}
.signal-icon {
    font-size: 2.5rem;
    margin-bottom: 8px;
}
.signal-text {
    font-size: 1.35rem;
    font-weight: 700;
    margin: 8px 0 4px 0;
}
.signal-sub {
    font-size: 0.95rem;
    color: #555555;
    margin: 4px 0;
}
.signal-action {
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 12px;
    padding: 8px 16px;
    border-radius: 8px;
    display: inline-block;
}
.action-up { background: #dcfce7; color: #15803d; }
.action-down { background: #fee2e2; color: #b91c1c; }
.action-hold { background: #e5e5e5; color: #525252; }
.signal-forecast {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 12px 0 4px 0;
}
.signal-confidence {
    font-size: 0.85rem;
    color: #666666;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.signal-neutral-band {
    font-size: 0.9rem;
    color: #737373;
    margin-top: 8px;
}

/* AI Assessment Card */
.ai-card {
    background: #fafafa;
    border: 1px solid #e5e5e5;
    border-radius: 14px;
    padding: 20px 24px;
    margin: 8px 0;
}
.ai-narrative {
    font-size: 0.95rem;
    line-height: 1.65;
    color: #333333;
}
.ai-unavailable {
    color: #999999;
    font-style: italic;
}

/* Sentiment bar */
.sentiment-bar-container {
    margin: 12px 0 4px 0;
}
.sentiment-bar-track {
    height: 10px;
    border-radius: 5px;
    background: linear-gradient(to right, #dc2626, #d4d4d4, #16a34a);
    position: relative;
}
.sentiment-bar-marker {
    width: 14px;
    height: 14px;
    border-radius: 50%;
    background: #111111;
    border: 2px solid #ffffff;
    position: absolute;
    top: -2px;
    transform: translateX(-50%);
    box-shadow: 0 1px 3px rgba(0,0,0,0.3);
}

/* Event alert */
.event-alert {
    background: #fefce8;
    border: 1px solid #facc15;
    border-radius: 10px;
    padding: 10px 16px;
    font-size: 0.88rem;
    color: #854d0e;
    margin: 10px 0;
}

/* Market driver cards */
.driver-card {
    background: #fafafa;
    border: 1px solid #e5e5e5;
    border-radius: 12px;
    padding: 14px 16px;
    margin: 4px 0;
}
.driver-label {
    font-size: 0.78rem;
    color: #999999;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin: 0 0 4px 0;
}
.driver-value {
    font-size: 1.15rem;
    font-weight: 700;
    color: #111111;
    margin: 0;
}
.driver-change {
    font-size: 0.82rem;
    margin: 2px 0 0 0;
}

/* Countdown */
.countdown {
    font-size: 0.85rem;
    color: #999999;
    text-align: center;
    padding: 8px 0;
}

/* Signal history table */
.history-accuracy {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 8px;
}

/* Performance card */
.perf-stat {
    text-align: center;
    padding: 16px;
}
.perf-number {
    font-size: 2rem;
    font-weight: 800;
    color: #111111;
}
.perf-label {
    font-size: 0.8rem;
    color: #888888;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Section headers */
.section-header {
    font-size: 0.78rem;
    font-weight: 600;
    color: #999999;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin: 24px 0 8px 0;
    padding-bottom: 4px;
    border-bottom: 1px solid #e5e5e5;
}

/* Override streamlit metric styling */
[data-testid="stMetric"] {
    background: #fafafa;
    border: 1px solid #e5e5e5;
    border-radius: 12px;
    padding: 16px;
}

/* Sidebar overrides */
[data-testid="stSidebar"] {
    background: #111111;
    color: #ffffff;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #ffffff !important;
}
</style>
""",
    unsafe_allow_html=True,
)


# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

SIGNAL_LOG_PATH = PROJECT_ROOT / "app" / "signal_log.csv"
SIGNAL_LOG_COLUMNS = [
    "datetime", "signal", "forecast_price", "current_price",
    "predicted_return", "actual_price_2h", "result", "pnl_bps",
    "confidence", "ai_sentiment", "ai_magnitude",
]


def get_signal_log() -> pd.DataFrame:
    if SIGNAL_LOG_PATH.exists():
        try:
            df = pd.read_csv(SIGNAL_LOG_PATH)
            df["datetime"] = pd.to_datetime(df["datetime"])
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
    """Check past signals that are now 2+ hours old and record outcomes."""
    if signal_log.empty:
        return signal_log
    now = datetime.now(UTC)
    updated = False
    for idx, row in signal_log.iterrows():
        if pd.notna(row.get("actual_price_2h")):
            continue
        signal_time = pd.to_datetime(row["datetime"])
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


def format_naira(value: float) -> str:
    return f"N{value:,.2f}"


def compute_next_cycle_time() -> datetime:
    """Compute the next 2-hour boundary during trading hours (8AM-10PM WAT)."""
    now_wat = datetime.now(WAT)
    hour = now_wat.hour
    next_even_hour = hour + (2 - hour % 2)
    if next_even_hour > 22:
        next_day = now_wat.date() + timedelta(days=1)
        return datetime(next_day.year, next_day.month, next_day.day, 8, 0, tzinfo=WAT)
    return now_wat.replace(hour=next_even_hour, minute=0, second=0, microsecond=0)


def get_countdown_text() -> str:
    next_cycle = compute_next_cycle_time()
    now = datetime.now(WAT)
    diff = next_cycle - now
    if diff.total_seconds() <= 0:
        return "Prediction cycle running now..."
    hours = int(diff.total_seconds() // 3600)
    minutes = int((diff.total_seconds() % 3600) // 60)
    if hours > 0:
        return f"Next prediction in: {hours}h {minutes}m"
    return f"Next prediction in: {minutes}m"


# ===================================================================
# INITIALIZATION / SESSION STATE
# ===================================================================

def init_services():
    if "settings" not in st.session_state:
        st.session_state.settings = get_settings()
    if "artifacts" not in st.session_state:
        settings = st.session_state.settings
        st.session_state.artifacts = ArtifactLoader(settings).load()
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


def run_prediction(settings: Settings, market_notes: str = "") -> dict:
    """Run the full prediction cycle and return all results."""
    artifacts = st.session_state.artifacts
    quidax_tickers = QuidaxTickerService(settings)
    external_market = ExternalDailyMarketDataService(settings)
    feature_builder = PublicFeatureBuilder()
    export_loader = ExportLoader(settings)

    # Fetch live quotes
    live_quotes = quidax_tickers.fetch()

    # Load runtime bars
    export_frame = export_loader.load_latest()
    latest_path = export_loader.latest_export_path()
    using_runtime_bars = latest_path.name == settings.runtime_bars_filename

    if using_runtime_bars:
        runtime_frame = export_frame
        synthetic_bars = 0
    else:
        runtime_frame, synthetic_bars = _apply_live_quotes(export_frame, live_quotes, settings)

    export_tail = runtime_frame.tail(settings.feature_lookback_bars).copy()

    # Fetch external market data
    start = (export_tail.index.min() - timedelta(days=10)).to_pydatetime()
    end = datetime.now(UTC) + timedelta(days=1)
    market_fetch = external_market.fetch(start=start, end=end)

    # Build features
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

    # Run ensemble
    xgb_pred = float(artifacts.xgb_model.predict(transformed_frame)[0])
    lgbm_pred = float(artifacts.lgbm_model.predict(transformed_frame)[0])
    ridge_pred = float(artifacts.ridge_model.predict(transformed)[0])
    raw_forecast = 0.50 * xgb_pred + 0.30 * lgbm_pred + 0.20 * ridge_pred

    live_last = float(live_quotes.usdtngn.last)
    live_bid = float(live_quotes.usdtngn.buy)
    live_ask = float(live_quotes.usdtngn.sell)

    # Extract external market values for AI and drivers
    merged = feature_result.merged
    brent_val = float(merged["brent"].iloc[-1]) if "brent" in merged.columns else None
    dxy_val = float(merged["dxy"].iloc[-1]) if "dxy" in merged.columns else None
    vix_val = float(merged["vix"].iloc[-1]) if "vix" in merged.columns else None
    usdzar_val = float(merged["usdzar"].iloc[-1]) if "usdzar" in merged.columns else None
    usdghs_val = float(merged["usdghs"].iloc[-1]) if "usdghs" in merged.columns else None
    usdkes_val = float(merged["usdkes"].iloc[-1]) if "usdkes" in merged.columns else None
    btcusd_val = float(merged["btcusd_global"].iloc[-1]) if "btcusd_global" in merged.columns else None
    usdngn_official_val = float(merged["usdngn_official"].iloc[-1]) if "usdngn_official" in merged.columns else None

    btc_premium_pct = None
    if "btc_premium" in latest_features.columns:
        btc_premium_pct = float(latest_features["btc_premium"].iloc[0]) * 100

    # Compute price changes
    close_series = export_tail["close"]
    change_2h_pct = float(close_series.pct_change(1).iloc[-1]) * 100 if len(close_series) > 1 else 0.0
    change_8h_pct = float(close_series.pct_change(4).iloc[-1]) * 100 if len(close_series) > 4 else 0.0
    change_24h_pct = float(close_series.pct_change(12).iloc[-1]) * 100 if len(close_series) > 12 else 0.0

    # Run Gemini AI Context Engine
    gemini_engine = GeminiAIContextEngine(settings)
    ai_result = gemini_engine.generate(
        current_rate=live_last,
        change_2h_pct=change_2h_pct,
        change_8h_pct=change_8h_pct,
        change_24h_pct=change_24h_pct,
        brent=brent_val,
        dxy=dxy_val,
        vix=vix_val,
        btc_premium_pct=btc_premium_pct,
        raw_forecast_return=raw_forecast,
        market_notes=market_notes,
    )

    # Apply AI sentiment adjustment
    adjusted_forecast = raw_forecast + (ai_result.sentiment_score * 0.001)

    # Determine signal
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

    # Confidence
    magnitude = abs(adjusted_forecast) / max(threshold, 1e-6)
    if signal != "HOLD":
        if magnitude >= 2.0:
            confidence = "HIGH"
        elif magnitude >= 1.3:
            confidence = "MEDIUM"
        else:
            confidence = "MEDIUM"
    else:
        confidence = ""

    # Widen confidence if high event magnitude
    if ai_result.event_magnitude > 0.5 and confidence == "HIGH":
        confidence = "MEDIUM"

    # 24h change for top bar
    price_24h_ago = float(close_series.iloc[-13]) if len(close_series) > 12 else live_last

    # Build 7-day history for chart
    chart_bars = min(84, len(export_tail))
    chart_data = export_tail.tail(chart_bars)[["close"]].copy()
    chart_data.index = chart_data.index.tz_convert(WAT) if chart_data.index.tzinfo else chart_data.index.tz_localize(UTC).tz_convert(WAT)

    # Weekly changes for drivers
    brent_week_change = None
    dxy_week_change = None
    vix_level = vix_val
    ghs_day_change = None
    zar_day_change = None

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

    if "usdzar" in merged.columns and len(merged) > 12:
        zar_old = float(merged["usdzar"].iloc[-13])
        if zar_old > 0 and usdzar_val:
            zar_day_change = ((usdzar_val - zar_old) / zar_old) * 100

    official_spread = None
    if usdngn_official_val and usdngn_official_val > 0:
        official_spread = ((live_last - usdngn_official_val) / usdngn_official_val) * 100

    # Source statuses
    source_statuses = {}
    for s in live_quotes.statuses:
        source_statuses[s.source_id] = s.status
    for s in market_fetch.statuses:
        source_statuses[s.source_id] = s.status

    return {
        "live_last": live_last,
        "live_bid": live_bid,
        "live_ask": live_ask,
        "spread": live_ask - live_bid,
        "signal": signal,
        "confidence": confidence,
        "raw_forecast": raw_forecast,
        "adjusted_forecast": adjusted_forecast,
        "forecast_price": forecast_price,
        "threshold": threshold,
        "change_24h_pct": change_24h_pct,
        "price_24h_ago": price_24h_ago,
        "ai_result": ai_result,
        "chart_data": chart_data,
        "brent": brent_val,
        "brent_week_change": brent_week_change,
        "dxy": dxy_val,
        "dxy_week_change": dxy_week_change,
        "vix": vix_level,
        "btc_premium_pct": btc_premium_pct,
        "official_spread": official_spread,
        "ghs_day_change": ghs_day_change,
        "zar_day_change": zar_day_change,
        "source_statuses": source_statuses,
        "timestamp": datetime.now(WAT),
    }


def _apply_live_quotes(export_frame, live_quotes, settings):
    """Apply live Quidax quotes to fill gaps since last export bar."""
    frame = export_frame.copy().sort_index()
    latest_export_time = frame.index.max()
    live_time = pd.Timestamp(live_quotes.usdtngn.at).tz_convert(UTC)
    live_bucket = live_time.floor("2h")
    synthetic_bars = 0

    if live_bucket > latest_export_time:
        missing_index = pd.date_range(
            start=latest_export_time + pd.Timedelta(hours=2),
            end=live_bucket,
            freq="2h",
            tz=UTC,
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
        st.markdown("### Settings")

        api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state.settings.gemini_api_key or "",
            type="password",
            help="Required for AI market assessment",
        )
        if api_key and api_key != (st.session_state.settings.gemini_api_key or ""):
            st.session_state.settings.gemini_api_key = api_key if api_key.strip() else None

        st.divider()

        threshold_pct = st.slider(
            "Signal Threshold (%)",
            min_value=0.10,
            max_value=1.00,
            value=0.30,
            step=0.05,
            format="%.2f",
            help="Minimum predicted move to generate a directional signal",
            disabled=True,
        )
        st.caption("Locked at 0.30% for launch")

        st.divider()

        if st.button("Run Prediction Now", use_container_width=True, type="primary"):
            with st.spinner("Running prediction cycle..."):
                try:
                    result = run_prediction(
                        st.session_state.settings,
                        market_notes=st.session_state.market_notes,
                    )
                    st.session_state.snapshot = result
                    st.session_state.ai_result = result["ai_result"]
                    st.session_state.last_refresh = datetime.now(WAT)
                    _log_signal(result)
                    st.rerun()
                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        st.divider()
        st.markdown("### Data Health")

        if st.session_state.snapshot:
            for src, status in st.session_state.snapshot["source_statuses"].items():
                if status == "ok":
                    st.markdown(f"<span style='color:#16a34a'>&#9679;</span> {src}", unsafe_allow_html=True)
                elif status == "degraded":
                    st.markdown(f"<span style='color:#f59e0b'>&#9679;</span> {src}", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:#dc2626'>&#9679;</span> {src}", unsafe_allow_html=True)

            ai_provider = st.session_state.snapshot["ai_result"].provider
            if "fallback" in ai_provider:
                st.markdown("<span style='color:#f59e0b'>&#9679;</span> Gemini AI", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#16a34a'>&#9679;</span> Gemini AI", unsafe_allow_html=True)
        else:
            st.caption("No data yet -- run a prediction")


# ===================================================================
# MAIN DASHBOARD
# ===================================================================

def render_top_bar(result: dict):
    """The Rate -- always visible, massive font."""
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown(f'<p class="rate-massive">{format_naira(result["live_last"])}</p>', unsafe_allow_html=True)

        spread_text = f'Bid {format_naira(result["live_bid"])} &nbsp;|&nbsp; Ask {format_naira(result["live_ask"])} &nbsp;|&nbsp; Spread {format_naira(result["spread"])}'
        st.markdown(f'<p class="rate-sub">{spread_text}</p>', unsafe_allow_html=True)

        change = result["change_24h_pct"]
        arrow = "&#9650;" if change >= 0 else "&#9660;"
        change_class = "rate-change-up" if change >= 0 else "rate-change-down"
        st.markdown(
            f'<p class="rate-sub"><span class="{change_class}">{arrow} {change:+.2f}%</span> from {format_naira(result["price_24h_ago"])} (24h)</p>',
            unsafe_allow_html=True,
        )

        ts = result["timestamp"]
        st.markdown(
            f'<p class="rate-timestamp">Last updated: {ts.strftime("%H:%M")} WAT &middot; {ts.strftime("%B %d, %Y")}</p>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(f'<p class="countdown">{get_countdown_text()}</p>', unsafe_allow_html=True)


def render_signal_card(result: dict):
    """The Signal -- the most important element."""
    signal = result["signal"]
    forecast_price = result["forecast_price"]
    confidence = result["confidence"]

    if signal == "UP":
        st.markdown(
            f"""
            <div class="signal-banner signal-up">
                <div class="signal-icon">&#8593;</div>
                <div class="signal-text" style="color: #16a34a;">RATE LIKELY MOVING UP</div>
                <div class="signal-forecast" style="color: #16a34a;">~{format_naira(forecast_price)}</div>
                <div class="signal-sub">Rate likely rising to ~{format_naira(forecast_price)} in next 2 hours</div>
                <div class="signal-confidence">Confidence: {confidence}</div>
                <div class="signal-action action-up">Consider accumulating USD before the move</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif signal == "DOWN":
        st.markdown(
            f"""
            <div class="signal-banner signal-down">
                <div class="signal-icon">&#8595;</div>
                <div class="signal-text" style="color: #dc2626;">RATE LIKELY MOVING DOWN</div>
                <div class="signal-forecast" style="color: #dc2626;">~{format_naira(forecast_price)}</div>
                <div class="signal-sub">Rate likely falling to ~{format_naira(forecast_price)} in next 2 hours</div>
                <div class="signal-confidence">Confidence: {confidence}</div>
                <div class="signal-action action-down">Consider accumulating NGN before the move</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="signal-banner signal-hold">
                <div class="signal-icon">&#8212;</div>
                <div class="signal-text" style="color: #525252;">NO SIGNIFICANT MOVE EXPECTED</div>
                <div class="signal-sub">No significant move expected in next 2 hours</div>
                <div class="signal-action action-hold">Trade normally at your standard spread</div>
                <div class="signal-neutral-band">Forecast within neutral band (&plusmn;0.30%)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_ai_assessment(result: dict):
    """AI Market Assessment Card."""
    ai = result["ai_result"]

    st.markdown('<p class="section-header">AI Market Assessment</p>', unsafe_allow_html=True)

    if ai.provider == "fallback":
        st.markdown(
            '<div class="ai-card"><p class="ai-unavailable">AI assessment unavailable -- signal based on quantitative model only</p></div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="ai-card"><p class="ai-narrative">{ai.narrative}</p></div>',
        unsafe_allow_html=True,
    )

    # Sentiment bar
    sentiment = ai.sentiment_score
    marker_pct = (sentiment + 1) / 2 * 100
    labels_html = '<div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#999;margin-top:2px;"><span>Bearish NGN</span><span>Neutral</span><span>Bullish NGN</span></div>'

    st.markdown(
        f"""
        <div class="sentiment-bar-container">
            <div class="sentiment-bar-track">
                <div class="sentiment-bar-marker" style="left: {marker_pct}%;"></div>
            </div>
            {labels_html}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Event magnitude alert
    if ai.event_magnitude > 0.5:
        st.markdown(
            '<div class="event-alert">&#9889; Significant market event detected -- forecast uncertainty is higher than usual</div>',
            unsafe_allow_html=True,
        )


def render_market_notes():
    """Interactive: Manual Market Notes Input."""
    st.markdown('<p class="section-header">Market Intelligence Notes</p>', unsafe_allow_html=True)

    notes = st.text_area(
        "Add market notes (included in next AI assessment)",
        value=st.session_state.market_notes,
        height=80,
        placeholder='e.g. "Heard CBN may intervene today" or "Large client asking for $2M quote"',
        label_visibility="collapsed",
    )
    if notes != st.session_state.market_notes:
        st.session_state.market_notes = notes

    if st.session_state.notes_history:
        with st.expander("Previous notes"):
            for note in reversed(st.session_state.notes_history[-10:]):
                st.caption(f"{note['time']} -- {note['text']}")


def render_price_chart(result: dict):
    """Price Chart -- 7-day USDT/NGN with signal overlays."""
    st.markdown('<p class="section-header">7-Day Price History</p>', unsafe_allow_html=True)

    chart_data = result["chart_data"]
    signal_log = get_signal_log()

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["close"],
            mode="lines",
            line=dict(color="#111111", width=2),
            name="USDT/NGN",
            hovertemplate="N%{y:,.2f}<br>%{x}<extra></extra>",
        )
    )

    # Overlay past signals on chart
    if not signal_log.empty:
        for _, row in signal_log.iterrows():
            sig_time = pd.to_datetime(row["datetime"])
            if sig_time.tzinfo is None:
                sig_time = sig_time.replace(tzinfo=UTC)
            sig_time_wat = sig_time.astimezone(WAT)

            if sig_time_wat < chart_data.index.min() or sig_time_wat > chart_data.index.max():
                continue

            sig = row["signal"]
            result_val = row.get("result", "")

            if sig == "UP":
                marker_color = "#16a34a"
                marker_symbol = "triangle-up"
            elif sig == "DOWN":
                marker_color = "#dc2626"
                marker_symbol = "triangle-down"
            else:
                marker_color = "#999999"
                marker_symbol = "circle"

            marker_size = 10
            annotation = ""
            if result_val == "correct":
                annotation = " OK"
            elif result_val == "wrong":
                annotation = " X"

            fig.add_trace(
                go.Scatter(
                    x=[sig_time_wat],
                    y=[float(row["current_price"])],
                    mode="markers+text",
                    marker=dict(color=marker_color, size=marker_size, symbol=marker_symbol),
                    text=[annotation],
                    textposition="top center",
                    textfont=dict(size=9, color=marker_color),
                    showlegend=False,
                    hovertemplate=f"{sig}{annotation}<br>N%{{y:,.2f}}<extra></extra>",
                )
            )

    fig.update_layout(
        height=340,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="#ffffff",
        paper_bgcolor="#ffffff",
        xaxis=dict(
            gridcolor="#f0f0f0",
            showgrid=True,
            tickformat="%b %d\n%H:%M",
        ),
        yaxis=dict(
            gridcolor="#f0f0f0",
            showgrid=True,
            tickformat=",.0f",
            tickprefix="N",
        ),
        hovermode="x unified",
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})


def render_market_drivers(result: dict):
    """Market Drivers Panel -- what's moving the rate."""
    st.markdown('<p class="section-header">Market Drivers</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        # Oil Price
        brent = result.get("brent")
        brent_change = result.get("brent_week_change")
        brent_str = f"${brent:.2f}" if brent else "N/A"
        change_str = ""
        if brent_change is not None:
            arrow = "&#9650;" if brent_change >= 0 else "&#9660;"
            color = "#16a34a" if brent_change >= 0 else "#dc2626"
            change_str = f'<p class="driver-change" style="color:{color}">{arrow} {brent_change:+.1f}% this week</p>'
        st.markdown(
            f'<div class="driver-card"><p class="driver-label">Oil Price (Brent)</p><p class="driver-value">{brent_str}</p>{change_str}</div>',
            unsafe_allow_html=True,
        )

        # Ghana Cedi
        ghs_change = result.get("ghs_day_change")
        ghs_str = "N/A"
        ghs_change_str = ""
        if ghs_change is not None:
            direction = "weakening" if ghs_change > 0 else "strengthening"
            color = "#dc2626" if ghs_change > 0 else "#16a34a"
            ghs_str = f"{direction} {abs(ghs_change):.1f}% today"
            ghs_change_str = f'<p class="driver-change" style="color:{color}">{ghs_str}</p>'
        st.markdown(
            f'<div class="driver-card"><p class="driver-label">Ghana Cedi</p>{ghs_change_str or "<p class=driver-value>N/A</p>"}</div>',
            unsafe_allow_html=True,
        )

    with col2:
        # Dollar Strength
        dxy = result.get("dxy")
        dxy_change = result.get("dxy_week_change")
        dxy_str = f"{dxy:.1f}" if dxy else "N/A"
        change_str = ""
        if dxy_change is not None:
            arrow = "&#9650;" if dxy_change >= 0 else "&#9660;"
            color = "#dc2626" if dxy_change >= 0 else "#16a34a"
            change_str = f'<p class="driver-change" style="color:{color}">{arrow} {dxy_change:+.1f}% this week</p>'
        st.markdown(
            f'<div class="driver-card"><p class="driver-label">Dollar Strength (DXY)</p><p class="driver-value">{dxy_str}</p>{change_str}</div>',
            unsafe_allow_html=True,
        )

        # BTC Premium
        btc_prem = result.get("btc_premium_pct")
        btc_str = "N/A"
        btc_note = ""
        if btc_prem is not None:
            btc_str = f"{btc_prem:+.1f}%"
            if abs(btc_prem) > 3:
                btc_note = '<p class="driver-change" style="color:#f59e0b">Elevated</p>'
            elif abs(btc_prem) > 5:
                btc_note = '<p class="driver-change" style="color:#dc2626">High</p>'
            else:
                btc_note = '<p class="driver-change" style="color:#16a34a">Normal</p>'
        st.markdown(
            f'<div class="driver-card"><p class="driver-label">BTC Premium on Quidax</p><p class="driver-value">{btc_str}</p>{btc_note}</div>',
            unsafe_allow_html=True,
        )

    with col3:
        # VIX / Market Fear
        vix = result.get("vix")
        vix_str = "N/A"
        vix_label = ""
        vix_color = "#999"
        if vix is not None:
            vix_str = f"{vix:.1f}"
            if vix < 20:
                vix_label = "Calm"
                vix_color = "#16a34a"
            elif vix < 30:
                vix_label = "Cautious"
                vix_color = "#f59e0b"
            else:
                vix_label = "Fear"
                vix_color = "#dc2626"
        st.markdown(
            f'<div class="driver-card"><p class="driver-label">Market Fear Index (VIX)</p><p class="driver-value">{vix_str}</p>'
            f'<p class="driver-change" style="color:{vix_color}">{vix_label}</p></div>',
            unsafe_allow_html=True,
        )

        # Official vs Parallel
        spread = result.get("official_spread")
        spread_str = f"{spread:.1f}%" if spread is not None else "N/A"
        st.markdown(
            f'<div class="driver-card"><p class="driver-label">Official vs Parallel Spread</p><p class="driver-value">{spread_str}</p></div>',
            unsafe_allow_html=True,
        )


def render_signal_history():
    """Signal History Table -- last 30 signals."""
    st.markdown('<p class="section-header">Signal History</p>', unsafe_allow_html=True)

    signal_log = get_signal_log()
    if signal_log.empty:
        st.caption("No signals recorded yet. Run a prediction to start building history.")
        return

    display = signal_log.tail(30).copy()
    display = display.sort_values("datetime", ascending=False)

    # Compute accuracy
    evaluated = display[display["result"].isin(["correct", "wrong"])]
    if not evaluated.empty:
        correct_count = (evaluated["result"] == "correct").sum()
        total = len(evaluated)
        acc = correct_count / total * 100
        st.markdown(
            f'<p class="history-accuracy">Last {total} evaluated signals: {correct_count}/{total} correct ({acc:.1f}%)</p>',
            unsafe_allow_html=True,
        )

    # Format for display
    display_df = display[["datetime", "signal", "forecast_price", "current_price", "actual_price_2h", "result", "pnl_bps"]].copy()
    display_df.columns = ["Date/Time", "Signal", "Forecast Price", "Entry Price", "Actual 2h Later", "Result", "PnL (bps)"]
    display_df["Date/Time"] = pd.to_datetime(display_df["Date/Time"]).dt.strftime("%b %d %H:%M")
    display_df["Forecast Price"] = display_df["Forecast Price"].apply(lambda x: f"N{x:,.2f}" if pd.notna(x) else "--")
    display_df["Entry Price"] = display_df["Entry Price"].apply(lambda x: f"N{x:,.2f}" if pd.notna(x) else "--")
    display_df["Actual 2h Later"] = display_df["Actual 2h Later"].apply(lambda x: f"N{x:,.2f}" if pd.notna(x) else "Pending...")
    display_df["Result"] = display_df["Result"].apply(lambda x: "OK" if x == "correct" else ("X" if x == "wrong" else ("--" if x == "--" else "...")))
    display_df["PnL (bps)"] = display_df["PnL (bps)"].apply(lambda x: f"{x:+.1f}" if pd.notna(x) and x != 0 else "--")

    st.dataframe(display_df, use_container_width=True, hide_index=True)


def render_performance_summary():
    """Performance Summary Card."""
    st.markdown('<p class="section-header">Performance Summary</p>', unsafe_allow_html=True)

    signal_log = get_signal_log()
    if signal_log.empty:
        st.caption("No performance data yet.")
        return

    # This month
    now = datetime.now(UTC)
    signal_log["datetime"] = pd.to_datetime(signal_log["datetime"])
    this_month = signal_log[
        signal_log["datetime"].dt.month == now.month
    ]

    directional = this_month[this_month["signal"] != "HOLD"]
    evaluated = directional[directional["result"].isin(["correct", "wrong"])]

    total_signals = len(this_month)
    total_evaluated = len(evaluated)
    correct = (evaluated["result"] == "correct").sum() if total_evaluated > 0 else 0
    accuracy = (correct / total_evaluated * 100) if total_evaluated > 0 else 0
    total_pnl = evaluated["pnl_bps"].sum() if total_evaluated > 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f'<div class="perf-stat"><p class="perf-number">{total_signals}</p><p class="perf-label">Signals This Month</p></div>',
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f'<div class="perf-stat"><p class="perf-number">{correct}/{total_evaluated}</p><p class="perf-label">Correct</p></div>',
            unsafe_allow_html=True,
        )

    with col3:
        color = "#16a34a" if accuracy > 50 else "#dc2626"
        st.markdown(
            f'<div class="perf-stat"><p class="perf-number" style="color:{color}">{accuracy:.1f}%</p><p class="perf-label">Accuracy</p></div>',
            unsafe_allow_html=True,
        )

    with col4:
        pnl_color = "#16a34a" if total_pnl >= 0 else "#dc2626"
        st.markdown(
            f'<div class="perf-stat"><p class="perf-number" style="color:{pnl_color}">{total_pnl:+.0f}</p><p class="perf-label">Value Added (bps)</p></div>',
            unsafe_allow_html=True,
        )

    # Model vs Random comparison
    if total_evaluated > 0:
        st.caption(f"Model: {accuracy:.0f}% vs Random: 50% -- {accuracy - 50:+.0f}pp edge")


def _log_signal(result: dict):
    """Log a signal to the CSV."""
    ai = result["ai_result"]
    row = {
        "datetime": datetime.now(UTC).isoformat(),
        "signal": result["signal"],
        "forecast_price": round(result["forecast_price"], 2),
        "current_price": round(result["live_last"], 2),
        "predicted_return": round(result["adjusted_forecast"], 6),
        "actual_price_2h": "",
        "result": "",
        "pnl_bps": "",
        "confidence": result["confidence"],
        "ai_sentiment": round(ai.sentiment_score, 3),
        "ai_magnitude": round(ai.event_magnitude, 3),
    }
    append_signal_log(row)

    # Log market notes
    if st.session_state.market_notes.strip():
        st.session_state.notes_history.append({
            "time": datetime.now(WAT).strftime("%H:%M %b %d"),
            "text": st.session_state.market_notes.strip(),
        })


# ===================================================================
# MAIN
# ===================================================================

def main():
    init_services()
    render_sidebar()

    # --- Top bar title ---
    st.markdown(
        '<p style="margin:0 0 -8px 0;font-size:0.75rem;letter-spacing:0.15em;text-transform:uppercase;color:#999999;">Quidax OTC Rate Intelligence</p>',
        unsafe_allow_html=True,
    )

    # --- Load existing snapshot or prompt to run ---
    result = st.session_state.snapshot

    if result is None:
        # Try to load from cached signal
        cache_path = st.session_state.settings.runtime_dir / "latest_signal.json"
        if cache_path.exists():
            try:
                cached = InferenceSnapshot.model_validate_json(cache_path.read_text())
                # Build a minimal result dict from cached data for initial display
                st.info("Showing cached data. Click **Run Prediction Now** in the sidebar for live data.")
                st.markdown(
                    f'<p class="rate-massive">{format_naira(cached.live_last_trade)}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'<p class="rate-sub">Bid {format_naira(cached.live_bid)} | Ask {format_naira(cached.live_ask)} | Spread {format_naira(cached.live_ask - cached.live_bid)}</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<p class="rate-timestamp">Cached from: {cached.as_of.strftime("%H:%M WAT, %B %d, %Y")}</p>', unsafe_allow_html=True)

                # Show cached signal
                signal = cached.signal
                if signal == "buy_usd":
                    st.markdown('<div class="signal-banner signal-up"><div class="signal-text" style="color:#16a34a;">RATE LIKELY MOVING UP (cached)</div></div>', unsafe_allow_html=True)
                elif signal == "buy_ngn":
                    st.markdown('<div class="signal-banner signal-down"><div class="signal-text" style="color:#dc2626;">RATE LIKELY MOVING DOWN (cached)</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="signal-banner signal-hold"><div class="signal-text" style="color:#525252;">NO SIGNIFICANT MOVE EXPECTED (cached)</div></div>', unsafe_allow_html=True)

                st.markdown("---")
                st.caption("Open the sidebar (top left) and click **Run Prediction Now** to get a fresh live signal with AI assessment.")
                return
            except Exception:
                pass

        st.markdown("## Welcome to the OTC Rate Intelligence Dashboard")
        st.markdown("Open the sidebar and click **Run Prediction Now** to generate your first live signal.")
        st.markdown("You can also enter your Gemini API key in the sidebar to enable AI market assessments.")
        return

    # --- Update signal outcomes with current price ---
    signal_log = get_signal_log()
    if not signal_log.empty:
        update_signal_outcomes(signal_log, result["live_last"])

    # --- Render the full dashboard ---
    render_top_bar(result)
    st.markdown("---")

    render_signal_card(result)

    render_ai_assessment(result)

    render_market_notes()

    render_price_chart(result)

    render_market_drivers(result)

    col_left, col_right = st.columns([1, 1])
    with col_left:
        render_signal_history()
    with col_right:
        render_performance_summary()


if __name__ == "__main__":
    main()
else:
    main()
