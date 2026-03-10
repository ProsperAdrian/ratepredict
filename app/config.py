from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_title: str = "Quidax USD/NGN Desk Signal"
    app_version: str = "1.0.0"

    base_dir: Path = Path("/Users/prosper/Documents/ratepredict")
    artifacts_dir: Path = Path("/Users/prosper/Documents/ratepredict/artifacts")
    data_dir: Path = Path("/Users/prosper/Documents/ratepredict/data/latest")
    runtime_dir: Path = Path("/Users/prosper/Documents/ratepredict/runtime")

    export_glob: str = "bquxjob_*.csv"
    runtime_bars_filename: str = "quidax_runtime_2h.csv"
    external_daily_filename: str = "external_daily.csv"
    feature_lookback_bars: int = 480
    price_threshold: float | None = None
    assumed_round_trip_cost_bps: float = 5.0
    http_timeout_seconds: float = 15.0
    external_live_fallback_enabled: bool = True
    quidax_kline_limit: int = 1000
    quidax_kline_period_minutes: int = 120

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"
    gemini_timeout_seconds: float = 20.0

    auto_refresh_enabled: bool = False
    auto_refresh_seconds: int = 900

    quidax_usdtngn_ticker_url: str = "https://app.quidax.io/api/v1/markets/tickers/usdtngn"
    quidax_btcngn_ticker_url: str = "https://app.quidax.io/api/v1/markets/tickers/btcngn"

    yahoo_tickers: dict[str, str] = Field(
        default_factory=lambda: {
            "BZ=F": "brent",
            "DX-Y.NYB": "dxy",
            "^VIX": "vix",
            "USDZAR=X": "usdzar",
            "USDNGN=X": "usdngn_official",
            "USDGHS=X": "usdghs",
            "USDKES=X": "usdkes",
            "BTC-USD": "btcusd_global",
        }
    )

    @field_validator("gemini_api_key", mode="before")
    @classmethod
    def blank_api_key_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings
