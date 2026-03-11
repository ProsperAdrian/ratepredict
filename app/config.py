from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

GEMINI_MODEL_ALIASES = {
    "gemini-3.1-flash": "gemini-3-flash-preview",
    "gemini-3-flash": "gemini-3-flash-preview",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: str = "development"
    app_title: str = "Quidax USD/NGN Desk Signal"
    app_version: str = "1.0.0"

    base_dir: Path = Path(__file__).resolve().parent.parent
    artifacts_dir: Path = Path(__file__).resolve().parent.parent / "artifacts"
    data_dir: Path = Path(__file__).resolve().parent.parent / "data" / "latest"
    runtime_dir: Path = Path(__file__).resolve().parent.parent / "runtime"

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
    gemini_model: str = "gemini-3-flash-preview"
    gemini_timeout_seconds: float = 20.0

    auto_refresh_enabled: bool = False
    auto_refresh_seconds: int = 900

    news_enabled: bool = True
    news_cache_ttl_seconds: int = 900
    news_max_items: int = 50
    news_max_age_hours: int = 72
    news_fetch_timeout_seconds: float = 10.0

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

    @field_validator("gemini_model", mode="before")
    @classmethod
    def normalize_gemini_model(cls, value: str | None) -> str:
        if value is None:
            return "gemini-3-flash-preview"
        stripped = str(value).strip().lower()
        if not stripped:
            return "gemini-3-flash-preview"
        return GEMINI_MODEL_ALIASES.get(stripped, stripped)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # In this app, the project-local .env should win over any exported shell
        # variables so the dashboard uses the workspace configuration the user sees.
        return init_settings, dotenv_settings, env_settings, file_secret_settings


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return settings


def reload_settings() -> Settings:
    get_settings.cache_clear()
    return get_settings()
