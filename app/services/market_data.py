from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
import json

import httpx
import pandas as pd

from app.config import Settings
from app.schemas import SourceStatus


@dataclass(frozen=True)
class MarketFetchResult:
    frame: pd.DataFrame
    statuses: list[SourceStatus]


@dataclass(frozen=True)
class LiveTicker:
    market: str
    at: datetime
    buy: float
    sell: float
    last: float
    low: float | None = None
    high: float | None = None
    open: float | None = None
    vol: float | None = None
    provider: str | None = None
    source: str | None = None


@dataclass(frozen=True)
class LiveMarketSnapshot:
    usdtngn: LiveTicker
    btcngn: LiveTicker
    statuses: list[SourceStatus]


@dataclass(frozen=True)
class QuidaxKlineFetchResult:
    market: str
    frame: pd.DataFrame
    status: SourceStatus


class ExternalDailyMarketDataService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch(self, start: datetime, end: datetime) -> MarketFetchResult:
        cached = self.load_cached()
        start_ts = pd.Timestamp(start)
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize(UTC)
        else:
            start_ts = start_ts.tz_convert(UTC)
        if cached is not None:
            frame, statuses = cached
            if frame.index.max().date() >= min(end.date(), datetime.now(UTC).date()):
                return MarketFetchResult(frame=frame.loc[frame.index >= start_ts], statuses=statuses)
            if not self.settings.external_live_fallback_enabled:
                return MarketFetchResult(frame=frame.loc[frame.index >= start_ts], statuses=statuses)

        try:
            live = self.fetch_live(start=start, end=end)
        except Exception as exc:
            if cached is None:
                raise
            frame, statuses = cached
            statuses = statuses + [
                SourceStatus(
                    source_id="external_daily_live",
                    status="degraded",
                    latest_timestamp=frame.index.max().to_pydatetime(),
                    message=f"Using cache because live refresh failed: {exc}",
                )
            ]
            return MarketFetchResult(frame=frame.loc[frame.index >= start_ts], statuses=statuses)

        if cached is None:
            self.write_cache(live.frame)
            return MarketFetchResult(frame=live.frame.loc[live.frame.index >= start_ts], statuses=live.statuses)

        cached_frame, cached_statuses = cached
        combined = live.frame.combine_first(cached_frame).sort_index()
        self.write_cache(combined)

        missing_live_aliases = [
            alias for alias in self.settings.yahoo_tickers.values()
            if alias not in live.frame.columns and alias in cached_frame.columns
        ]
        statuses = live.statuses.copy()
        statuses.append(
            SourceStatus(
                source_id="external_daily_cache",
                status="degraded" if missing_live_aliases else "ok",
                latest_timestamp=combined.index.max().to_pydatetime(),
                message=(
                    f"Filled live gaps from cache for: {', '.join(missing_live_aliases)}"
                    if missing_live_aliases
                    else "Live refresh merged into cache"
                ),
            )
        )
        return MarketFetchResult(frame=combined.loc[combined.index >= start_ts], statuses=statuses)

    def fetch_live(self, start: datetime, end: datetime) -> MarketFetchResult:
        import yfinance as yf

        external_frames: list[pd.Series] = []
        statuses: list[SourceStatus] = []

        for ticker, alias in self.settings.yahoo_tickers.items():
            try:
                data = yf.download(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    interval="1d",
                    progress=False,
                    auto_adjust=True,
                    threads=False,
                )
                if len(data) == 0:
                    statuses.append(SourceStatus(source_id=alias, status="missing", message="No rows returned"))
                    continue

                series = data["Close"]
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                series = series.rename(alias)
                series.index = pd.to_datetime(series.index, utc=True)
                external_frames.append(series)
                statuses.append(
                    SourceStatus(
                        source_id=alias,
                        status="ok",
                        latest_timestamp=series.index.max().to_pydatetime(),
                        message=f"{len(series)} daily rows",
                    )
                )
            except Exception as exc:
                statuses.append(SourceStatus(source_id=alias, status="error", message=str(exc)))

        if not external_frames:
            raise RuntimeError("External daily market data returned no usable series.")

        frame = pd.concat(external_frames, axis=1).sort_index()
        expected_aliases = list(self.settings.yahoo_tickers.values())
        frame = frame.reindex(columns=expected_aliases)
        return MarketFetchResult(frame=frame, statuses=statuses)

    def load_cached(self) -> tuple[pd.DataFrame, list[SourceStatus]] | None:
        path = self.settings.data_dir / self.settings.external_daily_filename
        if not path.exists():
            return None
        frame = pd.read_csv(path)
        if "date" not in frame.columns:
            raise ValueError(f"Cached external daily file is missing 'date': {path}")
        frame["date"] = pd.to_datetime(frame["date"], utc=True)
        frame = frame.sort_values("date").set_index("date")
        statuses = [
            SourceStatus(
                source_id="external_daily_cache",
                status="ok",
                latest_timestamp=frame.index.max().to_pydatetime(),
                message=path.name,
            )
        ]
        return frame, statuses

    def write_cache(self, frame: pd.DataFrame) -> None:
        path = self.settings.data_dir / self.settings.external_daily_filename
        output = frame.copy().sort_index()
        output.index.name = "date"
        output.reset_index().to_csv(path, index=False)


class LiveQuoteService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch(self) -> LiveMarketSnapshot:
        statuses: list[SourceStatus] = []
        usdtngn = self._fetch_usdtngn(statuses)
        btcngn = self._fetch_quidax_ticker(
            self.settings.quidax_btcngn_ticker_url,
            "quidax_btcngn",
            statuses,
        )
        return LiveMarketSnapshot(usdtngn=usdtngn, btcngn=btcngn, statuses=statuses)

    def _fetch_usdtngn(self, statuses: list[SourceStatus]) -> LiveTicker:
        return self._fetch_qbot_rate(statuses)

    def _fetch_qbot_rate(
        self,
        statuses: list[SourceStatus],
    ) -> LiveTicker:
        missing = [
            name
            for name, value in (
                ("qbot_service_token", self.settings.qbot_service_token),
                ("qbot_cf_access_client_id", self.settings.qbot_cf_access_client_id),
                ("qbot_cf_access_client_secret", self.settings.qbot_cf_access_client_secret),
            )
            if not value
        ]
        if missing:
            raise RuntimeError(f"Missing qbot credentials: {', '.join(missing)}")

        headers = {
            "x-service-token": self.settings.qbot_service_token,
            "CF-Access-Client-Id": self.settings.qbot_cf_access_client_id,
            "CF-Access-Client-Secret": self.settings.qbot_cf_access_client_secret,
            "Accept": "application/json",
        }

        payload = self._request_json(
            self.settings.qbot_usdtngn_rate_url,
            headers=headers,
            follow_redirects=False,
        )

        result = self._ticker_from_payload(payload, fallback_source_id="qbot_usdtngn")
        statuses.append(
            SourceStatus(
                source_id="qbot_usdtngn",
                status="ok",
                latest_timestamp=result.at,
                message=(
                    f"provider={result.provider or 'unknown'} "
                    f"bid={result.buy} ask={result.sell} mid={result.last}"
                ),
            )
        )
        return result

    def _ticker_from_payload(self, payload: dict, *, fallback_source_id: str) -> LiveTicker:
        if payload.get("status") == "success" and isinstance(payload.get("data"), dict):
            data = payload["data"]
            current = data.get("current")
            if isinstance(current, dict):
                recent = data.get("recent", [])
                recent_mids = [self._to_float(row["midRate"]) for row in recent if row.get("midRate") is not None]
                at = pd.Timestamp(current["rateAsAt"])
                if at.tzinfo is None:
                    at = at.tz_localize(UTC)
                else:
                    at = at.tz_convert(UTC)

                mid_rate = self._to_float(current["midRate"])
                current_price = self._to_float(current.get("sellRate", current["midRate"]))
                return LiveTicker(
                    market="usdtngn",
                    at=at.to_pydatetime(),
                    buy=self._to_float(current["buyRate"]),
                    sell=self._to_float(current.get("sellRate", current["midRate"])),
                    last=current_price,
                    low=min(recent_mids) if recent_mids else mid_rate,
                    high=max(recent_mids) if recent_mids else mid_rate,
                    open=recent_mids[-1] if recent_mids else mid_rate,
                    vol=None,
                    provider=str(current.get("provider") or ""),
                    source=str(current.get("source") or fallback_source_id),
                )

        raise RuntimeError(f"Unsupported qbot payload shape: {payload}")

    def _fetch_quidax_ticker(
        self,
        url: str,
        source_id: str,
        statuses: list[SourceStatus],
    ) -> LiveTicker:
        payload = self._request_json(
            url,
            headers={"Accept": "application/json"},
        )

        if payload.get("status") != "success":
            raise RuntimeError(f"{source_id} returned non-success payload: {payload}")

        data = payload["data"]
        ticker = data["ticker"]
        at = datetime.fromtimestamp(int(data["at"]), tz=UTC)

        result = LiveTicker(
            market=str(data["market"]),
            at=at,
            buy=self._to_float(ticker["buy"]),
            sell=self._to_float(ticker["sell"]),
            last=self._to_float(ticker["last"]),
            low=self._to_float(ticker["low"]),
            high=self._to_float(ticker["high"]),
            open=self._to_float(ticker["open"]),
            vol=self._to_float(ticker["vol"]),
            provider="quidax",
            source=source_id,
        )
        statuses.append(
            SourceStatus(
                source_id=source_id,
                status="ok",
                latest_timestamp=at,
                message=f"{result.market} bid={result.buy} ask={result.sell} last={result.last}",
            )
        )
        return result

    def _to_float(self, value: str | float | int) -> float:
        return float(Decimal(str(value)))

    def _auth_diag(self, headers: dict[str, str]) -> str:
        parts = [f"url={self.settings.qbot_usdtngn_rate_url}"]
        for name in (
            "CF-Access-Client-Id",
            "CF-Access-Client-Secret",
            "x-service-token",
        ):
            value = headers.get(name, "")
            prefix = value[:4] if value else ""
            parts.append(f"{name}_len={len(value)}_prefix={prefix!r}")
        return " ".join(parts)

    def _request_json(
        self,
        url: str,
        headers: dict[str, str],
        *,
        follow_redirects: bool = True,
    ) -> dict:
        timeout = max(self.settings.http_timeout_seconds, 1.0)
        status_code, body, transport = self._http_get(
            url,
            headers=headers,
            timeout=timeout,
            follow_redirects=follow_redirects,
        )

        body_snippet = " ".join(body.strip().split())[:280]
        lower = body_snippet.lower()
        is_access_login = "cloudflare access" in lower or "sign in" in lower
        is_waf_block = "no-js ie6 oldie" in lower or (
            status_code == 403 and "<!doctype html>" in lower and not is_access_login
        )

        if status_code in {301, 302, 303, 307, 308}:
            raise RuntimeError(
                f"Upstream returned HTTP {status_code} (Cloudflare Access redirect). "
                "CF-Access Client Id/Secret were not accepted. "
                f"transport={transport} {self._auth_diag(headers)}. "
                f"Response snippet: {body_snippet or '[empty body]'}"
            )
        if status_code == 403:
            if is_waf_block:
                raise RuntimeError(
                    "Upstream returned 403 Cloudflare bot/WAF HTML. "
                    f"transport={transport} {self._auth_diag(headers)}. "
                    "Credentials look present; Cloudflare still rejected the client fingerprint "
                    "from this host. Response snippet: "
                    f"{body_snippet or '[empty body]'}"
                )
            raise RuntimeError(
                "Upstream returned 403 Forbidden. "
                f"transport={transport} {self._auth_diag(headers)}. "
                f"Response snippet: {body_snippet or '[empty body]'}"
            )
        if status_code >= 400:
            raise RuntimeError(
                f"Request failed with HTTP {status_code}. "
                f"transport={transport} {self._auth_diag(headers)}. "
                f"Response snippet: {body_snippet or '[empty body]'}"
            )

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Non-JSON response returned from {url}. "
                f"transport={transport} {self._auth_diag(headers)}. "
                f"Response snippet: {body_snippet or '[empty body]'}"
            ) from exc

    def _http_get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float,
        follow_redirects: bool,
    ) -> tuple[int, str, str]:
        """Prefer browser-impersonated TLS for Cloudflare-protected hosts."""
        uses_cf_access = "CF-Access-Client-Id" in headers
        cf_error = ""
        if uses_cf_access:
            try:
                from curl_cffi import requests as cf_requests

                response = cf_requests.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                    allow_redirects=follow_redirects,
                    impersonate="chrome",
                )
                return response.status_code, response.text, "curl_cffi/chrome"
            except Exception as exc:
                cf_error = str(exc)

        try:
            with httpx.Client(timeout=timeout, follow_redirects=follow_redirects) as client:
                response = client.get(url, headers=headers)
            transport = "httpx"
            if cf_error:
                transport = f"httpx(after curl_cffi error: {cf_error[:120]})"
            return response.status_code, response.text, transport
        except httpx.HTTPError as exc:
            raise RuntimeError(f"HTTP request failed for {url}: {exc}") from exc


class QuidaxKlineService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch(self, market: str, *, period_minutes: int | None = None, limit: int | None = None) -> QuidaxKlineFetchResult:
        period = period_minutes or self.settings.quidax_kline_period_minutes
        capped_limit = limit or self.settings.quidax_kline_limit
        url = f"https://app.quidax.io/api/v1/markets/{market}/k?period={period}&limit={capped_limit}"

        payload = LiveQuoteService(self.settings)._request_json(
            url,
            headers={"Accept": "application/json"},
        )

        if payload.get("status") != "success":
            raise RuntimeError(f"Quidax k-line call failed for {market}: {payload}")

        rows = payload.get("data", [])
        frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame["bucket_2h"] = pd.to_datetime(frame["timestamp"], unit="s", utc=True)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.drop(columns=["timestamp"]).sort_values("bucket_2h").set_index("bucket_2h")
        status = SourceStatus(
            source_id=f"quidax_kline_{market}",
            status="ok",
            latest_timestamp=frame.index.max().to_pydatetime() if not frame.empty else None,
            message=f"{len(frame)} bars",
        )
        return QuidaxKlineFetchResult(market=market, frame=frame, status=status)
