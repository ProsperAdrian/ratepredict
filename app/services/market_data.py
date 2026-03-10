from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pandas as pd

from app.config import Settings
from app.schemas import SourceStatus


@dataclass(frozen=True)
class MarketFetchResult:
    frame: pd.DataFrame
    statuses: list[SourceStatus]


@dataclass(frozen=True)
class QuidaxTicker:
    market: str
    at: datetime
    buy: float
    sell: float
    low: float
    high: float
    open: float
    last: float
    vol: float


@dataclass(frozen=True)
class QuidaxMarketSnapshot:
    usdtngn: QuidaxTicker
    btcngn: QuidaxTicker
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

        return self.fetch_live(start=start, end=end)

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


class QuidaxTickerService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch(self) -> QuidaxMarketSnapshot:
        statuses: list[SourceStatus] = []
        with httpx.Client(timeout=self.settings.http_timeout_seconds) as client:
            usdtngn = self._fetch_one(client, self.settings.quidax_usdtngn_ticker_url, "quidax_usdtngn", statuses)
            btcngn = self._fetch_one(client, self.settings.quidax_btcngn_ticker_url, "quidax_btcngn", statuses)
        return QuidaxMarketSnapshot(usdtngn=usdtngn, btcngn=btcngn, statuses=statuses)

    def _fetch_one(
        self,
        client: httpx.Client,
        url: str,
        source_id: str,
        statuses: list[SourceStatus],
    ) -> QuidaxTicker:
        response = client.get(url)
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") != "success":
            raise RuntimeError(f"{source_id} returned non-success payload: {payload}")

        data = payload["data"]
        ticker = data["ticker"]
        at = datetime.fromtimestamp(int(data["at"]), tz=UTC)

        result = QuidaxTicker(
            market=str(data["market"]),
            at=at,
            buy=self._to_float(ticker["buy"]),
            sell=self._to_float(ticker["sell"]),
            low=self._to_float(ticker["low"]),
            high=self._to_float(ticker["high"]),
            open=self._to_float(ticker["open"]),
            last=self._to_float(ticker["last"]),
            vol=self._to_float(ticker["vol"]),
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


class QuidaxKlineService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def fetch(self, market: str, *, period_minutes: int | None = None, limit: int | None = None) -> QuidaxKlineFetchResult:
        period = period_minutes or self.settings.quidax_kline_period_minutes
        capped_limit = limit or self.settings.quidax_kline_limit
        url = f"https://app.quidax.io/api/v1/markets/{market}/k?period={period}&limit={capped_limit}"

        with httpx.Client(timeout=self.settings.http_timeout_seconds) as client:
            response = client.get(url)
            response.raise_for_status()
            payload = response.json()

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
