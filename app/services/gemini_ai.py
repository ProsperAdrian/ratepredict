from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from app.config import Settings


@dataclass(frozen=True)
class AIContextResult:
    sentiment_score: float
    event_magnitude: float
    narrative: str
    provider: str
    generated_at: datetime


class GeminiAIContextEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(
        self,
        *,
        current_rate: float,
        change_2h_pct: float,
        change_8h_pct: float,
        change_24h_pct: float,
        brent: float | None,
        dxy: float | None,
        vix: float | None,
        btc_premium_pct: float | None,
        raw_forecast_return: float,
        market_notes: str = "",
    ) -> AIContextResult:
        api_key = self.settings.gemini_api_key
        if not api_key:
            return self._fallback_result()

        prompt = self._build_prompt(
            current_rate=current_rate,
            change_2h_pct=change_2h_pct,
            change_8h_pct=change_8h_pct,
            change_24h_pct=change_24h_pct,
            brent=brent,
            dxy=dxy,
            vix=vix,
            btc_premium_pct=btc_premium_pct,
            raw_forecast_return=raw_forecast_return,
            market_notes=market_notes,
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.settings.gemini_model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 512},
        }
        headers = {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.settings.gemini_timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            text = (
                body.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            if not text:
                raise ValueError("Gemini returned no text content")
            parsed = self._parse_response(text)
            return AIContextResult(
                sentiment_score=parsed["sentiment_score"],
                event_magnitude=parsed["event_magnitude"],
                narrative=parsed["narrative"],
                provider=f"gemini:{self.settings.gemini_model}",
                generated_at=datetime.now(UTC),
            )
        except Exception:
            return self._fallback_result()

    def _build_prompt(
        self,
        *,
        current_rate: float,
        change_2h_pct: float,
        change_8h_pct: float,
        change_24h_pct: float,
        brent: float | None,
        dxy: float | None,
        vix: float | None,
        btc_premium_pct: float | None,
        raw_forecast_return: float,
        market_notes: str,
    ) -> str:
        brent_str = f"${brent:.2f}" if brent else "unavailable"
        dxy_str = f"{dxy:.2f}" if dxy else "unavailable"
        vix_str = f"{vix:.2f}" if vix else "unavailable"
        btc_str = f"{btc_premium_pct:.2f}%" if btc_premium_pct is not None else "unavailable"
        notes_section = f"\nMarket notes from desk:\n{market_notes}" if market_notes.strip() else ""

        return (
            "You are the AI market intelligence engine for a Nigerian OTC foreign exchange desk "
            "that trades USD/NGN (specifically USDT/NGN on Quidax). Your job is to assess current "
            "market conditions and provide a sentiment adjustment to the quantitative model.\n\n"
            "Current market data:\n"
            f"- Current USDT/NGN rate: {current_rate:,.2f}\n"
            f"- 2-hour price change: {change_2h_pct:+.3f}%\n"
            f"- 8-hour price change: {change_8h_pct:+.3f}%\n"
            f"- 24-hour price change: {change_24h_pct:+.3f}%\n"
            f"- Brent crude: {brent_str}\n"
            f"- Dollar index (DXY): {dxy_str}\n"
            f"- VIX (market fear): {vix_str}\n"
            f"- BTC premium on Quidax: {btc_str}\n"
            f"- ML model raw forecast (2h return): {raw_forecast_return*100:+.4f}%\n"
            f"{notes_section}\n\n"
            "Respond with ONLY a valid JSON object (no markdown, no code fences) with exactly these fields:\n"
            '{\n'
            '  "sentiment_score": <float from -1.0 to +1.0>,\n'
            '  "event_magnitude": <float from 0.0 to 1.0>,\n'
            '  "narrative": "<3-5 sentence market assessment in plain English>"\n'
            '}\n\n'
            "Interpretation guide:\n"
            "- sentiment_score: negative = bearish NGN (expect NGN to weaken, rate goes up), "
            "positive = bullish NGN (expect NGN to strengthen, rate goes down). Zero = neutral.\n"
            "- event_magnitude: 0.0 = nothing notable happening, 1.0 = major policy shock or CBN intervention.\n"
            "- narrative: Write for non-technical sales people. No jargon. Explain what is happening "
            "and what it means for the rate in the next few hours."
        )

    def _parse_response(self, text: str) -> dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        parsed = json.loads(cleaned)

        sentiment = float(parsed["sentiment_score"])
        sentiment = max(-1.0, min(1.0, sentiment))

        magnitude = float(parsed["event_magnitude"])
        magnitude = max(0.0, min(1.0, magnitude))

        narrative = str(parsed["narrative"]).strip()
        if not narrative:
            raise ValueError("Empty narrative")

        return {
            "sentiment_score": sentiment,
            "event_magnitude": magnitude,
            "narrative": narrative,
        }

    def _fallback_result(self) -> AIContextResult:
        return AIContextResult(
            sentiment_score=0.0,
            event_magnitude=0.0,
            narrative="AI assessment unavailable -- signal based on quantitative model only.",
            provider="fallback",
            generated_at=datetime.now(UTC),
        )
