from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from app.config import Settings


@dataclass(frozen=True)
class AIDriver:
    label: str
    score: float
    detail: str


@dataclass(frozen=True)
class AIContextResult:
    sentiment_score: float
    event_magnitude: float
    narrative: str
    drivers: list[AIDriver]
    provider: str
    generated_at: datetime


class GeminiAIContextEngine:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(
        self,
        *,
        # --- Raw market observables only (NO ML outputs) ---
        current_rate: float,
        bid: float,
        ask: float,
        change_2h_pct: float,
        change_8h_pct: float,
        change_24h_pct: float,
        brent: float | None,
        dxy: float | None,
        vix: float | None,
        btc_premium_pct: float | None,
        usdngn_official: float | None,
        official_parallel_spread_pct: float | None,
        usdghs: float | None,
        # --- Human + news intelligence ---
        market_notes: str = "",
        news_headlines: str = "",
    ) -> AIContextResult:
        api_key = self.settings.gemini_api_key
        if not api_key:
            return self._fallback_result(
                "Gemini is not configured. Set GEMINI_API_KEY to enable AI assessment."
            )

        prompt = self._build_prompt(
            current_rate=current_rate,
            bid=bid,
            ask=ask,
            change_2h_pct=change_2h_pct,
            change_8h_pct=change_8h_pct,
            change_24h_pct=change_24h_pct,
            brent=brent,
            dxy=dxy,
            vix=vix,
            btc_premium_pct=btc_premium_pct,
            usdngn_official=usdngn_official,
            official_parallel_spread_pct=official_parallel_spread_pct,
            usdghs=usdghs,
            market_notes=market_notes,
            news_headlines=news_headlines,
        )

        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.settings.gemini_model}:generateContent"
            f"?key={api_key}"
        )
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.15,
                "maxOutputTokens": 1024,
                "thinkingConfig": {"thinkingLevel": "minimal"},
                "responseMimeType": "application/json",
                "responseJsonSchema": {
                    "type": "object",
                    "properties": {
                        "sentiment_score": {"type": "number"},
                        "event_magnitude": {"type": "number"},
                        "narrative": {"type": "string"},
                        "drivers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "label": {"type": "string"},
                                    "score": {"type": "number"},
                                    "detail": {"type": "string"},
                                },
                                "required": ["label", "score", "detail"],
                            },
                        },
                    },
                    "required": ["sentiment_score", "event_magnitude", "narrative", "drivers"],
                },
            },
        }

        try:
            response = httpx.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.settings.gemini_timeout_seconds,
            )
            if response.status_code != 200:
                raise ValueError(self._format_api_error(response))
            body = response.json()
            parts = body.get("candidates", [{}])[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts if isinstance(part, dict)).strip()
            if not text:
                raise ValueError("Gemini returned no text content")
            parsed = self._parse_response(text)
            return AIContextResult(
                sentiment_score=parsed["sentiment_score"],
                event_magnitude=parsed["event_magnitude"],
                narrative=parsed["narrative"],
                drivers=parsed["drivers"],
                provider=f"gemini:{self.settings.gemini_model}",
                generated_at=datetime.now(UTC),
            )
        except Exception as exc:
            return self._fallback_result(error=str(exc))

    def _format_api_error(self, response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = {}

        detail = str(body.get("error", {}).get("message", "")).strip() or response.text[:200]
        detail_lower = detail.lower()

        if "api key expired" in detail_lower or "api key not valid" in detail_lower:
            return "Gemini API key expired or is invalid. Update GEMINI_API_KEY."
        if "permission denied" in detail_lower or "permission" in detail_lower and "api key" in detail_lower:
            return "Gemini API key is not authorized for this request."
        if "not found" in detail_lower and self.settings.gemini_model.lower() in detail_lower:
            return f"Gemini model '{self.settings.gemini_model}' is not available for this API."

        return f"Gemini request failed ({response.status_code})."

    def _build_prompt(
        self,
        *,
        current_rate: float,
        bid: float,
        ask: float,
        change_2h_pct: float,
        change_8h_pct: float,
        change_24h_pct: float,
        brent: float | None,
        dxy: float | None,
        vix: float | None,
        btc_premium_pct: float | None,
        usdngn_official: float | None,
        official_parallel_spread_pct: float | None,
        usdghs: float | None,
        market_notes: str,
        news_headlines: str = "",
    ) -> str:
        spread = ask - bid
        spread_bps = (spread / max(current_rate, 1.0)) * 10_000

        def _fmt(val: float | None, prefix: str = "", suffix: str = "", decimals: int = 2) -> str:
            if val is None:
                return "unavailable"
            return f"{prefix}{val:.{decimals}f}{suffix}"

        notes_block = ""
        if market_notes.strip():
            notes_block = (
                "\n=== DESK NOTES (from the human trader) ===\n"
                f"{market_notes.strip()}\n"
            )

        news_block = ""
        if news_headlines.strip():
            news_block = (
                "\n=== LIVE NEWS FEED (scraped from real sources, most relevant first) ===\n"
                f"{news_headlines}\n"
            )

        return (
            "ROLE: You are the AI market intelligence engine for a Nigerian OTC foreign exchange desk "
            "that trades USD/NGN (specifically USDT/NGN on Quidax). You provide an INDEPENDENT sentiment "
            "assessment based on market observables and news.\n\n"

            "INDEPENDENCE RULE: You have NO access to the ML model's forecast. Your job is to assess "
            "what the quantitative model might be MISSING — breaking news, policy shifts, liquidity "
            "conditions, geopolitical events, and market microstructure signals. Your output will be "
            "combined with the ML forecast downstream. You must not try to predict the rate direction "
            "from the numbers alone — that is the ML model's job.\n\n"

            "CRITICAL RULES — FOLLOW EXACTLY:\n"
            "1. Base your analysis ONLY on the data and news provided below. Do NOT invent news, events, "
            "or facts not present in the inputs.\n"
            "2. When referencing a news headline, cite it by its ID (e.g., 'H3', 'H7'). If no headlines "
            "are provided or relevant, base analysis purely on the market data.\n"
            "3. Your narrative must be specific and verifiable — mention actual numbers, actual headline "
            "content, or actual market data points. Never use vague language like 'various factors' or "
            "'market conditions suggest'.\n"
            "4. If the news and data show nothing unusual, sentiment_score MUST be close to 0.0 and "
            "event_magnitude MUST be low. Do not manufacture drama.\n"
            "5. Focus on information the ML model CANNOT see: breaking news, policy announcements, "
            "liquidity stress (wide spreads), official-vs-parallel rate divergence, and geopolitical "
            "events.\n\n"

            f"TIMESTAMP: {datetime.now(UTC).strftime('%A, %B %d, %Y %H:%M UTC')}\n\n"

            "=== SPOT MARKET (live from Quidax) ===\n"
            f"Current USDT/NGN mid:       {current_rate:,.2f}\n"
            f"Bid:                        {bid:,.2f}\n"
            f"Ask:                        {ask:,.2f}\n"
            f"Spread:                     {spread:,.2f} ({spread_bps:.0f} bps)\n\n"

            "=== PRICE MOMENTUM ===\n"
            f"2-hour change:              {change_2h_pct:+.3f}%\n"
            f"8-hour change:              {change_8h_pct:+.3f}%\n"
            f"24-hour change:             {change_24h_pct:+.3f}%\n\n"

            "=== MACRO OBSERVABLES (live from Yahoo Finance) ===\n"
            f"Brent crude:                {_fmt(brent, prefix='$')}\n"
            f"Dollar index (DXY):         {_fmt(dxy)}\n"
            f"VIX (volatility/fear):      {_fmt(vix)}\n\n"

            "=== CRYPTO PREMIUM ===\n"
            f"BTC premium on Quidax:      {_fmt(btc_premium_pct, suffix='%')}\n\n"

            "=== NIGERIAN FX STRUCTURE ===\n"
            f"Official CBN rate (NAFEM):  {_fmt(usdngn_official, prefix='NGN ')}\n"
            f"Official-to-parallel gap:   {_fmt(official_parallel_spread_pct, suffix='%')}\n\n"

            "=== AFRICAN PEER CURRENCIES ===\n"
            f"USD/GHS (Ghana cedi):       {_fmt(usdghs)}\n"
            f"{notes_block}"
            f"{news_block}\n"

            "=== RESPONSE FORMAT (strict JSON) ===\n"
            "Return ONLY a valid JSON object with these fields:\n"
            "{\n"
            '  "sentiment_score": <float from -1.0 to +1.0>,\n'
            '  "event_magnitude": <float from 0.0 to 1.0>,\n'
            '  "narrative": "<3-5 sentence assessment>",\n'
            '  "drivers": [\n'
            '    {"label": "<driver name>", "score": <-1.0 to +1.0>, "detail": "<one sentence with evidence>"}\n'
            "  ]\n"
            "}\n\n"

            "=== FIELD DEFINITIONS ===\n"
            "sentiment_score:\n"
            "  Positive = upward pressure on USD/NGN (naira weakening) in next 2-4 hours.\n"
            "  Negative = downward pressure on USD/NGN (naira strengthening).\n"
            "  Zero = no new information beyond what raw market data shows.\n"
            "  Scale: +/-0.1 = minor, +/-0.3 = moderate, +/-0.5 = significant, +/-1.0 = extreme shock.\n\n"
            "event_magnitude:\n"
            "  0.0 = nothing notable. 0.3 = routine data release. 0.5 = important policy signal.\n"
            "  0.7 = major policy change or intervention. 1.0 = emergency (CBN emergency meeting, war, sanctions).\n\n"
            "narrative:\n"
            "  Written for OTC desk sales staff. Plain English, no jargon.\n"
            "  MUST reference specific data points or headline IDs (H1, H2, etc.) to support every claim.\n"
            "  Structure: [What is happening] -> [Why it matters for USD/NGN] -> [What to expect next 2-4 hours].\n\n"
            "drivers:\n"
            "  Return 3 to 6 drivers. Each represents one distinct force acting on the rate.\n"
            "  The sum of driver scores should approximately equal sentiment_score.\n"
            "  Each 'detail' field must cite evidence: a headline ID, a data point, or a desk note.\n"
            "  If a news headline is clearly rate-moving, it MUST be its own driver line.\n"
            "  If the spread is unusually wide (>100 bps), flag it as a liquidity driver.\n"
            "  If the official-parallel gap is widening or >5%, flag it as a structural driver."
        )

    def _parse_response(self, text: str) -> dict:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)

        if not cleaned.startswith("{"):
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)

        parsed = json.loads(cleaned)

        sentiment = float(parsed["sentiment_score"])
        sentiment = max(-1.0, min(1.0, sentiment))

        magnitude = float(parsed["event_magnitude"])
        magnitude = max(0.0, min(1.0, magnitude))

        narrative = str(parsed["narrative"]).strip()
        if not narrative:
            raise ValueError("Empty narrative")

        raw_drivers = parsed.get("drivers", [])
        drivers: list[AIDriver] = []
        for item in raw_drivers[:6]:
            if not isinstance(item, dict):
                continue
            label = str(item.get("label", "")).strip()
            detail = str(item.get("detail", "")).strip()
            if not label or not detail:
                continue
            score = max(-1.0, min(1.0, float(item.get("score", 0.0))))
            drivers.append(AIDriver(label=label, score=score, detail=detail))

        return {
            "sentiment_score": sentiment,
            "event_magnitude": magnitude,
            "narrative": narrative,
            "drivers": drivers,
        }

    def _fallback_result(self, error: str = "") -> AIContextResult:
        narrative = "AI assessment unavailable -- signal based on quantitative model only."
        if error:
            narrative += f" Error: {error}"
        return AIContextResult(
            sentiment_score=0.0,
            event_magnitude=0.0,
            narrative=narrative,
            drivers=[],
            provider=f"fallback:{error[:80]}" if error else "fallback",
            generated_at=datetime.now(UTC),
        )
