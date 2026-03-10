from __future__ import annotations

from datetime import UTC, datetime

import httpx

from app.config import Settings
from app.schemas import MarketBrief


class GeminiBriefService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def generate(
        self,
        *,
        signal: str,
        confidence_label: str,
        predicted_return_pct: float,
        live_last_trade: float,
        forecast_price: float,
        top_features: list[dict[str, float | str]],
        threshold_bps: float,
    ) -> MarketBrief:
        if not self.settings.gemini_api_key:
            return self._fallback_brief(
                provider="fallback",
                signal=signal,
                confidence_label=confidence_label,
                predicted_return_pct=predicted_return_pct,
                live_last_trade=live_last_trade,
                forecast_price=forecast_price,
                top_features=top_features,
                threshold_bps=threshold_bps,
            )

        prompt = self._build_prompt(
            signal=signal,
            confidence_label=confidence_label,
            predicted_return_pct=predicted_return_pct,
            live_last_trade=live_last_trade,
            forecast_price=forecast_price,
            top_features=top_features,
            threshold_bps=threshold_bps,
        )
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.settings.gemini_model}:generateContent"
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": 300},
        }
        headers = {"x-goog-api-key": self.settings.gemini_api_key, "Content-Type": "application/json"}

        try:
            response = httpx.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.settings.gemini_timeout_seconds,
            )
            response.raise_for_status()
            body = response.json()
            content = (
                body.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            if not content:
                raise ValueError("Gemini returned no text content")
            return MarketBrief(provider=f"gemini:{self.settings.gemini_model}", content=content, generated_at=datetime.now(UTC))
        except Exception:
            return self._fallback_brief(
                provider="fallback_after_gemini_error",
                signal=signal,
                confidence_label=confidence_label,
                predicted_return_pct=predicted_return_pct,
                live_last_trade=live_last_trade,
                forecast_price=forecast_price,
                top_features=top_features,
                threshold_bps=threshold_bps,
            )

    def _build_prompt(
        self,
        *,
        signal: str,
        confidence_label: str,
        predicted_return_pct: float,
        live_last_trade: float,
        forecast_price: float,
        top_features: list[dict[str, float | str]],
        threshold_bps: float,
    ) -> str:
        feature_lines = "\n".join(
            f"- {feature['name']}: value={feature['value']:.6f}, importance={feature['importance']:.4f}"
            for feature in top_features
        )
        return (
            "You are writing an internal OTC desk market brief.\n"
            "Keep it to 4 short bullets followed by a one-line recommendation.\n"
            "Be factual, concise, and risk-aware. Do not overstate confidence.\n\n"
            f"Signal: {signal}\n"
            f"Confidence: {confidence_label}\n"
            f"Predicted 2h move: {predicted_return_pct:.3f}%\n"
            f"Live last trade: {live_last_trade:.4f}\n"
            f"Forecast price: {forecast_price:.4f}\n"
            f"Trade threshold: {threshold_bps:.1f} bps\n"
            "Top model drivers:\n"
            f"{feature_lines}\n"
        )

    def _fallback_brief(
        self,
        *,
        provider: str,
        signal: str,
        confidence_label: str,
        predicted_return_pct: float,
        live_last_trade: float,
        forecast_price: float,
        top_features: list[dict[str, float | str]],
        threshold_bps: float,
    ) -> MarketBrief:
        drivers = ", ".join(feature["name"] for feature in top_features[:3]) or "no dominant drivers"
        content = (
            f"- The model is signaling `{signal}` with `{confidence_label}` confidence.\n"
            f"- It projects a 2-hour move of `{predicted_return_pct:.3f}%`, from `{live_last_trade:.4f}` to `{forecast_price:.4f}`.\n"
            f"- The signal clears the configured trade filter of `{threshold_bps:.1f}` bps only if the predicted edge remains above threshold after desk review.\n"
            f"- Current model leadership comes primarily from `{drivers}`; treat this as a model-derived view, not a certainty.\n"
            f"Recommendation: {signal.upper()} only within desk risk limits and only after confirming market conditions still match the latest feed."
        )
        return MarketBrief(provider=provider, content=content, generated_at=datetime.now(UTC))
