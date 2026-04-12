from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from fastapi import FastAPI, Header, HTTPException

app = FastAPI(title="RatePredict USDTNGN Proxy", version="1.0.0")


def _required_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _optional_env(name: str) -> str | None:
    value = os.getenv(name, "").strip()
    return value or None


def _curl_json(url: str, headers: dict[str, str], timeout_seconds: int = 20) -> dict[str, Any]:
    command = [
        "curl",
        "--silent",
        "--show-error",
        "--location",
        "--max-time",
        str(timeout_seconds),
        "--write-out",
        "\n%{http_code}",
    ]
    for name, value in headers.items():
        command.extend(["-H", f"{name}: {value}"])
    command.append(url)

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "curl failed")

    body, _, status_text = result.stdout.rpartition("\n")
    status_code = int(status_text.strip() or "0")
    body_snippet = " ".join(body.strip().split())[:280]

    if status_code >= 400:
        raise HTTPException(
            status_code=status_code,
            detail={
                "message": f"Upstream returned HTTP {status_code}",
                "body_snippet": body_snippet or "[empty body]",
            },
        )

    try:
        payload = json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Upstream returned non-JSON: {body_snippet or '[empty body]'}") from exc

    return payload


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/usdtngn")
def usdtngn(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    proxy_token = _optional_env("LIVE_USDTNGN_PROXY_TOKEN")
    if proxy_token:
        expected = f"Bearer {proxy_token}"
        if authorization != expected:
            raise HTTPException(status_code=401, detail="Unauthorized")

    url = _required_env("QBOT_USDTNGN_RATE_URL")
    payload = _curl_json(
        url,
        headers={
            "x-service-token": _required_env("QBOT_SERVICE_TOKEN"),
            "CF-Access-Client-Id": _required_env("QBOT_CF_ACCESS_CLIENT_ID"),
            "CF-Access-Client-Secret": _required_env("QBOT_CF_ACCESS_CLIENT_SECRET"),
            "Accept": "application/json",
            "User-Agent": "curl/8.7.1",
        },
    )

    if payload.get("status") != "success":
        raise HTTPException(status_code=502, detail={"message": "Unexpected upstream payload", "payload": payload})

    current = payload.get("data", {}).get("current")
    if not isinstance(current, dict):
        raise HTTPException(status_code=502, detail={"message": "Missing upstream current payload", "payload": payload})

    return {
        "buyRate": current["buyRate"],
        "sellRate": current["sellRate"],
        "midRate": current["midRate"],
        "rateAsAt": current["rateAsAt"],
        "provider": current.get("provider", "unknown"),
        "source": current.get("source", "proxy"),
    }
