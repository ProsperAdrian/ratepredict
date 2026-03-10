from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import get_settings
from app.services.inference import LiveInferenceService


def create_app() -> FastAPI:
    settings = get_settings()
    templates = Jinja2Templates(directory=str(settings.base_dir / "app" / "templates"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = LiveInferenceService(settings)
        app.state.settings = settings
        app.state.service = service
        try:
            service.refresh()
        except Exception:
            pass

        refresh_task = None
        if settings.auto_refresh_enabled:
            refresh_task = asyncio.create_task(_auto_refresh_loop(service, settings.auto_refresh_seconds))
        try:
            yield
        finally:
            if refresh_task:
                refresh_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await refresh_task

    app = FastAPI(title=settings.app_title, version=settings.app_version, lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(settings.base_dir / "app" / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard(request: Request) -> HTMLResponse:
        service: LiveInferenceService = request.app.state.service
        snapshot = service.get_or_refresh()
        return templates.TemplateResponse(
            name="index.html",
            context={
                "request": request,
                "snapshot": snapshot,
                "settings": settings,
            },
        )

    @app.get("/health")
    async def health(request: Request) -> dict:
        service: LiveInferenceService = request.app.state.service
        snapshot = service.get_or_refresh()
        return {"status": "ok", "model_version": snapshot.model_version, "as_of": snapshot.as_of}

    @app.get("/api/signal")
    async def get_signal(request: Request) -> dict:
        service: LiveInferenceService = request.app.state.service
        snapshot = service.get_or_refresh()
        return snapshot.model_dump(mode="json")

    @app.post("/api/signal/refresh")
    async def refresh_signal(request: Request) -> dict:
        service: LiveInferenceService = request.app.state.service
        try:
            snapshot = service.refresh()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return snapshot.model_dump(mode="json")

    return app


async def _auto_refresh_loop(service: LiveInferenceService, refresh_seconds: int) -> None:
    while True:
        await asyncio.sleep(refresh_seconds)
        try:
            service.refresh()
        except Exception:
            continue


app = create_app()
