"""FastAPI application factory.

``create_app`` wires the object graph (config -> registry -> player ->
service) and installs the single place where domain errors are translated to
HTTP status codes. Each call builds an isolated app, which keeps tests and
embedding straightforward.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from importlib.resources import files

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from tts_daemon import __version__
from tts_daemon.api import http, openai_compat, websocket
from tts_daemon.api.schemas import HealthResponse
from tts_daemon.config import GatewayConfig, load_config
from tts_daemon.core.errors import (
    ConfigError,
    GatewayError,
    PlaybackError,
    ProviderUnavailableError,
    QueueFullError,
    SynthesisError,
    UnknownProviderError,
)
from tts_daemon.core.events import EventBus
from tts_daemon.core.service import SpeechService
from tts_daemon.players import create_player
from tts_daemon.providers.registry import create_default_registry

#: One place that decides how each domain error surfaces over HTTP.
ERROR_STATUS_CODES: tuple[tuple[type[Exception], int], ...] = (
    (UnknownProviderError, 404),
    (ProviderUnavailableError, 503),
    (QueueFullError, 429),
    (SynthesisError, 502),  # the upstream engine failed: literally a bad gateway
    (PlaybackError, 500),
    (ConfigError, 500),
    (GatewayError, 500),
    (ValueError, 422),
)


def status_code_for(exc: Exception) -> int:
    for exc_type, status in ERROR_STATUS_CODES:
        if isinstance(exc, exc_type):
            return status
    return 500


def create_app(config: GatewayConfig | None = None) -> FastAPI:
    """Build the gateway application (loading configuration if not given)."""
    config = config or load_config()

    events = EventBus()
    registry = create_default_registry(config)
    player = create_player(config.playback)
    service = SpeechService(config, registry, player, events)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        try:
            yield
        finally:
            service.close()

    app = FastAPI(
        title="tts-daemon",
        version=__version__,
        description=(
            "Local text-to-speech gateway: send text over HTTP or WebSocket, "
            "hear it through interchangeable TTS providers."
        ),
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.service = service

    if config.server.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server.cors_origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    @app.exception_handler(GatewayError)
    async def handle_gateway_error(_: Request, exc: GatewayError) -> JSONResponse:
        return JSONResponse(status_code=status_code_for(exc), content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def handle_value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=status_code_for(exc), content={"detail": str(exc)})

    app.include_router(http.router, prefix="/v1")
    app.include_router(openai_compat.router, prefix="/v1")
    app.include_router(websocket.router)

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    async def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    async def index() -> str:
        return _playground_html()

    return app


def _playground_html() -> str:
    """The static playground page, read from package data and cached."""
    global _PLAYGROUND_HTML
    if _PLAYGROUND_HTML is None:
        try:
            _PLAYGROUND_HTML = (
                files("tts_daemon.api").joinpath("static/index.html").read_text(encoding="utf-8")
            )
        except (FileNotFoundError, ModuleNotFoundError):
            _PLAYGROUND_HTML = _FALLBACK_HTML
    return _PLAYGROUND_HTML


_PLAYGROUND_HTML: str | None = None

#: Shown only if the packaged playground file is somehow missing.
_FALLBACK_HTML = (
    "<!doctype html><meta charset='utf-8'><title>tts-daemon</title>"
    "<h1>tts-daemon</h1><p>Interactive API docs: <a href='/docs'>/docs</a></p>"
)
