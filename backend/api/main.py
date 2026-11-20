"""FastAPI application factory. The API is the only thing front ends talk to (Streamlit now, React later)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.config.settings import get_settings, setup_logging
from backend.errors import (FinderError, IndexUnavailableError, InvalidImageError, ModelUnavailableError,
                            NoFaceError)

log = logging.getLogger(__name__)


class State:
    service = None            # FinderService, created at startup (None if the models are unavailable)
    startup_error: str | None = None


state = State()


def get_service():
    from fastapi import HTTPException
    if state.service is None:
        raise HTTPException(503, state.startup_error or "Service not initialised")
    return state.service


@asynccontextmanager
async def lifespan(app: FastAPI):
    s = get_settings()
    setup_logging(s, "api")
    try:
        from backend.service import FinderService
        state.service = FinderService(s)
    except ModelUnavailableError as e:      # keep the API up so /health can explain what is wrong
        state.startup_error = str(e)
        log.error("Startup failed: %s", e)
    yield


def create_app() -> FastAPI:
    from backend.api.routes import evaluation, health, indexing, search
    app = FastAPI(title="Studio Photo Finder API", version="0.1.0", lifespan=lifespan)

    @app.exception_handler(InvalidImageError)
    async def _bad_image(_: Request, e: InvalidImageError):
        return JSONResponse({"detail": str(e), "code": "invalid_image"}, status_code=400)

    @app.exception_handler(NoFaceError)
    async def _no_face(_: Request, e: NoFaceError):
        return JSONResponse({"detail": str(e), "code": "no_face"}, status_code=422)

    @app.exception_handler(IndexUnavailableError)
    async def _no_index(_: Request, e: IndexUnavailableError):
        return JSONResponse({"detail": str(e), "code": "index_unavailable"}, status_code=503)

    @app.exception_handler(ValueError)
    async def _value(_: Request, e: ValueError):
        return JSONResponse({"detail": str(e), "code": "bad_request"}, status_code=400)

    @app.exception_handler(FinderError)
    async def _finder(_: Request, e: FinderError):
        log.error("Unhandled domain error: %s", e)
        return JSONResponse({"detail": str(e), "code": "error"}, status_code=500)

    for r in (health.router, search.router, indexing.router, evaluation.router):
        app.include_router(r)
    return app


app = create_app()
