from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api import chat, ingestion
from app.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.db.session import init_db
from app.dependencies import initialize, shutdown

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize shared services and database for the application."""
    settings = get_settings()

    logger.info(
        "Starting %s (%s)",
        settings.app_name,
        settings.environment,
    )

    initialize(settings)
    await init_db()

    yield

    await shutdown()

    logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description=(
            "FastAPI backend implementing document ingestion and "
            "custom retrieval-augmented generation using Qdrant, "
            "Redis, SQLite, and an LLM."
        ),
        lifespan=lifespan,
    )

    register_exception_handlers(app)

    app.include_router(
        ingestion.router,
        prefix=settings.api_v1_prefix,
    )

    app.include_router(
        chat.router,
        prefix=settings.api_v1_prefix,
    )

    @app.get("/health", tags=["Health"])
    async def health() -> dict[str, str]:
        """Return API health status."""
        return {"status": "ok"}

    return app


app = create_app()