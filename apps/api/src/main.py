"""
FastAPI Control Plane API – Single-Pass 3D Reconstruction Platform
Entry point: `uvicorn apps.api.src.main:app --reload`
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db.base import init_engine
from .routers import flights, health, organizations, projects, reconstruction_jobs, uploads, websockets

# ---------------------------------------------------------------------------
# Lifespan – connect / disconnect on startup / shutdown
# ---------------------------------------------------------------------------

_redis_tracker = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage resource lifecycle: DB engine pool and Redis connection."""
    global _redis_tracker

    # Initialise SQLAlchemy async engine pool
    init_engine(
        url=os.environ.get(
            "DATABASE_URL",
            "postgresql+asyncpg://single_pass_3d:devpassword@localhost:5432/single_pass_3d_dev",
        ),
        echo=os.environ.get("SQL_ECHO", "false").lower() == "true",
    )

    # Initialise Redis state tracker
    try:
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "packages", "shared", "python"))
        from packages.shared.python.redis_client import RedisStateTracker  # type: ignore
        _redis_tracker = await RedisStateTracker.create(
            url=os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        )
        app.state.redis = _redis_tracker
    except Exception:
        # Redis is optional at startup; health probe will report status
        app.state.redis = None

    yield

    # Shutdown
    if _redis_tracker is not None:
        await _redis_tracker.close()


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Single-Pass 3D Reconstruction API",
    description=(
        "Control plane API for the Single-Pass Drone Video to Accurate 3D Model Generation System. "
        "Manages projects, flights, reconstruction jobs, models, and exports."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS – allow Next.js frontend origins
# ---------------------------------------------------------------------------

ALLOWED_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:3001",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in ALLOWED_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router, prefix="/health", tags=["Health"])
app.include_router(organizations.router, prefix="/organizations", tags=["Organizations"])
app.include_router(projects.router, prefix="/projects", tags=["Projects"])
app.include_router(uploads.router)
app.include_router(flights.project_flights_router)
app.include_router(flights.direct_flights_router)
app.include_router(reconstruction_jobs.flight_jobs_router)
app.include_router(reconstruction_jobs.direct_jobs_router)
app.include_router(websockets.router)
