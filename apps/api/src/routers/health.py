"""
Health probe endpoints.

GET /health/liveness  – Always returns 200 if the process is alive.
GET /health/readiness – Returns 200 only if PostgreSQL and Redis are reachable.
"""
from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/liveness", summary="Liveness probe")
async def liveness() -> dict:
    """Kubernetes liveness probe – returns 200 if process is running."""
    return {"status": "alive"}


@router.get("/readiness", summary="Readiness probe")
async def readiness(request: Request) -> JSONResponse:
    """
    Readiness probe – checks PostgreSQL and Redis connectivity.
    Returns HTTP 200 when both are UP, HTTP 503 otherwise.
    """
    checks: dict[str, str] = {}
    all_up = True

    # ---- PostgreSQL ----
    try:
        from src.db.base import _engine  # noqa: PLC0415
        if _engine is None:
            raise RuntimeError("Engine not initialised")
        async with _engine.connect() as conn:
            await conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        checks["postgres"] = "UP"
    except Exception as exc:
        checks["postgres"] = f"DOWN: {exc}"
        all_up = False

    # ---- Redis ----
    redis_client = getattr(request.app.state, "redis", None)
    if redis_client is None:
        checks["redis"] = "DOWN: not initialised"
        all_up = False
    else:
        try:
            ok = await redis_client.ping()
            checks["redis"] = "UP" if ok else "DOWN: ping failed"
            if not ok:
                all_up = False
        except Exception as exc:
            checks["redis"] = f"DOWN: {exc}"
            all_up = False

    status_code = 200 if all_up else 503
    return JSONResponse(content={"status": "ready" if all_up else "not_ready", "checks": checks}, status_code=status_code)
