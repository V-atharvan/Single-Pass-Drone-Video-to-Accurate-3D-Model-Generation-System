"""
Real-time WebSocket router for job status streaming – TASK-017.
Subscribes to Redis pub/sub channel `job_events:{job_id}` and streams
JSON progress packets to authenticated clients.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from uuid import UUID

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect, status

from src.auth.jwt import verify_token

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSockets"])


def authenticate_ws_token(token: str) -> dict[str, Any]:
    """
    Validate Bearer token.
    Decoupled helper so it can be cleanly mocked or overridden in tests.
    """
    return verify_token(token)


def format_job_event(raw_event: dict[str, Any]) -> dict[str, Any]:
    """
    Format event payload into standardized schema matching TASK-017:
    job_id, stage, progress, frames_processed, frames_total, gpu_utilization, warnings.
    """
    gpu_util = raw_event.get("gpu_utilization")
    if gpu_util is None and isinstance(raw_event.get("gpu_stats"), dict):
        gpu_util = raw_event["gpu_stats"].get("utilization", 0.0)

    formatted = {
        "job_id": str(raw_event.get("job_id", "")),
        "stage": raw_event.get("stage", ""),
        "progress": raw_event.get("progress", 0),
        "frames_processed": raw_event.get("frames_processed", 0),
        "frames_total": raw_event.get("total_frames", raw_event.get("frames_total", 0)),
        "gpu_utilization": gpu_util if gpu_util is not None else 0.0,
        "warnings": raw_event.get("warnings", []),
    }
    # Preserve any additional metadata
    for k, v in raw_event.items():
        if k not in formatted:
            formatted[k] = v
    return formatted


@router.websocket("/ws/jobs/{job_id}")
async def job_status_websocket(
    websocket: WebSocket,
    job_id: UUID,
    token: Optional[str] = Query(None),
) -> None:
    """
    WebSocket endpoint streaming real-time reconstruction job updates.
    Authentication is accepted via:
      1. Query param: `?token=<jwt>`
      2. Connection message: initial JSON packet `{"token": "<jwt>"}`
    """
    # ------------------------------------------------------------------
    # 1. Authentication
    # ------------------------------------------------------------------
    user_claims: Optional[dict[str, Any]] = None

    if token:
        try:
            user_claims = authenticate_ws_token(token)
            await websocket.accept()
        except Exception as exc:
            logger.warning("WebSocket auth failed via query param: %s", exc)
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
            return
    else:
        # Accept connection temporarily to receive auth message
        await websocket.accept()
        try:
            auth_msg = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
            msg_token = auth_msg.get("token")
            if not msg_token:
                raise ValueError("Missing token in auth packet")
            user_claims = authenticate_ws_token(msg_token)
            await websocket.send_json({"type": "authenticated", "job_id": str(job_id)})
        except Exception as exc:
            logger.warning("WebSocket auth failed via connection message: %s", exc)
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION, reason="Unauthorized")
            return

    # ------------------------------------------------------------------
    # 2. Redis State Tracker & Initial State
    # ------------------------------------------------------------------
    redis_tracker = getattr(websocket.app.state, "redis", None)
    if redis_tracker is None:
        logger.error("RedisStateTracker not initialised in app.state")
        await websocket.close(code=status.WS_1011_INTERNAL_ERROR, reason="Service unavailable")
        return

    # Send initial cached progress if available
    try:
        cached_progress = await redis_tracker.get_job_progress(job_id)
        if cached_progress:
            await websocket.send_json(format_job_event(cached_progress))
    except Exception as exc:
        logger.debug("Could not fetch cached progress for job %s: %s", job_id, exc)

    # ------------------------------------------------------------------
    # 3. Stream Redis Pub/Sub Events
    # ------------------------------------------------------------------
    async def redis_event_streamer() -> None:
        async for event in redis_tracker.subscribe_job_events(job_id):
            formatted_packet = format_job_event(event)
            await websocket.send_json(formatted_packet)

    async def client_heartbeat_listener() -> None:
        while True:
            # Keep receiving to detect client disconnects and ping messages
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_text("pong")

    streamer_task = asyncio.create_task(redis_event_streamer())
    listener_task = asyncio.create_task(client_heartbeat_listener())

    try:
        done, pending = await asyncio.wait(
            [streamer_task, listener_task],
            return_when=asyncio.FIRST_COMPLETED,
        )
        for task in pending:
            task.cancel()
        for task in done:
            # Raise exception if one failed unexpectedly
            exc = task.exception()
            if exc and not isinstance(exc, (WebSocketDisconnect, asyncio.CancelledError)):
                logger.debug("WebSocket subtask ended with: %s", exc)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client for job %s", job_id)
    except Exception as exc:
        logger.warning("WebSocket error for job %s: %s", job_id, exc)
    finally:
        streamer_task.cancel()
        listener_task.cancel()
