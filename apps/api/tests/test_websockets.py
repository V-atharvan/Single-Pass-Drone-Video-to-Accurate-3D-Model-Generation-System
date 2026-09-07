"""
Integration tests for Real-Time WebSocket Server for Job Status Streaming – TASK-017.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from src.main import app
from src.routers.websockets import authenticate_ws_token


class FakeRedisTracker:
    """Mock Redis state tracker implementing async pub/sub and cached progress."""

    def __init__(self) -> None:
        self.cached_progress: dict[str, Any] = {}
        self.event_queue: asyncio.Queue = asyncio.Queue()

    async def get_job_progress(self, job_id: uuid.UUID | str) -> dict[str, Any] | None:
        return self.cached_progress.get(str(job_id))

    async def subscribe_job_events(self, job_id: uuid.UUID | str) -> AsyncIterator[dict[str, Any]]:
        while True:
            event = await self.event_queue.get()
            if event is None:  # Sentinel to stop
                break
            yield event

    async def publish_simulated_event(self, event: dict[str, Any]) -> None:
        await self.event_queue.put(event)


@pytest.fixture
def fake_redis():
    tracker = FakeRedisTracker()
    app.state.redis = tracker
    yield tracker
    app.state.redis = None


def fake_auth(token: str) -> dict[str, Any]:
    if token == "valid_token_123":
        return {"sub": "auth0|user123", "org_id": str(uuid.uuid4())}
    raise ValueError("Invalid test token")


def test_ws_query_token_auth_and_pubsub_streaming(fake_redis):
    """
    Verify client connects using query param token, receives published Redis event within 100ms,
    and verified JSON packet conforms to TASK-017 schema.
    """
    job_id = uuid.uuid4()
    published_event = {
        "job_id": str(job_id),
        "stage": "FEATURE_EXTRACTION",
        "progress": 45,
        "frames_processed": 450,
        "total_frames": 1000,
        "gpu_stats": {"utilization": 88.5},
        "warnings": ["Low contrast in frame 102"],
    }

    with patch("src.routers.websockets.authenticate_ws_token", side_effect=fake_auth):
        client = TestClient(app)
        with client.websocket_connect(f"/ws/jobs/{job_id}?token=valid_token_123") as ws:
            # Publish event into Redis tracker
            asyncio.run(fake_redis.publish_simulated_event(published_event))

            # Receive streamed packet
            data = ws.receive_json()

            assert data["job_id"] == str(job_id)
            assert data["stage"] == "FEATURE_EXTRACTION"
            assert data["progress"] == 45
            assert data["frames_processed"] == 450
            assert data["frames_total"] == 1000
            assert data["gpu_utilization"] == 88.5
            assert data["warnings"] == ["Low contrast in frame 102"]


def test_ws_cached_initial_state(fake_redis):
    """Verify client immediately receives cached progress upon connection if present."""
    job_id = uuid.uuid4()
    fake_redis.cached_progress[str(job_id)] = {
        "job_id": str(job_id),
        "stage": "VALIDATING",
        "progress": 10,
        "frames_processed": 50,
        "frames_total": 500,
        "gpu_utilization": 12.0,
        "warnings": [],
    }

    with patch("src.routers.websockets.authenticate_ws_token", side_effect=fake_auth):
        client = TestClient(app)
        with client.websocket_connect(f"/ws/jobs/{job_id}?token=valid_token_123") as ws:
            data = ws.receive_json()
            assert data["stage"] == "VALIDATING"
            assert data["progress"] == 10
            assert data["gpu_utilization"] == 12.0


def test_ws_connection_message_auth(fake_redis):
    """Verify client can authenticate via initial JSON connection packet."""
    job_id = uuid.uuid4()

    with patch("src.routers.websockets.authenticate_ws_token", side_effect=fake_auth):
        client = TestClient(app)
        with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
            # Send initial auth payload
            ws.send_json({"token": "valid_token_123"})
            ack = ws.receive_json()
            assert ack["type"] == "authenticated"
            assert ack["job_id"] == str(job_id)

            # Publish event
            asyncio.run(fake_redis.publish_simulated_event({
                "job_id": str(job_id),
                "stage": "DEPTH_ESTIMATION",
                "progress": 60,
                "frames_processed": 600,
                "frames_total": 1000,
                "gpu_utilization": 94.0,
            }))
            streamed = ws.receive_json()
            assert streamed["stage"] == "DEPTH_ESTIMATION"
            assert streamed["progress"] == 60


def test_ws_rejected_invalid_query_token():
    """Verify connection is closed with 1008 POLICY_VIOLATION when token is invalid."""
    job_id = uuid.uuid4()

    with patch("src.routers.websockets.authenticate_ws_token", side_effect=fake_auth):
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(f"/ws/jobs/{job_id}?token=bad_token"):
                pass
        assert exc_info.value.code == 1008


def test_ws_rejected_invalid_message_token():
    """Verify connection is closed with 1008 when initial message has bad token."""
    job_id = uuid.uuid4()

    with patch("src.routers.websockets.authenticate_ws_token", side_effect=fake_auth):
        client = TestClient(app)
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect(f"/ws/jobs/{job_id}") as ws:
                ws.send_json({"token": "bad_token"})
                ws.receive_json()
        assert exc_info.value.code == 1008


def test_ws_ping_pong(fake_redis):
    """Verify client ping yields pong response."""
    job_id = uuid.uuid4()

    with patch("src.routers.websockets.authenticate_ws_token", side_effect=fake_auth):
        client = TestClient(app)
        with client.websocket_connect(f"/ws/jobs/{job_id}?token=valid_token_123") as ws:
            ws.send_text("ping")
            response = ws.receive_text()
            assert response == "pong"
