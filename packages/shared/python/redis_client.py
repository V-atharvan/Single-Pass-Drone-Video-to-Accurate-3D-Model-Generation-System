"""
RedisStateTracker – async Redis client for job progress caching, pub/sub fan-out,
and worker liveness heartbeats.

Channel schema:
    Job hash key:   job:{job_id}
    Pub/Sub channel: job_events:{job_id}
    Worker key:     worker_heartbeat:{worker_id}
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, AsyncIterator, Optional
from uuid import UUID

import redis.asyncio as aioredis

REDIS_URL: str = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
HEARTBEAT_TTL: int = 15  # seconds


class RedisStateTracker:
    """
    Async Redis state tracker for reconstruction job progress.
    Must be initialised with `await RedisStateTracker.create()`.
    """

    def __init__(self, client: aioredis.Redis) -> None:
        self._client = client

    @classmethod
    async def create(cls, url: str = REDIS_URL) -> "RedisStateTracker":
        client = await aioredis.from_url(url, encoding="utf-8", decode_responses=True)
        return cls(client)

    async def close(self) -> None:
        await self._client.aclose()

    # ------------------------------------------------------------------
    # Job progress
    # ------------------------------------------------------------------

    async def set_job_progress(
        self,
        job_id: UUID | str,
        stage: str,
        progress: int,
        frames_processed: int,
        total_frames: int,
        gpu_stats: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Atomically update job progress hash and publish an event to the
        job's pub/sub channel for WebSocket fan-out.
        """
        job_key = f"job:{job_id}"
        payload: dict[str, Any] = {
            "job_id": str(job_id),
            "stage": stage,
            "progress": progress,
            "frames_processed": frames_processed,
            "total_frames": total_frames,
            "gpu_stats": gpu_stats or {},
            "updated_at": time.time(),
        }

        # Atomic pipeline: HSET + PUBLISH
        async with self._client.pipeline(transaction=True) as pipe:
            pipe.hset(job_key, mapping={k: json.dumps(v) if isinstance(v, dict) else str(v) for k, v in payload.items()})
            pipe.publish(f"job_events:{job_id}", json.dumps(payload))
            await pipe.execute()

    async def get_job_progress(self, job_id: UUID | str) -> Optional[dict[str, Any]]:
        """Retrieve the latest progress snapshot for a job."""
        job_key = f"job:{job_id}"
        raw = await self._client.hgetall(job_key)
        if not raw:
            return None
        result: dict[str, Any] = {}
        for k, v in raw.items():
            try:
                result[k] = json.loads(v)
            except (json.JSONDecodeError, ValueError):
                result[k] = v
        return result

    async def delete_job_progress(self, job_id: UUID | str) -> None:
        """Remove the cached job progress hash (call on terminal states)."""
        await self._client.delete(f"job:{job_id}")

    # ------------------------------------------------------------------
    # Pub/Sub
    # ------------------------------------------------------------------

    async def subscribe_job_events(self, job_id: UUID | str) -> AsyncIterator[dict[str, Any]]:
        """
        Async generator yielding parsed job event messages from pub/sub.
        Caller is responsible for breaking out of the loop.
        """
        pubsub = self._client.pubsub()
        channel = f"job_events:{job_id}"
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        yield json.loads(message["data"])
                    except json.JSONDecodeError:
                        yield {"raw": message["data"]}
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    # ------------------------------------------------------------------
    # Worker heartbeats
    # ------------------------------------------------------------------

    async def record_worker_heartbeat(self, worker_id: str, metadata: Optional[dict[str, Any]] = None) -> None:
        """
        Record a liveness heartbeat for a worker with a TTL of 15 seconds.
        If the key expires, the worker is considered dead.
        """
        key = f"worker_heartbeat:{worker_id}"
        value = json.dumps({"worker_id": worker_id, "ts": time.time(), **(metadata or {})})
        await self._client.set(key, value, ex=HEARTBEAT_TTL)

    async def get_worker_heartbeat(self, worker_id: str) -> Optional[dict[str, Any]]:
        """Return the last heartbeat metadata for a worker, or None if expired/absent."""
        raw = await self._client.get(f"worker_heartbeat:{worker_id}")
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"raw": raw}

    async def list_active_workers(self, pattern: str = "worker_heartbeat:*") -> list[str]:
        """Return a list of worker IDs with active (non-expired) heartbeat keys."""
        keys = await self._client.keys(pattern)
        return [k.replace("worker_heartbeat:", "") for k in keys]

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    async def ping(self) -> bool:
        """Return True if Redis is reachable."""
        try:
            return await self._client.ping()
        except Exception:
            return False
