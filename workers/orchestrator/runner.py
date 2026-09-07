"""
Reconstruction Worker Dispatcher & SQS Consumer Loop – TASK-018.
Asynchronous worker daemon that polls SQS, acquires job locks, and orchestrates pipeline execution.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import UUID

# Ensure monorepo packages are resolvable
_ROOT_DIR = Path(__file__).resolve().parents[2]
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))
_APPS_API_DIR = _ROOT_DIR / "apps" / "api"
if str(_APPS_API_DIR) not in sys.path:
    sys.path.insert(0, str(_APPS_API_DIR))

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from packages.shared.python.queue import SQSQueueClient
from packages.shared.python.redis_client import RedisStateTracker
from src.db.base import _session_factory, get_session, init_engine
from src.db.models import Flight, JobStatus, ReconstructionJob

logger = logging.getLogger("orchestrator.runner")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)


class JobWorkerDaemon:
    """
    Asynchronous reconstruction worker daemon.
    Polls Amazon SQS for reconstruction jobs, coordinates state across PostgreSQL
    and Redis, acquires distributed locks, and dispatches processing stages.
    """

    def __init__(
        self,
        worker_id: Optional[str] = None,
        queue_client: Optional[SQSQueueClient] = None,
        redis_tracker: Optional[RedisStateTracker] = None,
        session_factory: Optional[async_sessionmaker[AsyncSession]] = None,
        poll_wait_seconds: int = 20,
        heartbeat_interval_seconds: int = 10,
    ) -> None:
        self.worker_id: str = worker_id or f"worker-{os.getpid()}-{os.urandom(3).hex()}"
        self.queue_client: SQSQueueClient = queue_client or SQSQueueClient()
        self.redis_tracker: Optional[RedisStateTracker] = redis_tracker
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = session_factory or _session_factory
        self.poll_wait_seconds: int = poll_wait_seconds
        self.heartbeat_interval_seconds: int = heartbeat_interval_seconds

        self._running: bool = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self.current_job_id: Optional[UUID] = None

    async def _ensure_dependencies(self) -> None:
        """Initialize Redis and DB connections if not provided at instantiation."""
        if self.redis_tracker is None:
            self.redis_tracker = await RedisStateTracker.create(
                url=os.environ.get("REDIS_URL", "redis://localhost:6379/0")
            )
        if self.session_factory is None:
            init_engine(
                url=os.environ.get(
                    "DATABASE_URL",
                    "postgresql+asyncpg://single_pass_3d:devpassword@localhost:5432/single_pass_3d_dev",
                )
            )
            from src.db.base import _session_factory as sf
            self.session_factory = sf

    def _get_db_session(self) -> AsyncSession:
        if self.session_factory is None:
            raise RuntimeError("Database session factory is not initialised.")
        return self.session_factory()

    # ------------------------------------------------------------------
    # Lifecycle and Signal Handling
    # ------------------------------------------------------------------

    def register_signal_handlers(self) -> None:
        """Attach SIGINT and SIGTERM handlers for graceful worker shutdown."""
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: asyncio.create_task(self.handle_shutdown_signal(s)))
            except (NotImplementedError, AttributeError):
                # Signals not fully supported in non-main threads or some Windows setups
                pass

    async def handle_shutdown_signal(self, sig: signal.Signals) -> None:
        """Handle SIGTERM / SIGINT: release active job locks and stop worker."""
        logger.info("Received termination signal %s. Initiating graceful shutdown...", sig.name)
        await self.stop()

    async def start(self) -> None:
        """Start worker polling loop and background liveness heartbeat."""
        await self._ensure_dependencies()
        self.register_signal_handlers()
        self._running = True

        logger.info("Worker [%s] started. Polling queue: %s", self.worker_id, self.queue_client.queue_url)

        # Start liveness heartbeat loop
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

        # Main consumer loop
        try:
            while self._running:
                try:
                    await self.poll_and_process_once()
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.exception("Unexpected error in consumer loop: %s", exc)
                    await asyncio.sleep(2.0)
        finally:
            await self.stop()

    async def stop(self) -> None:
        """Stop worker daemon, release locks, and clean up background tasks."""
        if not self._running and self.current_job_id is None:
            return

        self._running = False
        logger.info("Worker [%s] stopping...", self.worker_id)

        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Handle stalled in-flight job if interrupted during execution
        if self.current_job_id and self.redis_tracker:
            lock_key = f"lock:job:{self.current_job_id}"
            try:
                # Release lock so another worker can recover the job
                await self.redis_tracker._client.delete(lock_key)
                logger.warning(
                    "Worker [%s] released lock for in-flight job %s on shutdown.",
                    self.worker_id,
                    self.current_job_id,
                )
            except Exception as exc:
                logger.error("Failed to release lock on shutdown: %s", exc)
            self.current_job_id = None

        logger.info("Worker [%s] stopped successfully.", self.worker_id)

    async def _heartbeat_loop(self) -> None:
        """Periodically record worker liveness in Redis."""
        while self._running:
            try:
                if self.redis_tracker:
                    await self.redis_tracker.record_worker_heartbeat(
                        self.worker_id,
                        {
                            "status": "processing" if self.current_job_id else "idle",
                            "current_job": str(self.current_job_id) if self.current_job_id else None,
                        },
                    )
            except Exception as exc:
                logger.warning("Heartbeat update failed: %s", exc)
            await asyncio.sleep(self.heartbeat_interval_seconds)

    # ------------------------------------------------------------------
    # Message Consumption & Job Processing
    # ------------------------------------------------------------------

    async def poll_and_process_once(self) -> bool:
        """
        Poll SQS queue once and process a single message if received.
        Returns True if a message was consumed and processed, False otherwise.
        """
        messages = await asyncio.to_thread(
            self.queue_client.receive_messages,
            max_messages=1,
            wait_time_seconds=self.poll_wait_seconds,
        )
        if not messages:
            return False

        message = messages[0]
        return await self.process_one_message(message)

    async def process_one_message(self, message: dict[str, Any]) -> bool:
        """
        Parse SQS message, acquire distributed lock, fetch PostgreSQL records,
        transition job status to VALIDATING in DB and Redis, and dispatch execution.
        """
        receipt_handle = message.get("ReceiptHandle")
        body_raw = message.get("Body", "{}")

        try:
            body = json.loads(body_raw)
            job_id_str = body.get("job_id")
            if not job_id_str:
                logger.error("Received SQS message missing 'job_id': %s", body_raw)
                if receipt_handle:
                    await asyncio.to_thread(self.queue_client.delete_message, receipt_handle)
                return False
            job_id = UUID(job_id_str)
        except Exception as exc:
            logger.error("Failed to parse SQS message body: %s", exc)
            if receipt_handle:
                await asyncio.to_thread(self.queue_client.delete_message, receipt_handle)
            return False

        logger.info("Worker [%s] received message for job %s", self.worker_id, job_id)

        # 1. Acquire distributed lock in Redis to prevent duplicate processing
        if self.redis_tracker:
            lock_key = f"lock:job:{job_id}"
            acquired = await self.redis_tracker._client.set(
                lock_key, self.worker_id, nx=True, ex=3600
            )
            if not acquired:
                logger.warning("Job %s is already locked by another worker. Skipping.", job_id)
                return False

        self.current_job_id = job_id

        try:
            # 2. Fetch Job and Flight records from PostgreSQL
            async with self._get_db_session() as session:
                job_result = await session.execute(
                    select(ReconstructionJob).where(ReconstructionJob.id == job_id)
                )
                job = job_result.scalar_one_or_none()

                if job is None:
                    logger.error("Job %s not found in PostgreSQL. Deleting message.", job_id)
                    if receipt_handle:
                        await asyncio.to_thread(self.queue_client.delete_message, receipt_handle)
                    return False

                # If job was cancelled before worker pickup, abort
                if job.status in (JobStatus.CANCELLED, JobStatus.FAILED):
                    logger.info("Job %s is already %s. Acknowledging message.", job_id, job.status.value)
                    if receipt_handle:
                        await asyncio.to_thread(self.queue_client.delete_message, receipt_handle)
                    return False

                flight_result = await session.execute(
                    select(Flight).where(Flight.id == job.flight_id)
                )
                flight = flight_result.scalar_one_or_none()

                # 3. Transition Job to VALIDATING in DB
                job.status = JobStatus.VALIDATING
                job.current_stage = "VALIDATING"
                job.progress = 5
                job.started_at = datetime.now(timezone.utc)
                await session.commit()
                await session.refresh(job)

            # 4. Update Redis state and publish event
            total_frames = (flight.total_frames or 0) if flight else 0
            if self.redis_tracker:
                await self.redis_tracker.set_job_progress(
                    job_id=job_id,
                    stage="VALIDATING",
                    progress=5,
                    frames_processed=0,
                    total_frames=total_frames,
                    gpu_stats={"utilization": 0.0},
                )

            logger.info("Job %s status successfully updated to VALIDATING by worker [%s]", job_id, self.worker_id)

            # 5. Pipeline execution hook (delegated to subsequent stage workers)
            await self.execute_pipeline_stages(job_id, flight)

            # 6. Acknowledge and delete message from queue
            if receipt_handle:
                await asyncio.to_thread(self.queue_client.delete_message, receipt_handle)

            return True

        except Exception as exc:
            logger.exception("Error during execution of job %s: %s", job_id, exc)
            return False

        finally:
            if self.redis_tracker:
                try:
                    await self.redis_tracker._client.delete(f"lock:job:{job_id}")
                except Exception:
                    pass
            self.current_job_id = None

    async def execute_pipeline_stages(self, job_id: UUID, flight: Optional[Flight]) -> None:
        """
        Pipeline stage execution dispatcher.
        In Phase 3+, this invokes the sequential worker modules:
        video_validator -> telemetry_parser -> quality_evaluator -> etc.
        """
        logger.info("Pipeline stage execution initialized for job %s", job_id)
