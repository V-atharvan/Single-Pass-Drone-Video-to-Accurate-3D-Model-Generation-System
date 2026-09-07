"""
OpenTelemetry instrumentation baseline for Single-Pass 3D Reconstruction Platform.

Configures:
  - OTLP gRPC trace exporter (or JSON console fallback for local dev)
  - Structured JSON logging with trace_id / span_id injection
  - Standard trace tags: project_id, flight_id, job_id, worker_stage

Usage:
    from packages.shared.python.telemetry import setup_telemetry, get_tracer, get_logger

    setup_telemetry(service_name="api", otlp_endpoint="http://otel-collector:4317")
    tracer = get_tracer("api")
    logger = get_logger("api")
"""
from __future__ import annotations

import logging
import os
import sys
import json
from datetime import datetime, timezone
from typing import Any, Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_tracer_provider: Optional[TracerProvider] = None
_initialized: bool = False


# ---------------------------------------------------------------------------
# JSON Structured Logging
# ---------------------------------------------------------------------------

class _OtelLogFormatter(logging.Formatter):
    """
    Emit structured JSON log lines, injecting OpenTelemetry trace context
    (trace_id, span_id) if a span is active.
    """

    def format(self, record: logging.LogRecord) -> str:
        current_span = trace.get_current_span()
        span_context = current_span.get_span_context() if current_span else None

        log_record: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": (
                format(span_context.trace_id, "032x")
                if span_context and span_context.is_valid
                else None
            ),
            "span_id": (
                format(span_context.span_id, "016x")
                if span_context and span_context.is_valid
                else None
            ),
        }

        # Append extra fields set on the log record (e.g. project_id, job_id)
        for key in ("project_id", "flight_id", "job_id", "worker_stage"):
            val = getattr(record, key, None)
            if val is not None:
                log_record[key] = val

        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_record, default=str)


def _configure_json_logging(level: str = "INFO") -> None:
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_OtelLogFormatter())
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


# ---------------------------------------------------------------------------
# Tracer Setup
# ---------------------------------------------------------------------------

def setup_telemetry(
    service_name: str,
    otlp_endpoint: Optional[str] = None,
    log_level: str = "INFO",
) -> None:
    """
    Initialise the global OpenTelemetry TracerProvider and structured logging.

    Args:
        service_name:   Logical service name (e.g. "api", "worker-pose", "worker-depth").
        otlp_endpoint:  gRPC OTLP collector endpoint, e.g. "http://otel-collector:4317".
                        Falls back to console exporter when not supplied.
        log_level:      Root log level string (default "INFO").
    """
    global _tracer_provider, _initialized

    if _initialized:
        return

    resource = Resource.create({SERVICE_NAME: service_name})
    provider = TracerProvider(resource=resource)

    # Select exporter
    if otlp_endpoint or os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            endpoint = otlp_endpoint or os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"]
            exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        except ImportError:
            # Fall back to console if grpc extra not installed
            exporter = ConsoleSpanExporter()  # type: ignore[assignment]
    else:
        exporter = ConsoleSpanExporter()  # type: ignore[assignment]

    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    _tracer_provider = provider

    _configure_json_logging(log_level)
    _initialized = True


def get_tracer(name: str) -> trace.Tracer:
    """Return a named OpenTelemetry tracer. Ensure `setup_telemetry()` has been called."""
    return trace.get_tracer(name)


def get_logger(name: str) -> logging.Logger:
    """Return a standard Python logger configured with the JSON formatter."""
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Standard context helpers
# ---------------------------------------------------------------------------

def set_span_attributes(
    span: trace.Span,
    project_id: Optional[str] = None,
    flight_id: Optional[str] = None,
    job_id: Optional[str] = None,
    worker_stage: Optional[str] = None,
) -> None:
    """
    Attach standardised trace tags to an active span.
    All attributes are optional; only non-None values are set.
    """
    if project_id is not None:
        span.set_attribute("project_id", project_id)
    if flight_id is not None:
        span.set_attribute("flight_id", flight_id)
    if job_id is not None:
        span.set_attribute("job_id", job_id)
    if worker_stage is not None:
        span.set_attribute("worker_stage", worker_stage)
