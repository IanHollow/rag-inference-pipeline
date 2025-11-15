"""
OpenTelemetry tracing utilities.

This module centralizes configuration so every service exports spans with
consistent resource metadata and instrumentation coverage.
"""

from __future__ import annotations

import logging
from threading import Lock
from typing import TYPE_CHECKING

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SpanExporter

if TYPE_CHECKING:
    from fastapi import FastAPI

    from ..config import PipelineSettings

logger = logging.getLogger(__name__)
_setup_lock = Lock()
_configured = False


def setup_tracing(settings: PipelineSettings, service_name: str) -> None:
    """
    Configure the global tracer provider for this process.
    """
    global _configured

    if _configured or not settings.enable_tracing:
        return

    with _setup_lock:
        if _configured:
            return

        resource = Resource.create(
            {
                SERVICE_NAME: service_name,
                "service.namespace": "cs5416-ml-pipeline",
                "service.instance.id": f"node-{settings.node_number}",
                "pipeline.node": settings.node_number,
                "pipeline.role": settings.role.value,
            }
        )

        provider = TracerProvider(resource=resource)
        exporters: list[SpanExporter] = []

        try:
            exporters.append(
                OTLPSpanExporter(
                    endpoint=settings.otel_exporter_endpoint,
                    insecure=settings.otel_exporter_insecure,
                    timeout=5,
                )
            )
        except Exception as exc:
            logger.warning("Failed to initialize OTLP exporter: %s", exc)

        if not exporters:
            exporters.append(ConsoleSpanExporter())

        for exporter in exporters:
            provider.add_span_processor(BatchSpanProcessor(exporter))

        trace.set_tracer_provider(provider)

        try:
            HTTPXClientInstrumentor().instrument()
        except Exception as exc:
            logger.warning("Failed to instrument httpx for tracing: %s", exc)

        _configured = True
        logger.info("OpenTelemetry tracing configured for %s", service_name)


def instrument_fastapi_app(app: FastAPI) -> None:
    """
    Apply FastAPI instrumentation with sane defaults.
    """
    try:
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls="/metrics,/health",
        )
    except Exception as exc:
        logger.warning("Unable to instrument FastAPI app for tracing: %s", exc)
