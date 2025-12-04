"""
Generation service API

Handles reranking, LLM generation, sentiment analysis, and toxicity filtering.
"""

import asyncio
import logging
import time
from typing import TYPE_CHECKING, cast

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response
from prometheus_client import generate_latest

from pipeline.component_registry import ComponentRegistry
from pipeline.components.document_store import DocumentStore
from pipeline.components.schemas import Document
from pipeline.config import get_settings
from pipeline.dependencies import get_registry
from pipeline.services.gateway.batch_scheduler import Batch, BatchScheduler
from pipeline.services.gateway.schemas import PendingRequest
from pipeline.telemetry import (
    error_counter as pipeline_error_counter,
    get_resource_snapshot,
    memory_gauge,
    request_counter as pipeline_request_counter,
)
from pipeline.utils.executors import ServiceExecutorFactory

from .schemas import (
    ErrorResponse,
    GenerationRequest,
    GenerationRequestItem,
    GenerationResponse,
    GenerationResponseItem,
)
from .service import GenerationService


if TYPE_CHECKING:
    from pipeline.components.llm import LLMGenerator
    from pipeline.components.reranker import Reranker
    from pipeline.components.sentiment import SentimentAnalyzer
    from pipeline.components.toxicity import ToxicityFilter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
settings = get_settings()
NODE_LABEL = str(settings.node_number)


# Create APIRouter
router = APIRouter()


class GenerationExecutor:
    """
    Executor for generation requests using adaptive batching.
    """

    def __init__(self, registry: ComponentRegistry) -> None:
        self.registry = registry
        self.scheduler: BatchScheduler[GenerationResponseItem] = BatchScheduler(
            batch_size=settings.generation_batch_size,
            max_batch_delay_ms=settings.generation_max_batch_delay_ms,
            process_batch_fn=self._process_batch,
            service_name="generation",
            enable_adaptive=getattr(settings, "enable_adaptive_batching", True),
        )

    async def start(self) -> None:
        await self.scheduler.start()

    async def stop(self) -> None:
        await self.scheduler.stop()

    async def process_request(self, item: GenerationRequestItem) -> GenerationResponseItem:
        # Convert Document objects to dicts for PendingRequest
        docs_dicts = [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "content": d.content,
                "category": d.category or "",
                "score": getattr(d, "score", 0.0),
            }
            for d in item.docs
        ]

        req = PendingRequest(
            request_id=item.request_id,
            query=item.query,
            docs=docs_dicts,
            compressed_docs=item.compressed_docs,
            timestamp=time.time(),
        )
        return await self.scheduler.enqueue(req)

    async def _process_batch(
        self, batch: Batch[GenerationResponseItem]
    ) -> list[GenerationResponseItem]:
        """
        Process a batch of generation requests.
        """
        loop = asyncio.get_running_loop()
        return await ServiceExecutorFactory.run_cpu_bound(
            loop, "generation", self._process_batch_sync, batch
        )

    def _process_batch_sync(
        self, batch: Batch[GenerationResponseItem]
    ) -> list[GenerationResponseItem]:
        """
        Synchronous processing of a batch of generation requests.
        """
        # Reconstruct GenerationRequest
        items = []
        for req in batch.requests:
            # Convert dicts back to Document
            docs = [
                Document(
                    doc_id=int(d["doc_id"]),
                    title=str(d["title"]),
                    content=str(d["content"]),
                    category=str(d["category"]) if d.get("category") else "",
                )
                for d in (req.docs or [])
            ]
            items.append(
                GenerationRequestItem(
                    request_id=req.request_id,
                    query=req.query,
                    docs=docs,
                    compressed_docs=req.compressed_docs,
                )
            )

        gen_req = GenerationRequest(batch_id=str(batch.batch_id), items=items)
        logger.info("Processing batch sync with %d items", len(items))

        # Instantiate Service
        # We need to cast components because registry.get returns object
        reranker = cast("Reranker | None", self.registry.get("reranker"))
        llm_generator = cast("LLMGenerator", self.registry.get("llm_generator"))
        sentiment_analyzer = cast(
            "SentimentAnalyzer | None", self.registry.get("sentiment_analyzer")
        )
        toxicity_filter = cast("ToxicityFilter | None", self.registry.get("toxicity_filter"))
        document_store = cast("DocumentStore | None", self.registry.get("document_store"))

        if not llm_generator:
            # Should not happen if registry is healthy, but good to check
            msg = "LLM Generator not available"
            raise RuntimeError(msg)

        service = GenerationService(
            reranker=reranker,
            llm_generator=llm_generator,
            sentiment_analyzer=sentiment_analyzer,
            toxicity_filter=toxicity_filter,
            document_store=document_store,
        )

        response = service.process_batch(gen_req)
        return response.items


class _ExecutorContainer:
    """Container for singleton executor instance to avoid global statement."""

    instance: GenerationExecutor | None = None


_executor_container = _ExecutorContainer()


async def start_generation_executor(registry: ComponentRegistry) -> None:
    """Start the generation executor."""
    if _executor_container.instance is None:
        _executor_container.instance = GenerationExecutor(registry)
        await _executor_container.instance.start()
        logger.info("GenerationExecutor started")


async def stop_generation_executor() -> None:
    """Stop the generation executor."""
    if _executor_container.instance:
        await _executor_container.instance.stop()
        _executor_container.instance = None
        logger.info("GenerationExecutor stopped")


async def get_executor(request: Request) -> GenerationExecutor:
    """Get the generation executor instance."""
    if _executor_container.instance is None:
        registry = get_registry(request)
        await start_generation_executor(registry)

    if _executor_container.instance is None:
        msg = "Failed to initialize GenerationExecutor"
        raise RuntimeError(msg)

    return _executor_container.instance


def _record_memory_usage() -> None:
    snapshot = get_resource_snapshot()
    memory_gauge.labels(
        run_id=settings.profiling_run_id, node=NODE_LABEL, service="generation", type="rss"
    ).set(snapshot.rss)
    memory_gauge.labels(
        run_id=settings.profiling_run_id, node=NODE_LABEL, service="generation", type="vms"
    ).set(snapshot.vms)
    memory_gauge.labels(
        run_id=settings.profiling_run_id,
        node=NODE_LABEL,
        service="generation",
        type="percent",
    ).set(snapshot.memory_percent)


def _record_pipeline_error(error_type: str) -> None:
    pipeline_error_counter.labels(
        run_id=settings.profiling_run_id,
        node=NODE_LABEL,
        service="generation",
        error_type=error_type,
    ).inc()
    pipeline_request_counter.labels(
        run_id=settings.profiling_run_id,
        node=NODE_LABEL,
        service="generation",
        status="error",
    ).inc()


@router.post("", responses={500: {"model": ErrorResponse}})
async def generate(
    request: Request,
    generation_request: GenerationRequest,
) -> GenerationResponse:
    """
    Generate responses for a batch of queries.
    """
    start_time = time.time()
    _record_memory_usage()

    executor = await get_executor(request)

    # Check readiness before processing
    llm_generator = executor.registry.get("llm_generator")
    if not llm_generator or not getattr(llm_generator, "is_loaded", False):
        _record_pipeline_error("service_unavailable")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service not ready (LLM generator not loaded)",
        )

    try:
        futures = [executor.process_request(item) for item in generation_request.items]
        results = await asyncio.gather(*futures)

        return GenerationResponse(
            batch_id=generation_request.batch_id,
            items=results,
            processing_time=time.time() - start_time,
        )

    except Exception as e:
        _record_pipeline_error("unknown")
        logger.exception("Error processing generation batch")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process generation batch: {e!s}",
        ) from e


@router.post("/clear_cache")
async def clear_cache(request: Request) -> dict[str, str]:
    """Clear generation document cache."""
    try:
        executor = await get_executor(request)
        if executor:
            document_store = executor.registry.get("document_store")
            if isinstance(document_store, DocumentStore):
                document_store.clear_cache()
                logger.info("Generation document cache cleared")
                return {"status": "cleared", "service": "generation"}
    except Exception:
        logger.exception("Failed to clear generation cache")
        return {"status": "error", "detail": "Exception occurred"}
    else:
        return {"status": "error", "detail": "Executor not initialized"}


@router.get("/metrics", response_class=Response)
@router.head("/metrics", response_class=Response)
async def metrics() -> Response:
    """
    Prometheus metrics endpoint.

    Returns:
        Metrics in Prometheus text format
    """
    return Response(
        content=generate_latest(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
