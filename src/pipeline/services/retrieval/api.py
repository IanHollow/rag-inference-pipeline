"""
Retrieval service API

Provides embedding generation, FAISS ANN search, and document retrieval.
"""

import asyncio
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
import hashlib
import logging
import threading
import time
from typing import TYPE_CHECKING, Any, TypeVar, cast

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response
import lz4.frame  # type: ignore
import msgspec
from opentelemetry import trace
from prometheus_client import REGISTRY, Counter, Histogram, generate_latest

from ...component_registry import ComponentRegistry
from ...components.schemas import Document
from ...config import get_settings
from ...dependencies import get_registry
from ...services.gateway.batch_scheduler import Batch, BatchScheduler
from ...services.gateway.schemas import PendingRequest
from ...telemetry import (
    SampledStageProfiler,
    batch_size_histogram as pipeline_batch_size_histogram,
    error_counter as pipeline_error_counter,
    get_resource_snapshot,
    latency_histogram as pipeline_latency_histogram,
    memory_gauge,
    request_counter as pipeline_request_counter,
)
from ...utils.cache import LRUCache
from .schemas import (
    ErrorResponse,
    RetrievalDocument,
    RetrievalRequest,
    RetrievalRequestItem,
    RetrievalResponse,
    RetrievalResponseItem,
)

if TYPE_CHECKING:
    from ...components.document_store import DocumentStore
    from ...components.embedding import EmbeddingGenerator
    from ...components.faiss_store import FAISSStore
    from ...components.reranker import Reranker

MetricType = Counter | Histogram
T = TypeVar("T", bound=MetricType)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global state
settings = get_settings()
NODE_LABEL = str(settings.node_number)
tracer = trace.get_tracer(__name__)


def get_metric(
    name: str,
    type_cls: type[T],
    documentation: str,
    labelnames: Sequence[str] = (),
    buckets: Sequence[float] | None = None,
) -> T:
    if name in REGISTRY._names_to_collectors:
        return cast("T", REGISTRY._names_to_collectors[name])
    kwargs = {}
    if buckets and type_cls is Histogram:
        kwargs["buckets"] = buckets
    return cast("T", type_cls(name, documentation, labelnames, **cast("Any", kwargs)))


# Prometheus metrics
retrieval_requests_total = get_metric(
    "retrieval_requests_total",
    Counter,
    "Total number of retrieval requests",
)

retrieval_request_duration = get_metric(
    "retrieval_request_duration_seconds",
    Histogram,
    "Retrieval request duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
)

retrieval_batch_size = get_metric(
    "retrieval_batch_size",
    Histogram,
    "Number of items in retrieval batch",
    buckets=[1, 2, 4, 8, 16, 32],
)

faiss_search_duration = get_metric(
    "faiss_search_duration_seconds",
    Histogram,
    "FAISS search duration in seconds",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
)

document_fetch_duration = get_metric(
    "document_fetch_duration_seconds",
    Histogram,
    "Document fetch duration in seconds",
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0],
)

embedding_duration = get_metric(
    "embedding_duration_seconds",
    Histogram,
    "Embedding generation duration in seconds",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0],
)


# Create APIRouter
router = APIRouter()


def _record_memory_usage() -> None:
    """Record current process memory in Prometheus gauges."""
    snapshot = get_resource_snapshot()
    memory_gauge.labels(
        run_id=settings.profiling_run_id, node=NODE_LABEL, service="retrieval", type="rss"
    ).set(snapshot.rss)
    memory_gauge.labels(
        run_id=settings.profiling_run_id, node=NODE_LABEL, service="retrieval", type="vms"
    ).set(snapshot.vms)
    memory_gauge.labels(
        run_id=settings.profiling_run_id,
        node=NODE_LABEL,
        service="retrieval",
        type="percent",
    ).set(snapshot.memory_percent)


def _record_pipeline_error(error_type: str) -> None:
    """Increment shared error counters for this service."""
    pipeline_error_counter.labels(
        run_id=settings.profiling_run_id,
        node=NODE_LABEL,
        service="retrieval",
        error_type=error_type,
    ).inc()
    pipeline_request_counter.labels(
        run_id=settings.profiling_run_id,
        node=NODE_LABEL,
        service="retrieval",
        status="error",
    ).inc()


@router.post(
    "",
    responses={
        500: {"model": ErrorResponse, "description": "Internal server error"},
        400: {"model": ErrorResponse, "description": "Bad request"},
    },
)
async def retrieve(
    request: Request,
    retrieval_request: RetrievalRequest,
) -> RetrievalResponse:
    """
    Process a batch of retrieval requests.
    """
    start_time = time.time()
    _record_memory_usage()

    # Validate request
    if not retrieval_request.items:
        _record_pipeline_error("validation")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch must contain at least one item",
        )

    retrieval_requests_total.inc()

    try:
        executor = await get_executor(request)

        # Check readiness
        faiss_store = executor.registry.get("faiss_store")
        embedding_generator = executor.registry.get("embedding_generator")

        if not faiss_store or not getattr(faiss_store, "is_loaded", False):
            _record_pipeline_error("service_unavailable")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready (FAISS store not loaded)",
            )

        # Only check embedding generator if we actually need it (missing embeddings in request)
        needs_embedding = any(item.embedding is None for item in retrieval_request.items)
        # If we need embeddings, the generator must be present and loaded
        if needs_embedding and (
            not embedding_generator or not getattr(embedding_generator, "is_loaded", False)
        ):
            _record_pipeline_error("service_unavailable")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Service not ready (Embedding generator not loaded)",
            )

        # Enqueue all items
        futures = [executor.process_request(item) for item in retrieval_request.items]
        results = await asyncio.gather(*futures)

        # Record metrics
        total_duration = time.time() - start_time
        retrieval_request_duration.observe(total_duration)
        pipeline_latency_histogram.labels(
            run_id=settings.profiling_run_id, node=NODE_LABEL, service="retrieval"
        ).observe(total_duration)
        pipeline_request_counter.labels(
            run_id=settings.profiling_run_id,
            node=NODE_LABEL,
            service="retrieval",
            status="success",
        ).inc()

        return RetrievalResponse(
            batch_id=retrieval_request.batch_id,
            items=results,
        )

    except ValueError as e:
        _record_pipeline_error("validation")
        logger.warning("Validation error in retrieval batch: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        ) from e

    except HTTPException as e:
        raise e

    except Exception as e:
        _record_pipeline_error("unknown")
        logger.exception("Error processing retrieval batch: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process retrieval batch: {e!s}",
        ) from e


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


class RetrievalExecutor:
    """
    Executor for retrieval requests using adaptive batching.
    """

    def __init__(self, registry: ComponentRegistry) -> None:
        self.registry = registry
        self.scheduler: BatchScheduler[RetrievalResponseItem] = BatchScheduler(
            batch_size=settings.retrieval_batch_size,
            max_batch_delay_ms=settings.retrieval_max_batch_delay_ms,
            process_batch_fn=self._process_batch,
            service_name="retrieval",
            enable_adaptive=getattr(settings, "enable_adaptive_batching", True),
        )
        # Initialize cache
        self.cache = LRUCache[str, tuple[list[int], list[float]]](
            capacity=settings.retrieval_cache_capacity,
            ttl=settings.cache_max_ttl,
            name="retrieval_faiss_cache",
        )
        self._lock = threading.Lock()

    async def start(self) -> None:
        await self.scheduler.start()

    async def stop(self) -> None:
        await self.scheduler.stop()

    def clear_cache(self) -> None:
        """Clear the retrieval cache safely."""
        with self._lock:
            self.cache.clear()

    async def process_request(self, item: RetrievalRequestItem) -> RetrievalResponseItem:
        req = PendingRequest(
            request_id=item.request_id,
            query=item.query,
            embedding=item.embedding,
            timestamp=time.time(),
        )
        return await self.scheduler.enqueue(req)

    async def _process_batch(
        self, batch: Batch[RetrievalResponseItem]
    ) -> list[RetrievalResponseItem]:
        """
        Process a batch of retrieval requests.

        FAISS search runs in a thread pool executor to:
        1. Allow FAISS to use multiple threads (faiss.omp_set_num_threads)
        2. Prevent blocking the async event loop during CPU-intensive search
        This improves throughput by allowing concurrent request handling.
        """
        import asyncio

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._process_batch_sync, batch)

    def _process_batch_sync(
        self, batch: Batch[RetrievalResponseItem]
    ) -> list[RetrievalResponseItem]:
        """
        Synchronous processing of a batch of retrieval requests.
        """
        # Get components
        faiss_store = cast("FAISSStore", self.registry.get("faiss_store"))
        document_store = cast("DocumentStore | None", self.registry.get("document_store"))
        embedding_generator = cast(
            "EmbeddingGenerator | None", self.registry.get("embedding_generator")
        )
        reranker = cast("Reranker | None", self.registry.get("reranker"))

        if not faiss_store:
            raise RuntimeError("FAISS store not available")

        profiler = SampledStageProfiler(
            enabled=settings.enable_profiling,
            sample_rate=settings.profiling_sample_rate,
            logger=logger,
        )

        batch_size = len(batch)
        retrieval_batch_size.observe(batch_size)
        pipeline_batch_size_histogram.labels(
            run_id=settings.profiling_run_id, node=NODE_LABEL, service="retrieval"
        ).observe(batch_size)

        with tracer.start_as_current_span(
            "retrieval.batch_exec",
            attributes={
                "pipeline.batch_id": batch.batch_id,
                "pipeline.service": "retrieval",
                "pipeline.node": settings.node_number,
            },
        ):
            # Step 1: Get embeddings
            embeddings = None

            # Check if embeddings are provided
            if batch.requests[0].embedding is not None:
                try:
                    import numpy as np

                    embedding_list = []
                    for req in batch.requests:
                        if req.embedding is None:
                            raise ValueError("Missing embedding in batch")
                        embedding_list.append(req.embedding)
                    embeddings = np.array(embedding_list).astype("float32")
                except Exception as e:
                    logger.error("Error processing embeddings: %s", e)
                    raise

            else:
                if not embedding_generator:
                    raise RuntimeError("Embedding generator not available")

                queries = [req.query for req in batch.requests]
                emb_start = time.time()
                with profiler.track("retrieval.embedding"):
                    embeddings = embedding_generator.encode(queries)
                embedding_duration.observe(time.time() - emb_start)
                logger.info("Embedding generation completed in %.3fs", time.time() - emb_start)

            # Step 2: FAISS Search
            # FAISS is called directly (not via executor) to use full OpenMP parallelism
            # without thread conflicts. The batch scheduler serializes batches.
            faiss_start = time.time()

            doc_ids_batch: list[list[int]] = [[] for _ in range(batch_size)]
            distances_batch: list[list[float]] = [[] for _ in range(batch_size)]

            if getattr(settings, "disable_cache_for_profiling", True):
                with profiler.track("retrieval.faiss_search"):
                    distances, indices = faiss_store.search(embeddings, settings.retrieval_k)
                doc_ids_batch = [row.tolist() for row in indices]
                distances_batch = [row.tolist() for row in distances]
            else:
                missing_indices = []
                missing_embeddings = []

                for i in range(batch_size):
                    embedding = embeddings[i]
                    key = hashlib.sha256(embedding.tobytes()).hexdigest()
                    with self._lock:
                        cached = self.cache.get(key)
                    if cached:
                        doc_ids_batch[i], distances_batch[i] = cached
                    else:
                        missing_indices.append(i)
                        missing_embeddings.append(embedding)

                if missing_embeddings:
                    import numpy as np

                    missing_embeddings_np = np.array(missing_embeddings)
                    with profiler.track("retrieval.faiss_search"):
                        distances, indices = faiss_store.search(
                            missing_embeddings_np, settings.retrieval_k
                        )

                    for i, idx in enumerate(missing_indices):
                        doc_ids = indices[i].tolist()
                        dists = distances[i].tolist()
                        doc_ids_batch[idx] = doc_ids
                        distances_batch[idx] = dists

                        # Update cache
                        key = hashlib.sha256(missing_embeddings[i].tobytes()).hexdigest()
                        with self._lock:
                            self.cache.put(key, (doc_ids, dists))
                else:
                    logger.debug("Cache hit for all %d retrieval items", batch_size)

            faiss_search_duration.observe(time.time() - faiss_start)
            logger.info("FAISS search completed in %.3fs", time.time() - faiss_start)

            # Step 3: Fetch Documents
            documents_batch = []

            if document_store:
                # Check handoff mode
                mode = getattr(settings, "documents_payload_mode", "full")

                if mode == "id_only":
                    from ...components.document_store import Document as StoreDocument

                    for i, doc_ids in enumerate(doc_ids_batch):
                        dists = distances_batch[i]
                        docs = []
                        for _j, doc_id in enumerate(doc_ids):
                            doc = StoreDocument(doc_id=doc_id, title="", content="")
                            docs.append(doc)
                        documents_batch.append(docs)

                elif mode == "compressed":
                    # Fetch full docs then compress
                    fetch_start = time.time()
                    with profiler.track("retrieval.document_fetch"):
                        # Parallelize fetch if batch size > 1
                        max_workers = min(settings.cpu_worker_threads, batch_size)
                        if max_workers > 1:
                            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                futures = [
                                    executor.submit(
                                        document_store.fetch_documents_batch,
                                        [doc_ids],
                                        truncate_length=settings.truncate_length,
                                    )
                                    for doc_ids in doc_ids_batch
                                ]
                                results = [f.result()[0] for f in futures]
                                documents_batch = results
                        else:
                            documents_batch = document_store.fetch_documents_batch(
                                doc_ids_batch, truncate_length=settings.truncate_length
                            )
                    document_fetch_duration.observe(time.time() - fetch_start)

                else:
                    # Full fetch
                    fetch_start = time.time()
                    with profiler.track("retrieval.document_fetch"):
                        # Parallelize fetch if batch size > 1
                        max_workers = min(settings.cpu_worker_threads, batch_size)
                        if max_workers > 1:
                            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                                futures = [
                                    executor.submit(
                                        document_store.fetch_documents_batch,
                                        [doc_ids],
                                        truncate_length=settings.truncate_length,
                                    )
                                    for doc_ids in doc_ids_batch
                                ]
                                results = [f.result()[0] for f in futures]
                                documents_batch = results
                        else:
                            documents_batch = document_store.fetch_documents_batch(
                                doc_ids_batch, truncate_length=settings.truncate_length
                            )
                    document_fetch_duration.observe(time.time() - fetch_start)
                    logger.info("Document fetch completed in %.3fs", time.time() - fetch_start)
            else:
                # Dummy docs
                from ...components.document_store import Document as StoreDocument

                for doc_ids in doc_ids_batch:
                    docs = [
                        StoreDocument(doc_id=doc_id, title="", content="") for doc_id in doc_ids
                    ]
                    documents_batch.append(docs)

            # Step 4: Build Response Items & Rerank
            response_items: list[RetrievalResponseItem] = []

            # Import Document for type hint if needed, though it's likely available via StoreDocument alias or similar
            from ...components.document_store import Document as StoreDocument
            from ...services.gateway.schemas import PendingRequest, PendingRequestStruct

            def process_single_item(
                args: tuple[
                    int, PendingRequest | PendingRequestStruct, list[StoreDocument], list[float]
                ],
            ) -> RetrievalResponseItem:
                _idx, req, docs, scores = args
                retrieval_docs = [
                    RetrievalDocument(
                        doc_id=doc.doc_id,
                        title=doc.title,
                        content=doc.content,
                        category=doc.category or "",
                        score=float(score),
                    )
                    for doc, score in zip(docs, scores, strict=False)
                ]

                # Step 5: Rerank
                if reranker:
                    # Convert to Document
                    reranker_input = [
                        Document(
                            doc_id=d.doc_id,
                            title=d.title,
                            content=d.content,
                            category=d.category,
                        )
                        for d in retrieval_docs
                    ]
                    reranked = reranker.rerank(req.query, reranker_input)

                    retrieval_docs = [
                        RetrievalDocument(
                            doc_id=d.doc_id,
                            title=d.title,
                            content=d.content,
                            category=d.category,
                            score=d.score,
                        )
                        for d in reranked
                    ]

                # Handle compression
                compressed_payload = None
                mode = getattr(settings, "documents_payload_mode", "full")
                if mode == "compressed":
                    # Serialize and compress
                    docs_dicts = [d.model_dump() for d in retrieval_docs]
                    serialized = msgspec.json.encode(docs_dicts)
                    compressed_payload = lz4.frame.compress(serialized)
                    # Clear docs to save bandwidth
                    retrieval_docs = []

                return RetrievalResponseItem(
                    request_id=req.request_id,
                    docs=retrieval_docs,
                    compressed_docs=compressed_payload,
                )

            tasks = []
            for idx, req in enumerate(batch.requests):
                docs = documents_batch[idx]
                scores = distances_batch[idx]
                tasks.append((idx, req, docs, scores))

            if reranker and len(tasks) > 1:
                max_workers = min(settings.cpu_worker_threads, len(tasks))
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    response_items = list(executor.map(process_single_item, tasks))
            else:
                response_items = [process_single_item(t) for t in tasks]

            return response_items


# Global executor instance
_executor: RetrievalExecutor | None = None


async def start_retrieval_executor(registry: ComponentRegistry) -> None:
    """Start the retrieval executor."""
    global _executor
    if _executor is None:
        _executor = RetrievalExecutor(registry)
        await _executor.start()
        logger.info("RetrievalExecutor started")


async def stop_retrieval_executor() -> None:
    """Stop the retrieval executor."""
    global _executor
    if _executor:
        await _executor.stop()
        _executor = None
        logger.info("RetrievalExecutor stopped")


async def get_executor(request: Request) -> RetrievalExecutor:
    """Get the retrieval executor instance."""
    global _executor
    if _executor is None:
        # Fallback for tests or if not started via lifecycle
        registry = get_registry(request)
        await start_retrieval_executor(registry)

    if _executor is None:
        raise RuntimeError("Failed to initialize RetrievalExecutor")

    return _executor


@router.post("/clear_cache")
async def clear_cache(request: Request) -> dict[str, str]:
    """Clear all caches."""
    executor = await get_executor(request)
    with executor._lock:
        executor.cache.clear()

    # Clear component caches
    registry = get_registry(request)
    if (embedding_generator := registry.get("embedding_generator")) and hasattr(
        embedding_generator, "cache"
    ):
        embedding_generator.cache.clear()

    if (document_store := registry.get("document_store")) and hasattr(document_store, "cache"):
        document_store.cache.clear()

    return {"status": "cleared"}
