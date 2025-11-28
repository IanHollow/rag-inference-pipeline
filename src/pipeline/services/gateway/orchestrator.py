"""
Orchestrator for coordinating the entire ML pipeline across nodes.
"""

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
import json
import logging
import time

import msgspec
from opentelemetry import trace

from ...components.embedding import EmbeddingGenerator
from ...components.reranker import Reranker
from ...components.schemas import Document
from ...components.sentiment import SentimentAnalyzer
from ...components.toxicity import ToxicityFilter
from ...config import get_settings
from ...enums import ServiceEndpoint
from ...telemetry import (
    SampledStageProfiler,
    batch_size_histogram,
    latency_histogram,
    queue_depth_gauge,
    rpc_duration_histogram,
    stage_duration_gauge,
)
from ...utils.cache import LRUCache
from .batch_scheduler import Batch, BatchScheduler
from .rpc_client import RPCClient, RPCError
from .schemas import (
    GenerationRequest,
    GenerationRequestStruct,
    GenerationResponse,
    PendingRequest,
    PendingRequestStruct,
    QueryResponse,
    RetrievalRequest,
    RetrievalRequestStruct,
    RetrievalResponse,
)

logger = logging.getLogger(__name__)
settings = get_settings()
tracer = trace.get_tracer(__name__)


class Orchestrator:
    """
    Orchestrates the ML pipeline by coordinating batching and RPC calls.
    """

    def __init__(
        self,
        retrieval_url: str | None = None,
        generation_url: str | None = None,
        batch_size: int | None = None,
        batch_timeout: float | None = None,
    ) -> None:
        """Initialize the orchestrator."""
        self.embedding_generator: EmbeddingGenerator | None = None
        self.reranker: Reranker | None = None
        self.sentiment_analyzer: SentimentAnalyzer | None = None
        self.toxicity_filter: ToxicityFilter | None = None

        # Initialize RPC clients
        self.retrieval_client = RPCClient(
            base_url=retrieval_url or settings.retrieval_url,
            timeout_seconds=settings.request_timeout_seconds,
            max_retries=3,
            compression=settings.pipeline_rpc_compression,
            compression_level=settings.pipeline_rpc_compression_level,
        )

        self.generation_client = RPCClient(
            base_url=generation_url or settings.generation_url,
            timeout_seconds=settings.request_timeout_seconds,
            max_retries=3,
            compression=settings.pipeline_rpc_compression,
            compression_level=settings.pipeline_rpc_compression_level,
        )

        # Initialize batch scheduler
        # Use provided values or fall back to settings
        final_batch_size = batch_size if batch_size is not None else settings.gateway_batch_size
        # Convert seconds to ms for batch scheduler if provided, else use settings (already in ms)
        final_batch_timeout_ms = (
            int(batch_timeout * 1000)
            if batch_timeout is not None
            else settings.gateway_batch_timeout_ms
        )

        self.batch_scheduler: BatchScheduler[QueryResponse] = BatchScheduler(
            batch_size=final_batch_size,
            max_batch_delay_ms=final_batch_timeout_ms,
            process_batch_fn=self._process_batch,
            service_name="gateway",
            enable_adaptive=getattr(settings, "enable_adaptive_batching", True),
        )

        # Note: The BatchScheduler's AdaptiveBatchPolicy already uses the configured
        # batch_size as min_batch_size, so we don't need to override here unless
        # settings.gateway_min_batch_size is larger
        if self.batch_scheduler.policy and settings.gateway_min_batch_size > final_batch_size:
            self.batch_scheduler.policy.min_batch_size = settings.gateway_min_batch_size

        # Initialize cache
        self.query_cache = LRUCache[str, QueryResponse](
            capacity=settings.gateway_cache_capacity,
            ttl=settings.cache_max_ttl,
            name="gateway_response_cache",
        )

        # Initialize pipeline queues
        self.retrieval_queue: asyncio.Queue[PipelineChunk | None] = asyncio.Queue()
        self.generation_queue: asyncio.Queue[PipelineChunk | None] = asyncio.Queue()
        self.postproc_queue: asyncio.Queue[PipelineChunk | None] = asyncio.Queue()
        self.workers: list[asyncio.Task[None]] = []

        logger.info(
            "Orchestrator initialized with batch_size=%d, timeout=%dms, adaptive=%s",
            final_batch_size,
            final_batch_timeout_ms,
            getattr(settings, "enable_adaptive_batching", True),
        )

    async def start(self) -> None:
        """Start the orchestrator."""
        await self.batch_scheduler.start()

        # Start pipeline workers
        self.workers = [
            asyncio.create_task(self._retrieval_worker(), name="retrieval_worker"),
            asyncio.create_task(self._generation_worker(), name="generation_worker"),
            asyncio.create_task(self._postproc_worker(), name="postproc_worker"),
        ]

        logger.info("Orchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator and cleanup resources."""
        await self.batch_scheduler.stop()

        # Stop pipeline workers
        # Only enqueue one sentinel since we have one retrieval worker
        await self.retrieval_queue.put(None)

        if self.workers:
            await asyncio.gather(*self.workers, return_exceptions=True)

        await self.retrieval_client.close()
        await self.generation_client.close()
        logger.info("Orchestrator stopped")

    async def process_query(self, request_id: str, query: str) -> QueryResponse:
        """
        Process a single query through the entire pipeline.

        Args:
            request_id: Unique request identifier
            query: User query string

        Returns:
            QueryResponse with final results

        Raises:
            RPCError: If pipeline processing fails
        """
        start_time = time.time()
        logger.info("Processing query: request_id=%s", request_id)

        # Normalize query for caching
        normalized_query = " ".join(query.strip().lower().split())

        if getattr(settings, "fuzzy_cache_matching", False):
            # Token sort for fuzzy matching
            tokens = normalized_query.split()
            tokens.sort()
            normalized_query = " ".join(tokens)

        # Check cache
        # Respect DISABLE_CACHE_FOR_PROFILING
        cache_enabled = (
            settings.gateway_response_cache_enabled and not settings.disable_cache_for_profiling
        )

        if cache_enabled and (cached := self.query_cache.get(normalized_query)):
            logger.info("Cache hit for query: %s", query)
            # Update request_id
            response = cached.model_copy(update={"request_id": request_id})
            return response

        # Create pending request
        # Use Msgspec Struct for internal processing
        pending_request = PendingRequestStruct(
            request_id=request_id,
            query=query,
            timestamp=start_time,
            embedding=None,
            docs=None,
        )

        try:
            # Enqueue and wait for batch processing
            with tracer.start_as_current_span(
                "gateway.process_query",
                attributes={
                    "pipeline.request_id": request_id,
                    "pipeline.service": "gateway",
                    "pipeline.node": settings.node_number,
                },
            ):
                result = await self.batch_scheduler.enqueue(pending_request)

            # Cache result
            if cache_enabled:
                self.query_cache.put(normalized_query, result)

            elapsed = time.time() - start_time
            logger.info(
                "Query completed: request_id=%s, time=%.3fs",
                request_id,
                elapsed,
            )

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.exception(
                "Query failed: request_id=%s, time=%.3fs",
                request_id,
                elapsed,
            )
            raise RPCError(f"Pipeline processing failed: {e}") from e

    async def _process_batch(self, batch: Batch[QueryResponse]) -> list[QueryResponse]:
        """
        Process a batch of requests through the pipeline using pipelined execution.
        """
        batch_start = time.time()
        logger.info(
            "Processing batch %d with %d requests",
            batch.batch_id,
            len(batch),
        )

        profiler = SampledStageProfiler(
            enabled=settings.enable_profiling,
            sample_rate=settings.profiling_sample_rate,
            logger=logger,
        )
        node_label = str(settings.node_number)

        # Record batch size metric
        batch_size_histogram.labels(
            run_id=settings.profiling_run_id, node=node_label, service="gateway"
        ).observe(len(batch))

        # Create futures for results
        loop = asyncio.get_running_loop()
        futures = [loop.create_future() for _ in batch.requests]

        # Split into chunks for pipelining
        chunk_size = 4  # Smaller chunks for better pipelining
        chunks_reqs = [
            batch.requests[i : i + chunk_size] for i in range(0, len(batch.requests), chunk_size)
        ]
        chunks_futures = [
            futures[i : i + chunk_size] for i in range(0, len(batch.requests), chunk_size)
        ]

        # Submit chunks to pipeline
        for i, (chunk_reqs, chunk_futs) in enumerate(
            zip(chunks_reqs, chunks_futures, strict=False)
        ):
            chunk = PipelineChunk(
                batch_id=batch.batch_id,
                chunk_index=i,
                requests=chunk_reqs,
                futures=chunk_futs,
                profiler=profiler,
            )
            await self.retrieval_queue.put(chunk)

        # Wait for all results
        results = await asyncio.gather(*futures)

        batch_elapsed = time.time() - batch_start

        if summary := profiler.summary():
            logger.info(
                json.dumps(
                    {
                        "event": "gateway_profile",
                        "batch_id": batch.batch_id,
                        "summary": summary,
                    }
                )
            )

        # Structured log for profiling analysis
        logger.info(
            json.dumps(
                {
                    "event": "batch_completed",
                    "batch_id": batch.batch_id,
                    "size": len(batch),
                    "latency_ms": round(batch_elapsed * 1000, 2),
                    "timestamp": batch_start,
                }
            )
        )

        # Record latency metric
        latency_histogram.labels(
            run_id=settings.profiling_run_id, node=node_label, service="gateway"
        ).observe(batch_elapsed)

        return results

    async def _call_retrieval_service(
        self,
        batch_id: int,
        requests: Sequence[RetrievalRequest | RetrievalRequestStruct],
    ) -> list[RetrievalResponse]:
        """
        Call retrieval service with batch of requests.

        Args:
            batch_id: ID of the batch for logging
            requests: List of retrieval requests

        Returns:
            List of retrieval responses in same order

        Raises:
            RPCError: If retrieval service fails
        """
        logger.debug(
            "Calling retrieval service for batch %d (%d requests)",
            batch_id,
            len(requests),
        )

        try:
            # Use msgspec structs if available, otherwise fallback
            items_data = []
            for req in requests:
                if isinstance(req, RetrievalRequestStruct):
                    items_data.append(msgspec.to_builtins(req))
                else:
                    items_data.append(req.model_dump())

            payload = {
                "batch_id": str(batch_id),
                "items": items_data,
            }

            rpc_start = time.time()
            with tracer.start_as_current_span(
                "gateway.call_retrieval",
                attributes={
                    "pipeline.batch_id": batch_id,
                    "pipeline.service": "gateway",
                    "pipeline.node": settings.node_number,
                },
            ):
                response_data = await self.retrieval_client.post(
                    ServiceEndpoint.RETRIEVE.value,
                    payload,
                )
                rpc_duration = time.time() - rpc_start

            # Record RPC duration
            rpc_duration_histogram.labels(
                run_id=settings.profiling_run_id,
                source_node=str(settings.node_number),
                target_service="retrieval",
            ).observe(rpc_duration)

            # Parse responses
            # response_data is a dict (from orjson.loads)
            # We should probably use msgspec to decode if we can, but here we just get a dict.
            responses = [RetrievalResponse(**resp) for resp in response_data["items"]]

            logger.debug(
                "Retrieval service returned %d responses for batch %d (%.3fs)",
                len(responses),
                batch_id,
                rpc_duration,
            )

            return responses

        except Exception as e:
            logger.exception(
                "Retrieval service failed for batch %d",
                batch_id,
            )
            raise RPCError(f"Retrieval service error: {e}") from e

    async def _call_generation_service(
        self,
        batch_id: int,
        requests: Sequence[GenerationRequest | GenerationRequestStruct],
    ) -> list[GenerationResponse]:
        """
        Call generation service with batch of requests.

        Args:
            batch_id: ID of the batch for logging
            requests: List of generation requests

        Returns:
            List of generation responses in same order

        Raises:
            RPCError: If generation service fails
        """
        logger.debug(
            "Calling generation service for batch %d (%d requests)",
            batch_id,
            len(requests),
        )

        try:
            items_data = []
            for req in requests:
                if isinstance(req, GenerationRequestStruct):
                    items_data.append(msgspec.to_builtins(req))
                else:
                    items_data.append(req.model_dump())

            payload = {
                "batch_id": str(batch_id),
                "items": items_data,
            }

            rpc_start = time.time()
            with tracer.start_as_current_span(
                "gateway.call_generation",
                attributes={
                    "pipeline.batch_id": batch_id,
                    "pipeline.service": "gateway",
                    "pipeline.node": settings.node_number,
                },
            ):
                response_data = await self.generation_client.post(
                    ServiceEndpoint.GENERATE.value,
                    payload,
                )
                rpc_duration = time.time() - rpc_start

            # Record RPC duration
            rpc_duration_histogram.labels(
                run_id=settings.profiling_run_id,
                source_node=str(settings.node_number),
                target_service="generation",
            ).observe(rpc_duration)

            # Parse responses
            responses = [GenerationResponse(**resp) for resp in response_data["items"]]

            logger.debug(
                "Generation service returned %d responses for batch %d (%.3fs)",
                len(responses),
                batch_id,
                rpc_duration,
            )

            return responses

        except Exception as e:
            logger.exception(
                "Generation service failed for batch %d",
                batch_id,
            )
            raise RPCError(f"Generation service error: {e}") from e

    async def _retrieval_worker(self) -> None:
        """Worker for retrieval stage."""
        while True:
            chunk = await self.retrieval_queue.get()
            queue_depth_gauge.labels(
                run_id=settings.profiling_run_id,
                node=str(settings.node_number),
                service="gateway_retrieval",
            ).set(self.retrieval_queue.qsize())

            if chunk is None:
                await self.generation_queue.put(None)
                self.retrieval_queue.task_done()
                break

            try:
                # Generate embeddings if generator is available
                embeddings = None
                if self.embedding_generator and getattr(
                    self.embedding_generator, "is_loaded", False
                ):
                    queries = [req.query for req in chunk.requests]
                    with chunk.profiler.track("gateway.embedding"):
                        embeddings = self.embedding_generator.encode(queries)
                        if hasattr(embeddings, "tolist"):
                            embeddings = embeddings.tolist()

                retrieval_requests = [
                    RetrievalRequest(
                        request_id=req.request_id,
                        query=req.query,
                        embedding=embeddings[i] if embeddings is not None else None,
                    )
                    for i, req in enumerate(chunk.requests)
                ]

                retrieval_start = time.time()
                with chunk.profiler.track("gateway.retrieval_rpc"):
                    chunk.retrieval_responses = await self._call_retrieval_service(
                        chunk.batch_id,
                        retrieval_requests,
                    )
                stage_duration_gauge.labels(
                    run_id=settings.profiling_run_id,
                    node=str(settings.node_number),
                    stage="retrieval_rpc",
                ).set(time.time() - retrieval_start)

                await self.generation_queue.put(chunk)

            except Exception as e:
                logger.exception("Error in retrieval worker")
                # Fail all futures in chunk
                for f in chunk.futures:
                    if not f.done():
                        f.set_exception(e)
            finally:
                self.retrieval_queue.task_done()

    async def _generation_worker(self) -> None:
        """Worker for generation stage."""
        while True:
            chunk = await self.generation_queue.get()
            queue_depth_gauge.labels(
                run_id=settings.profiling_run_id,
                node=str(settings.node_number),
                service="gateway_generation",
            ).set(self.generation_queue.qsize())

            if chunk is None:
                await self.postproc_queue.put(None)
                self.generation_queue.task_done()
                break

            try:
                if not chunk.retrieval_responses:
                    raise ValueError("Missing retrieval responses")

                # Rerank if local reranker is available
                if self.reranker and getattr(self.reranker, "is_loaded", False):
                    rerank_start = time.time()
                    with chunk.profiler.track("gateway.rerank"):
                        for i, resp in enumerate(chunk.retrieval_responses):
                            # Convert dicts to Documents
                            docs = [
                                Document(
                                    doc_id=int(d["doc_id"]),
                                    title=str(d["title"]),
                                    content=str(d["content"]),
                                    category=str(d.get("category") or ""),
                                )
                                for d in resp.docs
                            ]
                            # Rerank
                            reranked = self.reranker.rerank(chunk.requests[i].query, docs)
                            resp.docs = [d.model_dump() for d in reranked]
                    stage_duration_gauge.labels(
                        run_id=settings.profiling_run_id,
                        node=str(settings.node_number),
                        stage="rerank",
                    ).set(time.time() - rerank_start)

                generation_requests = [
                    GenerationRequest(
                        request_id=req.request_id,
                        query=req.query,
                        docs=resp.docs,
                        compressed_docs=resp.compressed_docs,
                    )
                    for req, resp in zip(chunk.requests, chunk.retrieval_responses, strict=False)
                ]

                generation_start = time.time()
                with chunk.profiler.track("gateway.generation_rpc"):
                    chunk.generation_responses = await self._call_generation_service(
                        chunk.batch_id,
                        generation_requests,
                    )
                stage_duration_gauge.labels(
                    run_id=settings.profiling_run_id,
                    node=str(settings.node_number),
                    stage="generation_rpc",
                ).set(time.time() - generation_start)

                await self.postproc_queue.put(chunk)

            except Exception as e:
                logger.exception("Error in generation worker")
                for f in chunk.futures:
                    if not f.done():
                        f.set_exception(e)
            finally:
                self.generation_queue.task_done()

    async def _postproc_worker(self) -> None:
        """Worker for post-processing stage."""
        while True:
            chunk = await self.postproc_queue.get()
            queue_depth_gauge.labels(
                run_id=settings.profiling_run_id,
                node=str(settings.node_number),
                service="gateway_postproc",
            ).set(self.postproc_queue.qsize())

            if chunk is None:
                self.postproc_queue.task_done()
                break

            try:
                if not chunk.generation_responses:
                    raise ValueError("Missing generation responses")

                # Post-processing (Sentiment/Toxicity) if local
                if (
                    self.sentiment_analyzer and getattr(self.sentiment_analyzer, "is_loaded", False)
                ) or (self.toxicity_filter and getattr(self.toxicity_filter, "is_loaded", False)):
                    postproc_start = time.time()
                    with chunk.profiler.track("gateway.postprocessing"):
                        for gen_resp in chunk.generation_responses:
                            # Sentiment
                            if (
                                gen_resp.sentiment is None
                                and self.sentiment_analyzer
                                and getattr(self.sentiment_analyzer, "is_loaded", False)
                            ):
                                gen_resp.sentiment = self.sentiment_analyzer.analyze(
                                    gen_resp.generated_response
                                )

                            # Toxicity
                            if (
                                gen_resp.is_toxic is None
                                and self.toxicity_filter
                                and getattr(self.toxicity_filter, "is_loaded", False)
                            ):
                                is_toxic, _ = self.toxicity_filter.check(
                                    gen_resp.generated_response
                                )
                                gen_resp.is_toxic = "true" if is_toxic else "false"
                                if is_toxic:
                                    gen_resp.generated_response = (
                                        "[Content Filtered due to toxicity]"
                                    )
                    stage_duration_gauge.labels(
                        run_id=settings.profiling_run_id,
                        node=str(settings.node_number),
                        stage="postprocessing",
                    ).set(time.time() - postproc_start)

                # Build final responses and set futures
                for i, gen_resp in enumerate(chunk.generation_responses):
                    response = QueryResponse(
                        request_id=gen_resp.request_id,
                        generated_response=gen_resp.generated_response,
                        sentiment=gen_resp.sentiment or "neutral",
                        is_toxic=gen_resp.is_toxic or "false",
                    )
                    if not chunk.futures[i].done():
                        chunk.futures[i].set_result(response)

            except Exception as e:
                logger.exception("Error in postproc worker")
                for f in chunk.futures:
                    if not f.done():
                        f.set_exception(e)
            finally:
                self.postproc_queue.task_done()

    def set_components(
        self,
        embedding_generator: EmbeddingGenerator | None = None,
        reranker: Reranker | None = None,
        sentiment_analyzer: SentimentAnalyzer | None = None,
        toxicity_filter: ToxicityFilter | None = None,
    ) -> None:
        """Inject local components."""
        if embedding_generator:
            self.embedding_generator = embedding_generator
        if reranker:
            self.reranker = reranker
        if sentiment_analyzer:
            self.sentiment_analyzer = sentiment_analyzer
        if toxicity_filter:
            self.toxicity_filter = toxicity_filter

    def set_embedding_generator(self, embedding_generator: EmbeddingGenerator) -> None:
        """Set the embedding generator component."""
        self.embedding_generator = embedding_generator


@dataclass
class PipelineChunk:
    """Represents a chunk of requests moving through the pipeline."""

    batch_id: int
    chunk_index: int
    requests: list[PendingRequest | PendingRequestStruct]
    futures: list[asyncio.Future[QueryResponse]]
    profiler: SampledStageProfiler
    retrieval_responses: list[RetrievalResponse] | None = None
    generation_responses: list[GenerationResponse] | None = None
