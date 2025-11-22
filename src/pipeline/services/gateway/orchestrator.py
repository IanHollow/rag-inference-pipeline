"""
Orchestrator for coordinating the entire ML pipeline across nodes.
"""

import json
import logging
import time

from opentelemetry import trace

from ...components.embedding import EmbeddingGenerator
from ...config import get_settings
from ...enums import ServiceEndpoint
from ...telemetry import (
    SampledStageProfiler,
    batch_size_histogram,
    latency_histogram,
    rpc_duration_histogram,
    stage_duration_gauge,
)
from .batch_scheduler import Batch, BatchScheduler
from .rpc_client import RPCClient, RPCError
from .schemas import (
    GenerationRequest,
    GenerationResponse,
    PendingRequest,
    QueryResponse,
    RetrievalRequest,
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

        # Initialize RPC clients
        self.retrieval_client = RPCClient(
            base_url=retrieval_url or settings.retrieval_url,
            timeout_seconds=settings.request_timeout_seconds,
            max_retries=3,
        )

        self.generation_client = RPCClient(
            base_url=generation_url or settings.generation_url,
            timeout_seconds=settings.request_timeout_seconds,
            max_retries=3,
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
        )

        logger.info(
            "Orchestrator initialized with batch_size=%d, timeout=%dms",
            final_batch_size,
            final_batch_timeout_ms,
        )

    async def start(self) -> None:
        """Start the orchestrator."""
        await self.batch_scheduler.start()
        logger.info("Orchestrator started")

    async def stop(self) -> None:
        """Stop the orchestrator and cleanup resources."""
        await self.batch_scheduler.stop()
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

        # Create pending request
        pending_request = PendingRequest(
            request_id=request_id,
            query=query,
            timestamp=start_time,
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
        Process a batch of requests through the pipeline.

        Steps:
        1. Call /retrieve for all queries in batch
        2. Call /generate for all queries in batch
        3. Return list of QueryResponse in same order as batch.requests

        Args:
            batch: Batch of requests to process

        Returns:
            List of QueryResponse objects
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

        # Step 1: Retrieve documents from Node 1
        retrieval_start = time.time()

        # Generate embeddings if generator is available
        embeddings = None
        if self.embedding_generator and getattr(self.embedding_generator, "is_loaded", False):
            queries = [req.query for req in batch.requests]
            logger.debug("Generating embeddings for %d queries in gateway", len(queries))
            with profiler.track("gateway.embedding"):
                embeddings = self.embedding_generator.encode(queries)
                if hasattr(embeddings, "tolist"):
                    embeddings = embeddings.tolist()

        retrieval_requests = [
            RetrievalRequest(
                request_id=req.request_id,
                query=req.query,
                embedding=embeddings[i] if embeddings is not None else None,
            )
            for i, req in enumerate(batch.requests)
        ]

        with profiler.track("gateway.retrieval_rpc"):
            retrieval_responses = await self._call_retrieval_service(
                batch.batch_id,
                retrieval_requests,
            )
        retrieval_duration = time.time() - retrieval_start
        stage_duration_gauge.labels(
            run_id=settings.profiling_run_id, node=node_label, stage="gateway.retrieval_rpc"
        ).set(retrieval_duration)

        # Step 2: Generate responses from Node 2
        generation_start = time.time()
        generation_requests = [
            GenerationRequest(
                request_id=req.request_id,
                query=req.query,
                docs=resp.docs,
            )
            for req, resp in zip(batch.requests, retrieval_responses, strict=False)
        ]

        with profiler.track("gateway.generation_rpc"):
            generation_responses = await self._call_generation_service(
                batch.batch_id,
                generation_requests,
            )
        generation_duration = time.time() - generation_start
        stage_duration_gauge.labels(
            run_id=settings.profiling_run_id, node=node_label, stage="gateway.generation_rpc"
        ).set(generation_duration)

        # Step 3: Build final responses
        final_responses = [
            QueryResponse(
                request_id=gen_resp.request_id,
                generated_response=gen_resp.generated_response,
                sentiment=gen_resp.sentiment,
                is_toxic=gen_resp.is_toxic,
            )
            for gen_resp in generation_responses
        ]

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
                    "node1_ms": round(retrieval_duration * 1000, 2),
                    "node2_ms": round(generation_duration * 1000, 2),
                    "timestamp": batch_start,
                }
            )
        )

        # Record latency metric
        latency_histogram.labels(
            run_id=settings.profiling_run_id, node=node_label, service="gateway"
        ).observe(batch_elapsed)

        return final_responses

    async def _call_retrieval_service(
        self,
        batch_id: int,
        requests: list[RetrievalRequest],
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
            payload = {
                "batch_id": str(batch_id),
                "items": [req.model_dump() for req in requests],
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
        requests: list[GenerationRequest],
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
            payload = {
                "batch_id": str(batch_id),
                "items": [req.model_dump() for req in requests],
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

    def set_embedding_generator(self, embedding_generator: EmbeddingGenerator) -> None:
        """Set the embedding generator component."""
        self.embedding_generator = embedding_generator
