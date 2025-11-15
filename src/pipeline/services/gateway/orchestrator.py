"""
Orchestrator for coordinating the entire ML pipeline across nodes.
"""

import logging
import time

from ...config import get_settings
from ...enums import ServiceEndpoint
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


class Orchestrator:
    """
    Orchestrates the ML pipeline by coordinating batching and RPC calls.
    """

    def __init__(self) -> None:
        """Initialize the orchestrator."""
        # Initialize RPC clients
        self.retrieval_client = RPCClient(
            base_url=settings.retrieval_url,
            timeout_seconds=settings.request_timeout_seconds,
            max_retries=3,
        )

        self.generation_client = RPCClient(
            base_url=settings.generation_url,
            timeout_seconds=settings.request_timeout_seconds,
            max_retries=3,
        )

        # Initialize batch scheduler
        self.batch_scheduler: BatchScheduler[QueryResponse] = BatchScheduler(
            batch_size=settings.gateway_batch_size,
            max_batch_delay_ms=settings.gateway_batch_timeout_ms,
            process_batch_fn=self._process_batch,
        )

        logger.info(
            "Orchestrator initialized with batch_size=%d, timeout=%dms",
            settings.gateway_batch_size,
            settings.gateway_batch_timeout_ms,
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

    async def _process_batch(self, batch: Batch) -> list[QueryResponse]:
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

        # Step 1: Retrieve documents from Node 1
        retrieval_requests = [
            RetrievalRequest(
                request_id=req.request_id,
                query=req.query,
            )
            for req in batch.requests
        ]

        retrieval_responses = await self._call_retrieval_service(
            batch.batch_id,
            retrieval_requests,
        )

        # Step 2: Generate responses from Node 2
        generation_requests = [
            GenerationRequest(
                request_id=req.request_id,
                query=req.query,
                docs=resp.docs,
            )
            for req, resp in zip(batch.requests, retrieval_responses, strict=False)
        ]

        generation_responses = await self._call_generation_service(
            batch.batch_id,
            generation_requests,
        )

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
        logger.info(
            "Batch %d completed in %.3fs",
            batch.batch_id,
            batch_elapsed,
        )

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

            response_data = await self.retrieval_client.post(
                ServiceEndpoint.RETRIEVE.value,
                payload,
            )

            # Parse responses
            responses = [RetrievalResponse(**resp) for resp in response_data["items"]]

            logger.debug(
                "Retrieval service returned %d responses for batch %d",
                len(responses),
                batch_id,
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

            response_data = await self.generation_client.post(
                ServiceEndpoint.GENERATE.value,
                payload,
            )

            # Parse responses
            responses = [GenerationResponse(**resp) for resp in response_data["items"]]

            logger.debug(
                "Generation service returned %d responses for batch %d",
                len(responses),
                batch_id,
            )

            return responses

        except Exception as e:
            logger.exception(
                "Generation service failed for batch %d",
                batch_id,
            )
            raise RPCError(f"Generation service error: {e}") from e
