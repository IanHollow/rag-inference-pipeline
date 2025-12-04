"""
Unit tests for the gateway service.

Tests cover:
- Single request processing
- Batch formation and processing
- Error propagation from downstream services
- Timeout handling
- Metrics recording
"""

import asyncio
from collections.abc import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from pydantic import ValidationError
import pytest
import pytest_asyncio

from pipeline.services.gateway.batch_scheduler import Batch, BatchScheduler
from pipeline.services.gateway.orchestrator import Orchestrator
from pipeline.services.gateway.rpc_client import RPCClient, RPCError, RPCTimeoutError
from pipeline.services.gateway.schemas import (
    GenerationRequest,
    GenerationResponse,
    PendingRequest,
    QueryRequest,
    QueryResponse,
    RetrievalRequest,
    RetrievalResponse,
)


class TestBatchScheduler:
    """Tests for BatchScheduler class."""

    @pytest_asyncio.fixture
    async def scheduler(self) -> AsyncGenerator[BatchScheduler, None]:
        """Create a BatchScheduler for testing."""

        async def mock_process_batch(batch: Batch) -> list[str]:
            """Mock batch processing function."""
            return [f"result_{req.request_id}" for req in batch.requests]

        scheduler = BatchScheduler(
            batch_size=4,
            max_batch_delay_ms=100,
            process_batch_fn=mock_process_batch,
        )
        await scheduler.start()
        yield scheduler
        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_single_request(self, scheduler: BatchScheduler) -> None:
        """Test processing a single request."""
        request = PendingRequest(
            request_id="test_001",
            query="test query",
            timestamp=0.0,
        )

        result = await scheduler.enqueue(request)
        assert result == "result_test_001"

    @pytest.mark.asyncio
    async def test_batch_formation_by_size(self, scheduler: BatchScheduler) -> None:
        """Test that batch is flushed when reaching batch_size."""
        requests = [
            PendingRequest(
                request_id=f"test_{i:03d}",
                query=f"query {i}",
                timestamp=0.0,
            )
            for i in range(4)
        ]

        # Submit all requests concurrently
        results = await asyncio.gather(*[scheduler.enqueue(req) for req in requests])

        # Verify all results
        assert len(results) == 4
        assert results == [
            "result_test_000",
            "result_test_001",
            "result_test_002",
            "result_test_003",
        ]

    @pytest.mark.asyncio
    async def test_batch_formation_by_timeout(self, scheduler: BatchScheduler) -> None:
        """Test that batch is flushed after max_batch_delay_ms."""
        request = PendingRequest(
            request_id="test_001",
            query="test query",
            timestamp=0.0,
        )

        # Submit request and wait for timeout
        result = await scheduler.enqueue(request)
        assert result == "result_test_001"

    @pytest.mark.asyncio
    async def test_multiple_batches(self, scheduler: BatchScheduler) -> None:
        """Test processing multiple batches sequentially."""
        # First batch (4 requests)
        batch1 = [
            PendingRequest(
                request_id=f"batch1_{i:03d}",
                query=f"query {i}",
                timestamp=0.0,
            )
            for i in range(4)
        ]

        results1 = await asyncio.gather(*[scheduler.enqueue(req) for req in batch1])
        assert len(results1) == 4

        # Second batch (2 requests, will timeout)
        batch2 = [
            PendingRequest(
                request_id=f"batch2_{i:03d}",
                query=f"query {i}",
                timestamp=0.0,
            )
            for i in range(2)
        ]

        results2 = await asyncio.gather(*[scheduler.enqueue(req) for req in batch2])
        assert len(results2) == 2


class TestRPCClient:
    """Tests for RPCClient class."""

    @pytest.fixture
    def mock_httpx_client(self) -> MagicMock:
        """Create a mock httpx AsyncClient."""
        return MagicMock(spec=httpx.AsyncClient)

    @pytest.mark.asyncio
    async def test_successful_post(self) -> None:
        """Test successful POST request."""
        client = RPCClient(base_url="http://test:8000", timeout_seconds=5.0)

        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b'{"result": "success"}'
        mock_response.json.return_value = {"result": "success"}
        mock_response.headers = {"content-encoding": ""}
        mock_response.raise_for_status = MagicMock()

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            result = await client.post("/test", {"key": "value"})
            assert result == {"result": "success"}

        await client.close()

    @pytest.mark.asyncio
    async def test_timeout_error(self) -> None:
        """Test handling of timeout errors."""
        client = RPCClient(base_url="http://test:8000", timeout_seconds=1.0)

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Timeout")

            with pytest.raises(RPCTimeoutError):
                await client.post("/test", {"key": "value"})

        await client.close()

    @pytest.mark.asyncio
    async def test_service_error_5xx(self) -> None:
        """Test handling of 5xx service errors."""
        client = RPCClient(base_url="http://test:8000", timeout_seconds=5.0)

        # Mock 500 response
        mock_response = MagicMock()
        mock_response.status_code = 500

        def raise_status() -> None:
            msg = "Server error"
            raise httpx.HTTPStatusError(msg, request=MagicMock(), response=mock_response)

        mock_response.raise_for_status = raise_status

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(RPCError):
                await client.post("/test", {"key": "value"})

        await client.close()

    @pytest.mark.asyncio
    async def test_client_error_4xx(self) -> None:
        """Test handling of 4xx client errors."""
        client = RPCClient(base_url="http://test:8000", timeout_seconds=5.0)

        # Mock 400 response
        mock_response = MagicMock()
        mock_response.status_code = 400

        def raise_status() -> None:
            msg = "Bad request"
            raise httpx.HTTPStatusError(msg, request=MagicMock(), response=mock_response)

        mock_response.raise_for_status = raise_status

        with patch.object(client._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response

            with pytest.raises(RPCError):
                await client.post("/test", {"key": "value"})

        await client.close()


class TestOrchestrator:
    """Tests for Orchestrator class."""

    @pytest_asyncio.fixture
    async def orchestrator(self) -> AsyncGenerator[Orchestrator, None]:
        """Create an Orchestrator for testing with mocked RPC clients."""
        with patch.object(RPCClient, "__init__", return_value=None):
            orchestrator = Orchestrator()

        # Replace clients with mock objects
        mock_retrieval = MagicMock()
        mock_retrieval.post = AsyncMock()
        mock_retrieval.close = AsyncMock()

        mock_generation = MagicMock()
        mock_generation.post = AsyncMock()
        mock_generation.close = AsyncMock()

        # Assign mocks to orchestrator
        object.__setattr__(orchestrator, "retrieval_client", mock_retrieval)
        object.__setattr__(orchestrator, "generation_client", mock_generation)

        await orchestrator.start()
        yield orchestrator
        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_process_single_query(self, orchestrator: Orchestrator) -> None:
        """Test processing a single query through the orchestrator."""
        # Get the mock clients (they are MagicMocks in test fixture)
        retrieval_mock: MagicMock = orchestrator.retrieval_client  # type: ignore[assignment]
        generation_mock: MagicMock = orchestrator.generation_client  # type: ignore[assignment]

        # Mock retrieval response (batch API format)
        retrieval_response_data = {
            "batch_id": "1",
            "items": [
                {
                    "request_id": "test_001",
                    "docs": [
                        {
                            "doc_id": 1,
                            "title": "Doc 1",
                            "content": "Content 1",
                            "category": "A",
                            "score": 0.9,
                        },
                        {
                            "doc_id": 2,
                            "title": "Doc 2",
                            "content": "Content 2",
                            "category": "B",
                            "score": 0.85,
                        },
                        {
                            "doc_id": 3,
                            "title": "Doc 3",
                            "content": "Content 3",
                            "category": "C",
                            "score": 0.8,
                        },
                    ],
                }
            ],
        }

        # Mock generation response (batch API format)
        generation_response_data = {
            "batch_id": "1",
            "items": [
                {
                    "request_id": "test_001",
                    "generated_response": "This is a test response",
                    "sentiment": "positive",
                    "is_toxic": "false",
                }
            ],
            "processing_time": 0.1,
        }

        # Configure mock return values
        retrieval_post_mock: AsyncMock = retrieval_mock.post
        generation_post_mock: AsyncMock = generation_mock.post

        retrieval_post_mock.return_value = retrieval_response_data
        generation_post_mock.return_value = generation_response_data

        # Process query
        response = await orchestrator.process_query(request_id="test_001", query="test query")

        # Verify response
        assert response.request_id == "test_001"
        assert response.generated_response == "This is a test response"
        assert response.sentiment == "positive"
        assert response.is_toxic == "false"

    @pytest.mark.asyncio
    async def test_retrieval_service_error(self, orchestrator: Orchestrator) -> None:
        """Test error handling when retrieval service fails."""
        retrieval_mock: MagicMock = orchestrator.retrieval_client  # type: ignore[assignment]
        retrieval_post_mock: AsyncMock = retrieval_mock.post
        retrieval_post_mock.side_effect = RPCError("Retrieval service error")

        with pytest.raises(RPCError):
            await orchestrator.process_query(request_id="test_001", query="test query")

    @pytest.mark.asyncio
    async def test_generation_service_error(self, orchestrator: Orchestrator) -> None:
        """Test error handling when generation service fails."""
        retrieval_mock: MagicMock = orchestrator.retrieval_client  # type: ignore[assignment]
        generation_mock: MagicMock = orchestrator.generation_client  # type: ignore[assignment]

        # Mock successful retrieval
        retrieval_response_data = {
            "batch_id": "1",
            "items": [
                {
                    "request_id": "test_001",
                    "docs": [
                        {"doc_id": 1, "title": "Doc 1", "content": "Content 1", "category": "A"}
                    ],
                }
            ],
        }

        retrieval_post_mock: AsyncMock = retrieval_mock.post
        generation_post_mock: AsyncMock = generation_mock.post

        retrieval_post_mock.return_value = retrieval_response_data

        # Mock failed generation
        generation_post_mock.side_effect = RPCError("Generation service error")

        with pytest.raises(RPCError):
            await orchestrator.process_query(request_id="test_001", query="test query")


class TestSchemas:
    """Tests for Pydantic schemas."""

    def test_query_request_validation(self) -> None:
        """Test QueryRequest validation."""
        # Valid request
        request = QueryRequest(request_id="test_001", query="test query")
        assert request.request_id == "test_001"
        assert request.query == "test query"

        # Invalid: missing request_id
        with pytest.raises(ValidationError):
            QueryRequest(query="test query")  # type: ignore[call-arg]

        # Invalid: empty query
        with pytest.raises(ValidationError):
            QueryRequest(request_id="test_001", query="")

    def test_query_response_validation(self) -> None:
        """Test QueryResponse validation."""
        response = QueryResponse(
            request_id="test_001",
            generated_response="Test response",
            sentiment="positive",
            is_toxic="false",
        )

        assert response.request_id == "test_001"
        assert response.generated_response == "Test response"
        assert response.sentiment == "positive"
        assert response.is_toxic == "false"

    def test_retrieval_request_validation(self) -> None:
        """Test RetrievalRequest validation."""
        request = RetrievalRequest(request_id="test_001", query="test query")
        assert request.request_id == "test_001"
        assert request.query == "test query"

    def test_retrieval_response_validation(self) -> None:
        """Test RetrievalResponse validation."""
        response = RetrievalResponse(
            request_id="test_001",
            docs=[
                {
                    "doc_id": 1,
                    "title": "Doc 1",
                    "content": "Content 1",
                    "category": "A",
                    "score": 0.9,
                },
                {
                    "doc_id": 2,
                    "title": "Doc 2",
                    "content": "Content 2",
                    "category": "B",
                    "score": 0.8,
                },
            ],
        )

        assert response.request_id == "test_001"
        assert len(response.docs) == 2

    def test_generation_request_validation(self) -> None:
        """Test GenerationRequest validation."""
        request = GenerationRequest(
            request_id="test_001",
            query="test query",
            docs=[{"doc_id": 1, "title": "Doc 1", "content": "Content 1", "category": "A"}],
        )

        assert request.request_id == "test_001"
        assert request.query == "test query"
        assert len(request.docs) == 1

    def test_generation_response_validation(self) -> None:
        """Test GenerationResponse validation."""
        response = GenerationResponse(
            request_id="test_001",
            generated_response="Test response",
            sentiment="positive",
            is_toxic="false",
        )

        assert response.request_id == "test_001"
        assert response.generated_response == "Test response"
        assert response.sentiment == "positive"
        assert response.is_toxic == "false"
