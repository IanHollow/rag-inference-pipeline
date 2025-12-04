"""
Tests for the batch scheduler module.

Tests cover:
- AdaptiveBatchPolicy behavior
- Batch dataclass
- BatchScheduler advanced scenarios
"""

import asyncio
import time
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from pipeline.services.gateway.batch_scheduler import AdaptiveBatchPolicy, Batch, BatchScheduler
from pipeline.services.gateway.schemas import PendingRequest


if TYPE_CHECKING:
    from pipeline.services.gateway.schemas import PendingRequestStruct


class TestAdaptiveBatchPolicy:
    """Tests for AdaptiveBatchPolicy class."""

    def test_init_default_values(self) -> None:
        """Test initialization with default values."""
        policy = AdaptiveBatchPolicy()

        assert policy.min_batch_size == 1
        assert policy.max_batch_size == 32
        assert policy.min_delay_sec == 0.05
        assert policy.max_delay_sec == 0.5
        assert policy.current_batch_size == 1.0
        assert policy.current_delay == 0.05

    def test_init_custom_values(self) -> None:
        """Test initialization with custom values."""
        policy = AdaptiveBatchPolicy(
            min_batch_size=2,
            max_batch_size=64,
            min_delay_sec=0.01,
            max_delay_sec=1.0,
        )

        assert policy.min_batch_size == 2
        assert policy.max_batch_size == 64
        assert policy.min_delay_sec == 0.01
        assert policy.max_delay_sec == 1.0

    def test_update_low_queue_depth(self) -> None:
        """Test policy update with low queue depth."""
        policy = AdaptiveBatchPolicy(
            min_batch_size=1,
            max_batch_size=32,
            min_delay_sec=0.01,
            max_delay_sec=0.5,
        )

        batch_size, delay = policy.update(queue_depth=2)

        # Low queue depth should keep values near minimum
        assert batch_size >= policy.min_batch_size
        assert batch_size <= policy.max_batch_size
        assert delay >= policy.min_delay_sec

    def test_update_medium_queue_depth(self) -> None:
        """Test policy update with medium queue depth (5-10)."""
        policy = AdaptiveBatchPolicy(
            min_batch_size=1,
            max_batch_size=32,
            min_delay_sec=0.01,
            max_delay_sec=0.5,
        )

        # Call multiple times to stabilize
        for _ in range(10):
            batch_size, _ = policy.update(queue_depth=7)

        # Should converge toward mid-range values
        assert batch_size > policy.min_batch_size
        assert batch_size <= policy.max_batch_size

    def test_update_high_queue_depth(self) -> None:
        """Test policy update with high queue depth (>= max_batch_size)."""
        policy = AdaptiveBatchPolicy(
            min_batch_size=1,
            max_batch_size=32,
            min_delay_sec=0.01,
            max_delay_sec=0.5,
        )

        # Call multiple times to stabilize - use queue_depth >= max_batch_size
        for _ in range(20):
            batch_size, delay = policy.update(queue_depth=32)

        # Should converge toward maximum values (EWMA may not reach exact max)
        assert batch_size >= policy.max_batch_size * 0.9  # Allow 10% tolerance
        assert delay >= policy.max_delay_sec * 0.9

    def test_update_ewma_smoothing(self) -> None:
        """Test that EWMA smoothing prevents abrupt changes."""
        policy = AdaptiveBatchPolicy(
            min_batch_size=1,
            max_batch_size=32,
            min_delay_sec=0.01,
            max_delay_sec=0.5,
        )

        # Start with high queue depth
        for _ in range(10):
            policy.update(queue_depth=15)

        batch_before = policy.current_batch_size

        # Suddenly low queue depth
        _ = policy.update(queue_depth=1)

        # Change should be gradual, not instant
        assert policy.current_batch_size < batch_before
        assert policy.current_batch_size > policy.min_batch_size

    def test_update_clamps_values(self) -> None:
        """Test that update clamps values to valid range."""
        policy = AdaptiveBatchPolicy(
            min_batch_size=4,
            max_batch_size=16,
            min_delay_sec=0.1,
            max_delay_sec=0.3,
        )

        batch_size, delay = policy.update(queue_depth=0)

        assert batch_size >= policy.min_batch_size
        assert batch_size <= policy.max_batch_size
        assert delay >= policy.min_delay_sec
        assert delay <= policy.max_delay_sec


class TestBatch:
    """Tests for Batch dataclass."""

    def test_batch_creation(self) -> None:
        """Test creating a Batch."""
        requests: list[PendingRequest | PendingRequestStruct] = [
            PendingRequest(request_id="1", query="q1", timestamp=time.time()),
            PendingRequest(request_id="2", query="q2", timestamp=time.time()),
        ]

        batch: Batch[str] = Batch(batch_id=1, requests=requests)

        assert batch.batch_id == 1
        assert len(batch.requests) == 2
        assert batch.futures == []
        assert batch.created_at > 0

    def test_batch_len(self) -> None:
        """Test batch length."""
        requests: list[PendingRequest | PendingRequestStruct] = [
            PendingRequest(request_id=str(i), query=f"q{i}", timestamp=time.time())
            for i in range(5)
        ]

        batch: Batch[str] = Batch(batch_id=1, requests=requests)

        assert len(batch) == 5

    def test_batch_empty(self) -> None:
        """Test empty batch."""
        batch: Batch[str] = Batch(batch_id=1, requests=[])

        assert len(batch) == 0

    def test_batch_with_futures(self) -> None:
        """Test batch with futures."""
        loop = asyncio.new_event_loop()
        requests: list[PendingRequest | PendingRequestStruct] = [
            PendingRequest(request_id="1", query="q1", timestamp=time.time())
        ]
        futures: list[asyncio.Future[str]] = [loop.create_future()]

        batch: Batch[str] = Batch(batch_id=1, requests=requests, futures=futures)

        assert len(batch.futures) == 1
        loop.close()


class TestBatchSchedulerAdvanced:
    """Advanced tests for BatchScheduler."""

    @pytest_asyncio.fixture
    async def process_fn(self) -> AsyncMock:
        """Create a mock process function."""

        async def mock_process(batch: Batch) -> list[str]:
            return [f"result_{req.request_id}" for req in batch.requests]

        return AsyncMock(side_effect=mock_process)

    @pytest.mark.asyncio
    async def test_scheduler_initialization(self, process_fn: AsyncMock) -> None:
        """Test scheduler initialization."""
        scheduler = BatchScheduler(
            batch_size=8,
            max_batch_delay_ms=200,
            process_batch_fn=process_fn,
            service_name="test",
            enable_adaptive=False,
        )

        assert scheduler.batch_size == 8
        assert scheduler.max_batch_delay_sec == 0.2
        assert scheduler.policy is None
        assert not scheduler._running

    @pytest.mark.asyncio
    async def test_scheduler_with_adaptive_policy(self, process_fn: AsyncMock) -> None:
        """Test scheduler with adaptive policy enabled."""
        scheduler = BatchScheduler(
            batch_size=16,
            max_batch_delay_ms=300,
            process_batch_fn=process_fn,
            enable_adaptive=True,
        )

        assert scheduler.policy is not None
        # Adaptive policy uses batch_size as min and 4x as max for scaling under load
        assert scheduler.policy.min_batch_size == 16
        assert scheduler.policy.max_batch_size == 64  # 16 * 4
        assert scheduler.policy.max_delay_sec == 0.3

    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self, process_fn: AsyncMock) -> None:
        """Test scheduler start and stop."""
        scheduler = BatchScheduler(
            batch_size=4,
            max_batch_delay_ms=100,
            process_batch_fn=process_fn,
        )

        await scheduler.start()
        assert scheduler._running is True

        await scheduler.stop()
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_batch_counter_increments(self, process_fn: AsyncMock) -> None:
        """Test that batch counter increments."""
        scheduler = BatchScheduler(
            batch_size=1,  # Flush immediately
            max_batch_delay_ms=100,
            process_batch_fn=process_fn,
        )
        await scheduler.start()

        initial_counter = scheduler._batch_counter

        request = PendingRequest(request_id="test", query="q", timestamp=time.time())
        await scheduler.enqueue(request)

        assert scheduler._batch_counter > initial_counter

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_concurrent_requests(self, process_fn: AsyncMock) -> None:
        """Test handling concurrent requests."""
        scheduler = BatchScheduler(
            batch_size=4,
            max_batch_delay_ms=50,
            process_batch_fn=process_fn,
        )
        await scheduler.start()

        requests = [
            PendingRequest(request_id=f"req_{i}", query=f"q{i}", timestamp=time.time())
            for i in range(10)
        ]

        results = await asyncio.gather(*[scheduler.enqueue(req) for req in requests])

        assert len(results) == 10
        assert all(r.startswith("result_") for r in results)

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_process_error_propagation(self) -> None:
        """Test that errors from process_fn are propagated to callers as RuntimeError."""

        async def failing_process(batch: Batch) -> list[str]:
            msg = "Processing failed"
            raise ValueError(msg)

        scheduler = BatchScheduler(
            batch_size=1,
            max_batch_delay_ms=100,
            process_batch_fn=failing_process,
        )
        await scheduler.start()

        request = PendingRequest(request_id="test", query="q", timestamp=time.time())

        # Errors are wrapped in RuntimeError by the batch scheduler
        with pytest.raises(RuntimeError, match="Batch processing failed"):
            await scheduler.enqueue(request)

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_service_name_tracking(self, process_fn: AsyncMock) -> None:
        """Test that service name is properly set."""
        scheduler = BatchScheduler(
            batch_size=4,
            max_batch_delay_ms=100,
            process_batch_fn=process_fn,
            service_name="custom_service",
        )

        assert scheduler.service_name == "custom_service"


class TestBatchSchedulerTimingBehavior:
    """Tests for BatchScheduler timing behavior."""

    @pytest.mark.asyncio
    async def test_batch_flushes_at_max_size(self) -> None:
        """Test that batch flushes when max size is reached."""
        results_received: list[int] = []

        async def track_batch_size(batch: Batch) -> list[str]:
            results_received.append(len(batch))
            return [f"result_{req.request_id}" for req in batch.requests]

        scheduler = BatchScheduler(
            batch_size=3,
            max_batch_delay_ms=1000,  # Long timeout
            process_batch_fn=track_batch_size,
        )
        await scheduler.start()

        requests = [
            PendingRequest(request_id=str(i), query=f"q{i}", timestamp=time.time())
            for i in range(3)
        ]

        # Submit 3 requests (equal to batch_size)
        await asyncio.gather(*[scheduler.enqueue(req) for req in requests])

        # Batch should have been flushed with size 3
        assert 3 in results_received

        await scheduler.stop()

    @pytest.mark.asyncio
    async def test_batch_flushes_on_timeout(self) -> None:
        """Test that batch flushes after timeout with partial batch."""
        results_received: list[int] = []

        async def track_batch_size(batch: Batch) -> list[str]:
            results_received.append(len(batch))
            return [f"result_{req.request_id}" for req in batch.requests]

        scheduler = BatchScheduler(
            batch_size=10,  # Large batch size
            max_batch_delay_ms=50,  # Short timeout
            process_batch_fn=track_batch_size,
        )
        await scheduler.start()

        request = PendingRequest(request_id="single", query="q", timestamp=time.time())

        # Submit single request
        await scheduler.enqueue(request)

        # Should have flushed partial batch
        assert 1 in results_received

        await scheduler.stop()
