"""
Batch scheduler for grouping incoming requests.

This module implements opportunistic batching: requests are grouped by batch_size
or max_batch_delay_ms, whichever comes first. Uses asyncio.Future for result handling
to avoid busy-waiting.
"""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
import logging
import time
from typing import Generic, TypeVar

from pipeline.config import get_settings
from pipeline.telemetry import batch_flush_counter, queue_depth_gauge

from .schemas import PendingRequest, PendingRequestStruct


logger = logging.getLogger(__name__)
settings = get_settings()

T = TypeVar("T")


class AdaptiveBatchPolicy:
    """
    Policy for dynamically adjusting batch delay based on load.
    """

    def __init__(
        self,
        max_batch_size: int = 16,
        min_delay_sec: float = 0.01,
        max_delay_sec: float = 0.2,
    ) -> None:
        self.max_batch_size = max_batch_size
        self.min_delay_sec = min_delay_sec
        self.max_delay_sec = max_delay_sec

        # Current state - start with shorter delay for responsiveness
        self.current_delay = min_delay_sec
        # Track recent queue depths to detect load trends
        self._recent_depths: list[int] = []
        self._max_history = 10

    def update(self, queue_depth: int) -> float:
        """
        Update policy based on current queue depth.

        Returns:
            The recommended delay in seconds before flushing a partial batch.
        """
        # Track recent queue depths
        self._recent_depths.append(queue_depth)
        if len(self._recent_depths) > self._max_history:
            self._recent_depths.pop(0)

        # Use average recent depth to smooth out fluctuations
        avg_depth = sum(self._recent_depths) / len(self._recent_depths)

        # Scale delay based on average queue depth relative to max batch size
        # Higher queue depth = more load = longer delay to batch more
        if avg_depth >= self.max_batch_size:
            target_delay = self.max_delay_sec
        else:
            ratio = avg_depth / self.max_batch_size
            target_delay = self.min_delay_sec + ratio * (self.max_delay_sec - self.min_delay_sec)

        # Smooth transitions using EWMA
        self.current_delay = (self.current_delay * 0.7) + (target_delay * 0.3)

        # Clamp value
        return max(self.min_delay_sec, min(self.current_delay, self.max_delay_sec))


@dataclass
class Batch(Generic[T]):
    """Represents a batch of requests ready for processing."""

    batch_id: int
    requests: list[PendingRequest | PendingRequestStruct]
    futures: list[asyncio.Future[T]] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)

    def __len__(self) -> int:
        """Return the number of requests in this batch."""
        return len(self.requests)


class BatchScheduler(Generic[T]):
    """
    Async batch scheduler that groups requests for processing.
    """

    def __init__(
        self,
        batch_size: int,
        max_batch_delay_ms: int,
        process_batch_fn: Callable[[Batch[T]], Awaitable[list[T]]],
        service_name: str = "gateway",
        enable_adaptive: bool = False,
    ) -> None:
        """
        Initialize the batch scheduler.

        Args:
            batch_size: Maximum number of requests per batch
            max_batch_delay_ms: Maximum milliseconds to wait for a full batch
            process_batch_fn: Async function to call when a batch is ready
            service_name: Name of the service for metrics
            enable_adaptive: Whether to enable adaptive batching
        """
        self.batch_size = batch_size
        self.max_batch_delay_sec = max_batch_delay_ms / 1000.0
        self.process_batch_fn = process_batch_fn
        self.service_name = service_name
        self.enable_adaptive = enable_adaptive

        self.policy: AdaptiveBatchPolicy | None
        self._configured_batch_size = batch_size  # Store original configured value (used as max)
        if enable_adaptive:
            # Adaptive policy adjusts delay based on load, batch_size is fixed as maximum
            self.policy = AdaptiveBatchPolicy(
                max_batch_size=batch_size,
                min_delay_sec=min(0.01, self.max_batch_delay_sec),
                max_delay_sec=self.max_batch_delay_sec,
            )
        else:
            self.policy = None

        self._pending_requests: list[PendingRequest | PendingRequestStruct] = []
        self._pending_futures: list[asyncio.Future[T]] = []
        self._batch_counter = 0
        self._lock = asyncio.Lock()
        self._timer_task: asyncio.Task[None] | None = None
        self._background_tasks: set[asyncio.Task[None]] = set()
        self._running = False

    async def start(self) -> None:
        """Start the batch scheduler."""
        self._running = True
        logger.info(
            "BatchScheduler started: batch_size=%d, max_delay=%.3fs",
            self.batch_size,
            self.max_batch_delay_sec,
        )

    async def stop(self) -> None:
        """Stop the batch scheduler and process any remaining requests."""
        self._running = False

        # Cancel timer if running
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()
            await asyncio.wait([self._timer_task])

        # Process remaining requests
        async with self._lock:
            if self._pending_requests:
                await self._flush_batch(reason="shutdown")

        logger.info("BatchScheduler stopped")

    async def enqueue(self, request: PendingRequest | PendingRequestStruct) -> T:
        """
        Add a request to the queue and wait for the result.

        Args:
            request: The request to process

        Returns:
            The result of processing the request
        """
        if not self._running:
            msg = "BatchScheduler is not running"
            raise RuntimeError(msg)

        # Create a Future for this request's result
        loop = asyncio.get_running_loop()
        future: asyncio.Future[T] = loop.create_future()

        async with self._lock:
            self._pending_requests.append(request)
            self._pending_futures.append(future)

            # Update adaptive delay if enabled (batch_size stays fixed)
            if self.enable_adaptive and self.policy:
                new_delay = self.policy.update(len(self._pending_requests))
                if abs(new_delay - self.max_batch_delay_sec) > 0.001:
                    logger.debug("Adaptive batching: delay=%.3fs", new_delay)
                    self.max_batch_delay_sec = new_delay

            # Update queue depth metric
            queue_depth_gauge.labels(
                run_id=settings.profiling_run_id,
                node=str(settings.node_number),
                service=self.service_name,
            ).set(len(self._pending_requests))

            logger.debug(
                "Enqueued request %s (pending: %d/%d)",
                request.request_id,
                len(self._pending_requests),
                self.batch_size,
            )

            # If batch is full (reached max batch size), flush immediately
            if len(self._pending_requests) >= self.batch_size:
                await self._flush_batch(reason="full")
            elif len(self._pending_requests) == 1:
                # First request in batch, start timer
                self._timer_task = asyncio.create_task(self._batch_timer())

        # Wait for result
        return await future

        # Wait for result
        return await future

    async def _batch_timer(self) -> None:
        """Wait for max_batch_delay and then flush the batch."""
        try:
            await asyncio.sleep(self.max_batch_delay_sec)
            async with self._lock:
                if self._pending_requests:
                    logger.debug(
                        "Batch timer expired, flushing %d requests", len(self._pending_requests)
                    )
                    await self._flush_batch(reason="timeout")
        except asyncio.CancelledError:
            # Timer was cancelled because batch was flushed early
            pass

    async def _flush_batch(self, reason: str = "unknown") -> None:
        """
        Flush the current batch for processing.

        Must be called while holding self._lock.
        """
        if not self._pending_requests:
            return

        # Cancel timer if it's running
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()

        # Record flush reason
        batch_flush_counter.labels(
            run_id=settings.profiling_run_id,
            node=str(settings.node_number),
            service=self.service_name,
            reason=reason,
        ).inc()

        # Create batch
        self._batch_counter += 1
        batch: Batch[T] = Batch(
            batch_id=self._batch_counter,
            requests=self._pending_requests[:],
            futures=self._pending_futures[:],
        )

        # Clear pending lists
        self._pending_requests.clear()
        self._pending_futures.clear()

        # Update queue depth metric
        queue_depth_gauge.labels(
            run_id=settings.profiling_run_id,
            node=str(settings.node_number),
            service=self.service_name,
        ).set(0)

        logger.info(
            "Batch %d <- %d requests (age: %.3fs, reason: %s)",
            batch.batch_id,
            len(batch),
            time.time() - batch.created_at,
            reason,
        )

        # Process batch asynchronously (don't block enqueue)
        task = asyncio.create_task(self._process_batch(batch))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _process_batch(self, batch: Batch[T]) -> None:
        """
        Process a batch and set results on futures.

        Args:
            batch: The batch to process
        """
        try:
            # Call the processing function
            results = await self.process_batch_fn(batch)
        except Exception:
            logger.exception("Error processing batch %d", batch.batch_id)
            # Set exception on all futures
            for future in batch.futures:
                if not future.done():
                    future.set_exception(RuntimeError("Batch processing failed"))
            return

        # Validate result count - this is a programming error, not a processing error
        if len(results) != len(batch.futures):
            msg = f"Result count mismatch: expected {len(batch.futures)}, got {len(results)}"
            error = ValueError(msg)
            logger.error(msg)
            for future in batch.futures:
                if not future.done():
                    future.set_exception(error)
            return

        for future, result in zip(batch.futures, results, strict=False):
            if not future.done():
                future.set_result(result)

        logger.debug("Batch %d processed successfully", batch.batch_id)
