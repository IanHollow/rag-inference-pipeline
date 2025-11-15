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

from .schemas import PendingRequest

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class Batch(Generic[T]):
    """Represents a batch of requests ready for processing."""

    batch_id: int
    requests: list[PendingRequest]
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
    ) -> None:
        """
        Initialize the batch scheduler.

        Args:
            batch_size: Maximum number of requests per batch
            max_batch_delay_ms: Maximum milliseconds to wait for a full batch
            process_batch_fn: Async function to call when a batch is ready
        """
        self.batch_size = batch_size
        self.max_batch_delay_sec = max_batch_delay_ms / 1000.0
        self.process_batch_fn = process_batch_fn

        self._pending_requests: list[PendingRequest] = []
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
                await self._flush_batch()

        logger.info("BatchScheduler stopped")

    async def enqueue(self, request: PendingRequest) -> T:
        """
        Add a request to the batch queue and wait for its result.

        Args:
            request: The request to process

        Returns:
            The result of processing this request
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

            logger.debug(
                "Enqueued request %s (pending: %d/%d)",
                request.request_id,
                len(self._pending_requests),
                self.batch_size,
            )

            # If batch is full, flush immediately
            if len(self._pending_requests) >= self.batch_size:
                await self._flush_batch()
            elif len(self._pending_requests) == 1:
                # First request in batch, start timer
                self._timer_task = asyncio.create_task(self._batch_timer())

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
                    await self._flush_batch()
        except asyncio.CancelledError:
            # Timer was cancelled because batch was flushed early
            pass

    async def _flush_batch(self) -> None:
        """
        Flush the current batch for processing.

        Must be called while holding self._lock.
        """
        if not self._pending_requests:
            return

        # Cancel timer if it's running
        if self._timer_task and not self._timer_task.done():
            self._timer_task.cancel()

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

        logger.info(
            "Batch %d <- %d requests (age: %.3fs)",
            batch.batch_id,
            len(batch),
            time.time() - batch.created_at,
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

            # Set results on futures
            if len(results) != len(batch.futures):
                msg = f"Result count mismatch: expected {len(batch.futures)}, got {len(results)}"
                raise ValueError(msg)

            for future, result in zip(batch.futures, results, strict=False):
                if not future.done():
                    future.set_result(result)

            logger.debug("Batch %d processed successfully", batch.batch_id)

        except Exception:
            logger.exception("Error processing batch %d", batch.batch_id)
            # Set exception on all futures
            for future in batch.futures:
                if not future.done():
                    future.set_exception(RuntimeError("Batch processing failed"))
