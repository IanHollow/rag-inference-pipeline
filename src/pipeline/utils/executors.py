import asyncio
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import logging
import os
from typing import ClassVar, TypeVar

from pipeline.config import PipelineSettings


logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceExecutorFactory:
    _executors: ClassVar[dict[str, ThreadPoolExecutor]] = {}
    _settings: ClassVar[PipelineSettings | None] = None

    @classmethod
    def initialize(cls, settings: PipelineSettings) -> None:
        cls._settings = settings

    @classmethod
    def get_executor(cls, service_name: str) -> ThreadPoolExecutor:
        if service_name not in cls._executors:
            if cls._settings is None:
                # Fallback if not initialized
                cpu_count = os.cpu_count() or 1
                max_workers = min(8, cpu_count)
                logger.warning(
                    "ServiceExecutorFactory not initialized, using default max_workers=%s",
                    max_workers,
                )
            else:
                # Clamp to available cores to avoid over-subscription
                cpu_count = os.cpu_count() or 1
                max_workers = min(cls._settings.cpu_worker_threads, cpu_count)

            logger.info("Creating executor for %s with max_workers=%s", service_name, max_workers)
            cls._executors[service_name] = ThreadPoolExecutor(
                max_workers=max_workers, thread_name_prefix=f"{service_name}-worker"
            )
        return cls._executors[service_name]

    @classmethod
    async def run_cpu_bound(
        cls,
        loop: asyncio.AbstractEventLoop,
        service_name: str,
        func: Callable[..., T],
        *args: object,
    ) -> T:
        """
        Run a CPU-bound function in the service's thread pool.
        """
        executor = cls.get_executor(service_name)
        return await loop.run_in_executor(executor, partial(func, *args))

    @classmethod
    def shutdown(cls) -> None:
        for name, executor in cls._executors.items():
            logger.info("Shutting down executor for %s", name)
            executor.shutdown(wait=True)
        cls._executors.clear()
