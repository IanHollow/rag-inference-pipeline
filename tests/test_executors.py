"""
Tests for the executors module.

Tests cover:
- ServiceExecutorFactory initialization
- Executor creation and caching
- run_cpu_bound execution
- Shutdown behavior
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from pipeline.config import PipelineSettings
from pipeline.utils.executors import ServiceExecutorFactory


class TestServiceExecutorFactory:
    """Tests for ServiceExecutorFactory class."""

    def setup_method(self) -> None:
        """Reset factory state before each test."""
        ServiceExecutorFactory._executors.clear()
        ServiceExecutorFactory._settings = None

    def teardown_method(self) -> None:
        """Clean up executors after each test."""
        ServiceExecutorFactory.shutdown()

    def test_initialize_sets_settings(self) -> None:
        """Test that initialize sets the settings."""
        settings = PipelineSettings(node_number=0)

        ServiceExecutorFactory.initialize(settings)

        assert ServiceExecutorFactory._settings == settings

    def test_get_executor_creates_new(self) -> None:
        """Test that get_executor creates a new executor for new service."""
        settings = PipelineSettings(node_number=0, cpu_worker_threads=4)
        ServiceExecutorFactory.initialize(settings)

        executor = ServiceExecutorFactory.get_executor("test_service")

        assert isinstance(executor, ThreadPoolExecutor)
        assert "test_service" in ServiceExecutorFactory._executors

    def test_get_executor_returns_cached(self) -> None:
        """Test that get_executor returns cached executor for same service."""
        settings = PipelineSettings(node_number=0)
        ServiceExecutorFactory.initialize(settings)

        executor1 = ServiceExecutorFactory.get_executor("test_service")
        executor2 = ServiceExecutorFactory.get_executor("test_service")

        assert executor1 is executor2

    def test_get_executor_different_services(self) -> None:
        """Test that different services get different executors."""
        settings = PipelineSettings(node_number=0)
        ServiceExecutorFactory.initialize(settings)

        executor1 = ServiceExecutorFactory.get_executor("service_a")
        executor2 = ServiceExecutorFactory.get_executor("service_b")

        assert executor1 is not executor2
        assert len(ServiceExecutorFactory._executors) == 2

    def test_get_executor_without_initialization(self) -> None:
        """Test that get_executor works without initialization (fallback)."""
        # Don't initialize - should use fallback

        executor = ServiceExecutorFactory.get_executor("fallback_service")

        assert isinstance(executor, ThreadPoolExecutor)

    @patch("os.cpu_count")
    def test_get_executor_respects_cpu_count(self, mock_cpu_count: MagicMock) -> None:
        """Test that executor respects CPU count limits."""
        mock_cpu_count.return_value = 2
        settings = PipelineSettings(
            node_number=0, cpu_worker_threads=8
        )  # Request more than available
        ServiceExecutorFactory.initialize(settings)

        executor = ServiceExecutorFactory.get_executor("test_service")

        # Should be clamped to cpu_count (2)
        assert executor._max_workers <= 2

    def test_shutdown_clears_executors(self) -> None:
        """Test that shutdown clears all executors."""
        settings = PipelineSettings(node_number=0)
        ServiceExecutorFactory.initialize(settings)
        ServiceExecutorFactory.get_executor("service_a")
        ServiceExecutorFactory.get_executor("service_b")

        ServiceExecutorFactory.shutdown()

        assert len(ServiceExecutorFactory._executors) == 0


class TestRunCPUBound:
    """Tests for run_cpu_bound method."""

    def setup_method(self) -> None:
        """Reset factory state before each test."""
        ServiceExecutorFactory._executors.clear()
        ServiceExecutorFactory._settings = None
        ServiceExecutorFactory.initialize(PipelineSettings(node_number=0))

    def teardown_method(self) -> None:
        """Clean up executors after each test."""
        ServiceExecutorFactory.shutdown()

    @pytest.mark.asyncio
    async def test_run_cpu_bound_executes_function(self) -> None:
        """Test that run_cpu_bound executes the function."""

        def cpu_intensive(x: int, y: int) -> int:
            return x + y

        loop = asyncio.get_event_loop()
        result = await ServiceExecutorFactory.run_cpu_bound(
            loop, "test_service", cpu_intensive, 3, 4
        )

        assert result == 7

    @pytest.mark.asyncio
    async def test_run_cpu_bound_with_kwargs_via_partial(self) -> None:
        """Test run_cpu_bound with a function that uses keyword args via partial."""
        from functools import partial

        def compute(a: int, b: int, multiplier: int = 1) -> int:
            return (a + b) * multiplier

        loop = asyncio.get_event_loop()
        func = partial(compute, multiplier=2)
        result = await ServiceExecutorFactory.run_cpu_bound(loop, "test_service", func, 3, 4)

        assert result == 14

    @pytest.mark.asyncio
    async def test_run_cpu_bound_exception_propagation(self) -> None:
        """Test that exceptions are propagated from run_cpu_bound."""

        def failing_function() -> None:
            raise ValueError("Intentional error")

        loop = asyncio.get_event_loop()

        with pytest.raises(ValueError, match="Intentional error"):
            await ServiceExecutorFactory.run_cpu_bound(loop, "test_service", failing_function)

    @pytest.mark.asyncio
    async def test_run_cpu_bound_concurrent_execution(self) -> None:
        """Test that multiple run_cpu_bound calls can execute concurrently."""
        import time

        def slow_function(x: int) -> int:
            time.sleep(0.1)
            return x * 2

        loop = asyncio.get_event_loop()

        start = time.time()
        results = await asyncio.gather(
            ServiceExecutorFactory.run_cpu_bound(loop, "test_service", slow_function, 1),
            ServiceExecutorFactory.run_cpu_bound(loop, "test_service", slow_function, 2),
            ServiceExecutorFactory.run_cpu_bound(loop, "test_service", slow_function, 3),
        )
        elapsed = time.time() - start

        assert list(results) == [2, 4, 6]
        # Should complete faster than serial execution (0.3s)
        assert elapsed < 0.3

    @pytest.mark.asyncio
    async def test_run_cpu_bound_creates_executor_on_demand(self) -> None:
        """Test that run_cpu_bound creates executor if not exists."""

        def simple_func() -> str:
            return "done"

        # Clear executors to ensure fresh state
        ServiceExecutorFactory._executors.clear()

        loop = asyncio.get_event_loop()
        result = await ServiceExecutorFactory.run_cpu_bound(loop, "new_service", simple_func)

        assert result == "done"
        assert "new_service" in ServiceExecutorFactory._executors


class TestExecutorThreadNaming:
    """Tests for executor thread naming."""

    def setup_method(self) -> None:
        """Reset factory state before each test."""
        ServiceExecutorFactory._executors.clear()
        ServiceExecutorFactory._settings = None

    def teardown_method(self) -> None:
        """Clean up executors after each test."""
        ServiceExecutorFactory.shutdown()

    def test_thread_name_prefix(self) -> None:
        """Test that executor threads have correct name prefix."""
        settings = PipelineSettings(node_number=0)
        ServiceExecutorFactory.initialize(settings)

        executor = ServiceExecutorFactory.get_executor("my_service")

        assert executor._thread_name_prefix == "my_service-worker"
