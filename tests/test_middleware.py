"""
Tests for the middleware module.

Tests cover:
- CompressionMiddleware request decompression (zstd, lz4)
- CompressionMiddleware response compression (zstd, lz4)
- Pass-through for non-HTTP requests
- Pass-through when no compression needed
"""

from collections.abc import Awaitable, Callable, MutableMapping
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import lz4.frame  # type: ignore[import-untyped]
import pytest
import zstandard as zstd

from pipeline.middleware import CompressionMiddleware


# Type aliases for ASGI
Scope = MutableMapping[str, Any]
Receive = Callable[[], Awaitable[MutableMapping[str, Any]]]
Send = Callable[[MutableMapping[str, Any]], Awaitable[None]]
ASGIApp = Callable[[Scope, Receive, Send], Awaitable[None]]


class TestCompressionMiddlewareBasic:
    """Basic tests for CompressionMiddleware."""

    @pytest.fixture
    def mock_app(self) -> AsyncMock:
        """Create a mock ASGI app."""
        return AsyncMock()

    @pytest.fixture
    def middleware(self, mock_app: AsyncMock) -> CompressionMiddleware:
        """Create middleware instance with mock app."""
        return CompressionMiddleware(mock_app)

    @pytest.mark.asyncio
    async def test_non_http_passthrough(
        self, middleware: CompressionMiddleware, mock_app: AsyncMock
    ) -> None:
        """Test that non-HTTP requests are passed through."""
        scope = {"type": "websocket"}
        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)

        mock_app.assert_awaited_once_with(scope, receive, send)

    @pytest.mark.asyncio
    async def test_http_no_compression_passthrough(
        self, middleware: CompressionMiddleware, mock_app: AsyncMock
    ) -> None:
        """Test HTTP request without compression headers passes through."""
        scope = {
            "type": "http",
            "path": "/test",
            "headers": [],
        }
        receive = AsyncMock()
        send = AsyncMock()

        await middleware(scope, receive, send)

        mock_app.assert_awaited_once()


class TestRequestDecompression:
    """Tests for request body decompression."""

    @pytest.fixture
    def mock_app(self) -> AsyncMock:
        """Create a mock ASGI app that captures the decompressed body."""
        return AsyncMock()

    @pytest.fixture
    def middleware(self, mock_app: AsyncMock) -> CompressionMiddleware:
        """Create middleware instance."""
        return CompressionMiddleware(mock_app)

    @pytest.mark.asyncio
    @patch("pipeline.middleware.settings")
    @patch("pipeline.middleware.metrics")
    async def test_zstd_decompression(
        self,
        mock_metrics: MagicMock,
        mock_settings: MagicMock,
        middleware: CompressionMiddleware,
        mock_app: AsyncMock,
    ) -> None:
        """Test that zstd-compressed request body is decompressed."""
        # Setup mock settings
        mock_settings.profiling_run_id = "test"
        mock_settings.node_number = 0

        # Setup mock metrics
        mock_metrics.compression_ratio_histogram.labels.return_value.observe = MagicMock()
        mock_metrics.compressed_bytes_counter.labels.return_value.inc = MagicMock()

        # Create compressed body
        original_body = b'{"test": "data"}'
        cctx = zstd.ZstdCompressor()
        compressed_body = cctx.compress(original_body)

        # Mock request
        body_returned = [False]

        async def mock_receive() -> dict:
            if not body_returned[0]:
                body_returned[0] = True
                return {"type": "http.request", "body": compressed_body, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {
            "type": "http",
            "path": "/test",
            "headers": [
                (b"content-encoding", b"zstd"),
                (b"content-length", str(len(compressed_body)).encode()),
            ],
        }
        send = AsyncMock()

        await middleware(scope, mock_receive, send)

        # Verify app was called
        mock_app.assert_awaited_once()

        # Check that headers were updated (content-encoding removed)
        call_scope = mock_app.call_args[0][0]
        header_names = [h[0].lower() for h in call_scope["headers"]]
        assert b"content-encoding" not in header_names

    @pytest.mark.asyncio
    @patch("pipeline.middleware.settings")
    @patch("pipeline.middleware.metrics")
    async def test_lz4_decompression(
        self,
        mock_metrics: MagicMock,
        mock_settings: MagicMock,
        middleware: CompressionMiddleware,
        mock_app: AsyncMock,
    ) -> None:
        """Test that lz4-compressed request body is decompressed."""
        # Setup mock settings
        mock_settings.profiling_run_id = "test"
        mock_settings.node_number = 0

        # Setup mock metrics
        mock_metrics.compression_ratio_histogram.labels.return_value.observe = MagicMock()
        mock_metrics.compressed_bytes_counter.labels.return_value.inc = MagicMock()

        # Create compressed body
        original_body = b'{"test": "data"}'
        compressed_body = lz4.frame.compress(original_body)

        # Mock request
        body_returned = [False]

        async def mock_receive() -> dict:
            if not body_returned[0]:
                body_returned[0] = True
                return {"type": "http.request", "body": compressed_body, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        scope = {
            "type": "http",
            "path": "/test",
            "headers": [
                (b"content-encoding", b"lz4"),
                (b"content-length", str(len(compressed_body)).encode()),
            ],
        }
        send = AsyncMock()

        await middleware(scope, mock_receive, send)

        # Verify app was called
        mock_app.assert_awaited_once()


class TestResponseCompression:
    """Tests for response body compression."""

    @pytest.mark.asyncio
    @patch("pipeline.middleware.settings")
    @patch("pipeline.middleware.metrics")
    async def test_zstd_response_compression(
        self, mock_metrics: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test that response is compressed with zstd when requested."""
        # Setup mock settings
        mock_settings.profiling_run_id = "test"
        mock_settings.node_number = 0

        # Setup mock metrics
        mock_metrics.compression_ratio_histogram.labels.return_value.observe = MagicMock()
        mock_metrics.compressed_bytes_counter.labels.return_value.inc = MagicMock()

        # Track what was sent
        sent_messages: list[dict[str, Any]] = []

        async def capture_send(message: MutableMapping[str, Any]) -> None:
            sent_messages.append(dict(message))

        # Create mock app that sends a response
        async def mock_app(_scope: Scope, _receive: Receive, send: Send) -> None:
            await send(
                {"type": "http.response.start", "headers": [(b"content-type", b"application/json")]}
            )
            await send({"type": "http.response.body", "body": b'{"result": "data"}'})

        middleware = CompressionMiddleware(mock_app)

        scope: Scope = {
            "type": "http",
            "path": "/test",
            "headers": [(b"accept-encoding", b"zstd, gzip")],
        }
        receive = AsyncMock()

        await middleware(scope, receive, capture_send)

        # Check response headers
        start_message = next(m for m in sent_messages if m["type"] == "http.response.start")
        header_dict = dict(start_message["headers"])
        assert b"content-encoding" in header_dict
        assert header_dict[b"content-encoding"] == b"zstd"

        # Check body is compressed
        body_message = next(m for m in sent_messages if m["type"] == "http.response.body")
        compressed_body = body_message["body"]

        # Verify it's valid zstd
        dctx = zstd.ZstdDecompressor()
        decompressed = dctx.decompress(compressed_body)
        assert decompressed == b'{"result": "data"}'

    @pytest.mark.asyncio
    @patch("pipeline.middleware.settings")
    @patch("pipeline.middleware.metrics")
    async def test_lz4_response_compression(
        self, mock_metrics: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test that response is compressed with lz4 when requested."""
        # Setup mock settings
        mock_settings.profiling_run_id = "test"
        mock_settings.node_number = 0

        # Setup mock metrics
        mock_metrics.compression_ratio_histogram.labels.return_value.observe = MagicMock()
        mock_metrics.compressed_bytes_counter.labels.return_value.inc = MagicMock()

        # Track what was sent
        sent_messages: list[dict[str, Any]] = []

        async def capture_send(message: MutableMapping[str, Any]) -> None:
            sent_messages.append(dict(message))

        # Create mock app that sends a response
        async def mock_app(_scope: Scope, _receive: Receive, send: Send) -> None:
            await send(
                {"type": "http.response.start", "headers": [(b"content-type", b"application/json")]}
            )
            await send({"type": "http.response.body", "body": b'{"result": "data"}'})

        middleware = CompressionMiddleware(mock_app)

        scope: Scope = {
            "type": "http",
            "path": "/test",
            "headers": [(b"accept-encoding", b"lz4, gzip")],
        }
        receive = AsyncMock()

        await middleware(scope, receive, capture_send)

        # Check response headers
        start_message = next(m for m in sent_messages if m["type"] == "http.response.start")
        header_dict = dict(start_message["headers"])
        assert b"content-encoding" in header_dict
        assert header_dict[b"content-encoding"] == b"lz4"

        # Check body is compressed
        body_message = next(m for m in sent_messages if m["type"] == "http.response.body")
        compressed_body = body_message["body"]

        # Verify it's valid lz4
        decompressed = lz4.frame.decompress(compressed_body)
        assert decompressed == b'{"result": "data"}'

    @pytest.mark.asyncio
    async def test_no_compression_when_not_accepted(self) -> None:
        """Test that response is not compressed when not accepted."""
        # Track what was sent
        sent_messages: list[dict[str, Any]] = []

        async def capture_send(message: MutableMapping[str, Any]) -> None:
            sent_messages.append(dict(message))

        # Create mock app that sends a response
        async def mock_app(_scope: Scope, _receive: Receive, send: Send) -> None:
            await send(
                {"type": "http.response.start", "headers": [(b"content-type", b"application/json")]}
            )
            await send({"type": "http.response.body", "body": b'{"result": "data"}'})

        middleware = CompressionMiddleware(mock_app)

        scope: Scope = {
            "type": "http",
            "path": "/test",
            "headers": [(b"accept-encoding", b"gzip")],  # Only gzip, not zstd or lz4
        }
        receive = AsyncMock()

        await middleware(scope, receive, capture_send)

        # Check that body was not compressed (should be original)
        body_message = next(m for m in sent_messages if m["type"] == "http.response.body")
        assert body_message["body"] == b'{"result": "data"}'

    @pytest.mark.asyncio
    @patch("pipeline.middleware.settings")
    @patch("pipeline.middleware.metrics")
    async def test_zstd_preferred_over_lz4(
        self, mock_metrics: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test that zstd is preferred when both are accepted."""
        # Setup mock settings
        mock_settings.profiling_run_id = "test"
        mock_settings.node_number = 0

        # Setup mock metrics
        mock_metrics.compression_ratio_histogram.labels.return_value.observe = MagicMock()
        mock_metrics.compressed_bytes_counter.labels.return_value.inc = MagicMock()

        # Track what was sent
        sent_messages: list[dict[str, Any]] = []

        async def capture_send(message: MutableMapping[str, Any]) -> None:
            sent_messages.append(dict(message))

        # Create mock app
        async def mock_app(_scope: Scope, _receive: Receive, send: Send) -> None:
            await send({"type": "http.response.start", "headers": []})
            await send({"type": "http.response.body", "body": b"test"})

        middleware = CompressionMiddleware(mock_app)

        scope: Scope = {
            "type": "http",
            "path": "/test",
            "headers": [(b"accept-encoding", b"lz4, zstd, gzip")],
        }
        receive = AsyncMock()

        await middleware(scope, receive, capture_send)

        # Check zstd was chosen
        start_message = next(m for m in sent_messages if m["type"] == "http.response.start")
        header_dict = dict(start_message["headers"])
        assert header_dict[b"content-encoding"] == b"zstd"


class TestCompressionMetrics:
    """Tests for compression metrics recording."""

    @pytest.mark.asyncio
    @patch("pipeline.middleware.settings")
    @patch("pipeline.middleware.metrics")
    async def test_inbound_metrics_recorded(
        self, mock_metrics: MagicMock, mock_settings: MagicMock
    ) -> None:
        """Test that inbound compression metrics are recorded."""
        # Setup
        mock_settings.profiling_run_id = "test_run"
        mock_settings.node_number = 1

        mock_observe = MagicMock()
        mock_inc = MagicMock()
        mock_metrics.compression_ratio_histogram.labels.return_value.observe = mock_observe
        mock_metrics.compressed_bytes_counter.labels.return_value.inc = mock_inc

        # Create compressed body
        original_body = b'{"test": "data"}'
        cctx = zstd.ZstdCompressor()
        compressed_body = cctx.compress(original_body)

        body_returned = [False]

        async def mock_receive() -> dict:
            if not body_returned[0]:
                body_returned[0] = True
                return {"type": "http.request", "body": compressed_body, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        mock_app = AsyncMock()
        middleware = CompressionMiddleware(mock_app)

        scope = {
            "type": "http",
            "path": "/test",
            "headers": [
                (b"content-encoding", b"zstd"),
                (b"content-length", str(len(compressed_body)).encode()),
            ],
        }
        send = AsyncMock()

        await middleware(scope, mock_receive, send)

        # Verify metrics were called for inbound
        mock_metrics.compression_ratio_histogram.labels.assert_called()
        mock_observe.assert_called()
