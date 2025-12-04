import logging

import lz4.frame  # type: ignore[import-untyped]
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send
import zstandard as zstd

from .config import get_settings
from .telemetry import metrics


logger = logging.getLogger(__name__)
settings = get_settings()

# Pre-create compressor/decompressor contexts for reuse (thread-safe)
_ZSTD_DECOMPRESSOR = zstd.ZstdDecompressor()
_ZSTD_COMPRESSOR = zstd.ZstdCompressor(level=3)


def _record_compression_metrics(
    original_size: int,
    compressed_size: int,
    direction: str,
    algorithm: str,
) -> None:
    """Record compression metrics for monitoring."""
    ratio = original_size / compressed_size if compressed_size > 0 else 1.0

    metrics.compression_ratio_histogram.labels(
        run_id=settings.profiling_run_id,
        node=str(settings.node_number),
        direction=direction,
        algorithm=algorithm,
    ).observe(ratio)

    metrics.compressed_bytes_counter.labels(
        run_id=settings.profiling_run_id,
        node=str(settings.node_number),
        direction=direction,
        algorithm=algorithm,
        type="original",
    ).inc(original_size)

    metrics.compressed_bytes_counter.labels(
        run_id=settings.profiling_run_id,
        node=str(settings.node_number),
        direction=direction,
        algorithm=algorithm,
        type="compressed",
    ).inc(compressed_size)


def _decompress_body(body: bytes, encoding: str) -> bytes:
    """Decompress request body based on encoding."""
    if encoding == "zstd":
        return _ZSTD_DECOMPRESSOR.decompress(body)
    if encoding == "lz4":
        result: bytes = lz4.frame.decompress(body)
        return result
    return body


def _compress_body(body: bytes, encoding: str) -> bytes:
    """Compress response body based on encoding."""
    if encoding == "zstd":
        return _ZSTD_COMPRESSOR.compress(body)
    if encoding == "lz4":
        result: bytes = lz4.frame.compress(body, compression_level=3)
        return result
    return body


class CompressionMiddleware:
    """
    Middleware to handle request decompression (zstd, lz4).
    Optimized for low-latency by reusing compressor/decompressor contexts.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def _handle_decompressed_request(
        self,
        scope: Scope,
        decompressed_body: bytes,
        content_encoding: str,
        original_size: int,
    ) -> Receive:
        """Prepare a new receive function with the decompressed body."""
        # Record metrics
        _record_compression_metrics(
            len(decompressed_body), original_size, "inbound", content_encoding
        )

        # Replace the body in a new receive channel
        body_container = [decompressed_body]

        async def new_receive() -> dict:
            if body_container:
                chunk = body_container.pop(0)
                return {"type": "http.request", "body": chunk, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        # Update headers: remove Content-Encoding and update Content-Length
        new_headers = [
            (k, v)
            for k, v in scope["headers"]
            if k.lower() not in (b"content-encoding", b"content-length")
        ]
        new_headers.append((b"content-length", str(len(decompressed_body)).encode("latin-1")))
        scope["headers"] = new_headers

        return new_receive

    def _create_response_send_wrapper(self, send: Send, response_encoding: str) -> Send:
        """Create a send wrapper that compresses response bodies."""

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    h for h in message.get("headers", []) if h[0].lower() != b"content-length"
                ]
                headers.append((b"content-encoding", response_encoding.encode("latin-1")))
                headers.append((b"vary", b"Accept-Encoding"))
                message["headers"] = headers
                await send(message)
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                if body:
                    compressed_body = _compress_body(body, response_encoding)
                    _record_compression_metrics(
                        len(body), len(compressed_body), "outbound", response_encoding
                    )
                    message["body"] = compressed_body
                await send(message)

        return send_wrapper

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = dict(scope.get("headers", []))
        content_encoding = headers.get(b"content-encoding", b"").decode("latin-1")

        # Handle request decompression
        if content_encoding in ("zstd", "lz4"):
            request = Request(scope, receive)
            body = await request.body()
            decompressed_body = _decompress_body(body, content_encoding)
            receive = await self._handle_decompressed_request(
                scope, decompressed_body, content_encoding, len(body)
            )

        # Determine response compression
        accept_encoding = headers.get(b"accept-encoding", b"").decode("latin-1")

        if scope["path"] == "/retrieve":
            logger.info("Middleware received Accept-Encoding: %s", accept_encoding)

        response_encoding = None
        if "zstd" in accept_encoding:
            response_encoding = "zstd"
        elif "lz4" in accept_encoding:
            response_encoding = "lz4"

        if response_encoding:
            send = self._create_response_send_wrapper(send, response_encoding)

        await self.app(scope, receive, send)
