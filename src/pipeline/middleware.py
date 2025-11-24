import logging

import lz4.frame  # type: ignore
from starlette.requests import Request
from starlette.types import ASGIApp, Message, Receive, Scope, Send
import zstandard as zstd

from .config import get_settings
from .telemetry import metrics

logger = logging.getLogger(__name__)
settings = get_settings()


class CompressionMiddleware:
    """
    Middleware to handle request decompression (zstd, lz4).
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # --- Request Decompression ---
        headers = dict(scope.get("headers", []))
        content_encoding = headers.get(b"content-encoding", b"").decode("latin-1")

        if content_encoding in ("zstd", "lz4"):
            # Read the entire body
            request = Request(scope, receive)
            body = await request.body()

            if content_encoding == "zstd":
                dctx = zstd.ZstdDecompressor()
                decompressed_body = dctx.decompress(body)
            elif content_encoding == "lz4":
                decompressed_body = lz4.frame.decompress(body)
            else:
                # Should not happen due to if check
                decompressed_body = body

            # Record metrics
            compressed_size = len(body)
            original_size = len(decompressed_body)
            ratio = original_size / compressed_size if compressed_size > 0 else 1.0

            metrics.compression_ratio_histogram.labels(
                run_id=settings.profiling_run_id,
                node=str(settings.node_number),
                direction="inbound",
                algorithm=content_encoding,
            ).observe(ratio)

            metrics.compressed_bytes_counter.labels(
                run_id=settings.profiling_run_id,
                node=str(settings.node_number),
                direction="inbound",
                algorithm=content_encoding,
                type="original",
            ).inc(original_size)

            metrics.compressed_bytes_counter.labels(
                run_id=settings.profiling_run_id,
                node=str(settings.node_number),
                direction="inbound",
                algorithm=content_encoding,
                type="compressed",
            ).inc(compressed_size)

            # Replace the body in a new receive channel
            # We use a list to hold the body so we can pop it (simulating a stream that ends)
            body_container = [decompressed_body]

            async def new_receive() -> dict:
                if body_container:
                    chunk = body_container.pop(0)
                    return {"type": "http.request", "body": chunk, "more_body": False}
                return {"type": "http.request", "body": b"", "more_body": False}

            # Remove Content-Encoding header and update Content-Length
            new_headers = []
            for k, v in scope["headers"]:
                if k.lower() == b"content-encoding":
                    continue
                if k.lower() == b"content-length":
                    continue
                new_headers.append((k, v))

            # Add new Content-Length
            new_headers.append((b"content-length", str(len(decompressed_body)).encode("latin-1")))

            # Update scope
            scope["headers"] = new_headers
            receive = new_receive

        # --- Response Compression ---
        accept_encoding = headers.get(b"accept-encoding", b"").decode("latin-1")

        if scope["path"] == "/retrieve":
            logger.info("Middleware received Accept-Encoding: %s", accept_encoding)

        # Determine compression algorithm
        response_encoding = None
        if "zstd" in accept_encoding:
            response_encoding = "zstd"
        elif "lz4" in accept_encoding:
            response_encoding = "lz4"

        if not response_encoding:
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                # Remove Content-Length as it will change
                headers = [h for h in headers if h[0].lower() != b"content-length"]
                # Add Content-Encoding
                headers.append((b"content-encoding", response_encoding.encode("latin-1")))
                # Add Vary header
                headers.append((b"vary", b"Accept-Encoding"))
                message["headers"] = headers
                await send(message)
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")

                if body:
                    if response_encoding == "zstd":
                        cctx = zstd.ZstdCompressor(level=3)
                        compressed_body = cctx.compress(body)
                    elif response_encoding == "lz4":
                        compressed_body = lz4.frame.compress(body, compression_level=3)
                    else:
                        compressed_body = body

                    # Record metrics (approximate for streaming)
                    original_size = len(body)
                    compressed_size = len(compressed_body)
                    ratio = original_size / compressed_size if compressed_size > 0 else 1.0

                    metrics.compression_ratio_histogram.labels(
                        run_id=settings.profiling_run_id,
                        node=str(settings.node_number),
                        direction="outbound",
                        algorithm=response_encoding,
                    ).observe(ratio)

                    metrics.compressed_bytes_counter.labels(
                        run_id=settings.profiling_run_id,
                        node=str(settings.node_number),
                        direction="outbound",
                        algorithm=response_encoding,
                        type="original",
                    ).inc(original_size)

                    metrics.compressed_bytes_counter.labels(
                        run_id=settings.profiling_run_id,
                        node=str(settings.node_number),
                        direction="outbound",
                        algorithm=response_encoding,
                        type="compressed",
                    ).inc(compressed_size)

                    message["body"] = compressed_body

                await send(message)

        await self.app(scope, receive, send_wrapper)
