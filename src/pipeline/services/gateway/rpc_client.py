"""
RPC client for making requests to downstream services (Node 1 and Node 2).

Uses httpx.AsyncClient with Tenacity retry policies for robust communication.
"""

import io
import logging
from typing import Any, cast

import httpx
import lz4.frame  # type: ignore[import-untyped]
import msgspec
import orjson
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
import zstandard as zstd

from pipeline.config import get_settings
from pipeline.telemetry import metrics


logger = logging.getLogger(__name__)
settings = get_settings()


class RPCError(Exception):
    """Base exception for RPC errors."""


class RPCTimeoutError(RPCError):
    """Raised when RPC call times out."""


class RPCServiceError(RPCError):
    """Raised when downstream service returns 5xx error."""


class RPCClient:
    """
    Async HTTP client for making RPC calls to downstream services.

    Features:
    - Automatic retries on 5xx errors
    - Timeout handling
    - Connection pooling
    - Compression (zstd, lz4)
    """

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
        compression: str = "none",
        compression_level: int = 3,
    ) -> None:
        """
        Initialize RPC client.

        Args:
            base_url: Base URL for the service
            timeout_seconds: Request timeout in seconds
            max_retries: Maximum number of retry attempts for 5xx errors
            compression: Compression algorithm ("none", "zstd", "lz4")
            compression_level: Compression level
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.compression = compression
        self.compression_level = compression_level

        # Create httpx client with connection pooling
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
            ),
            http2=True,  # Enable HTTP/2
        )

        logger.info(
            "RPCClient initialized: base_url=%s, timeout=%.1fs, max_retries=%d, compression=%s",
            self.base_url,
            self.timeout_seconds,
            self.max_retries,
            self.compression,
        )

    async def close(self) -> None:
        """Close the HTTP client and cleanup resources."""
        await self._client.aclose()
        logger.info("RPCClient closed: base_url=%s", self.base_url)

    @retry(
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.1, min=0.1, max=2.0),
        reraise=True,
    )
    async def _post_with_retry(
        self,
        endpoint: str,
        payload: dict[str, object] | object,
    ) -> dict[str, Any]:
        """
        Internal method to execute POST request with retries.
        """
        url = f"{self.base_url}{endpoint}"

        # Serialize
        # Check if payload is a msgspec Struct or dict with structs
        # We use msgspec.json.encode which handles both structs and standard types
        try:
            payload_bytes = msgspec.json.encode(payload)
        except TypeError:
            # Fallback to orjson if msgspec fails (e.g. pydantic models not converted)
            payload_bytes = orjson.dumps(payload)

        original_size = len(payload_bytes)
        headers = {
            "Content-Type": "application/json",
        }

        # Compress
        if self.compression == "zstd":
            cctx = zstd.ZstdCompressor(level=self.compression_level)
            payload_bytes = cctx.compress(payload_bytes)
            headers["Content-Encoding"] = "zstd"
            headers["Accept-Encoding"] = "zstd"
        elif self.compression == "lz4":
            payload_bytes = lz4.frame.compress(
                payload_bytes, compression_level=self.compression_level
            )
            headers["Content-Encoding"] = "lz4"
            headers["Accept-Encoding"] = "lz4"
        else:
            # Explicitly set Accept-Encoding to avoid httpx adding zstd if it's installed
            headers["Accept-Encoding"] = "gzip, deflate, br"

        if self.compression != "none":
            compressed_size = len(payload_bytes)
            ratio = original_size / compressed_size if compressed_size > 0 else 1.0

            metrics.compression_ratio_histogram.labels(
                run_id=settings.profiling_run_id,
                node=str(settings.node_number),
                direction="outbound",
                algorithm=self.compression,
            ).observe(ratio)

            metrics.compressed_bytes_counter.labels(
                run_id=settings.profiling_run_id,
                node=str(settings.node_number),
                direction="outbound",
                algorithm=self.compression,
                type="original",
            ).inc(original_size)

            metrics.compressed_bytes_counter.labels(
                run_id=settings.profiling_run_id,
                node=str(settings.node_number),
                direction="outbound",
                algorithm=self.compression,
                type="compressed",
            ).inc(compressed_size)

        response = await self._client.post(
            endpoint,
            content=payload_bytes,
            headers=headers,
        )

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            # Don't retry 4xx errors
            if 400 <= e.response.status_code < 500:
                logger.exception("RPC client error: %s returned %d", url, e.response.status_code)
                msg = f"Client error {e.response.status_code} from {url}"
                raise RPCError(msg) from e
            # Re-raise 5xx to trigger retry
            raise

        logger.debug("RPC POST %s succeeded", url)

        # Handle response decompression
        content = response.content
        encoding = response.headers.get("content-encoding", "")

        if encoding == "zstd":
            # Check for Zstandard magic number (0xFD2FB528) in little endian
            if len(content) >= 4 and content[:4] == b"\x28\xb5\x2f\xfd":
                dctx = zstd.ZstdDecompressor()
                # Use stream_reader to handle cases where content size is unknown
                with dctx.stream_reader(io.BytesIO(content)) as reader:
                    content = reader.read()
        elif encoding == "lz4":
            content = lz4.frame.decompress(content)

        # Decode response
        result: dict[str, Any] = orjson.loads(content)
        return result

    async def post(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Make a POST request to downstream service with retry logic.

        Args:
            endpoint: API endpoint path (e.g., /retrieve)
            payload: Request payload (will be JSON serialized)

        Returns:
            Response JSON as dictionary

        Raises:
            RPCTimeoutError: If request times out
            RPCServiceError: If service returns 5xx error after retries
            RPCError: For other RPC failures
        """
        url = f"{self.base_url}{endpoint}"

        try:
            return await self._post_with_retry(endpoint, payload)

        except httpx.TimeoutException as e:
            logger.exception("RPC timeout for %s after %.1fs", url, self.timeout_seconds)
            msg = f"Request to {url} timed out"
            raise RPCTimeoutError(msg) from e

        except httpx.HTTPStatusError as e:
            # This catches 5xx errors that exhausted retries
            status_code = e.response.status_code
            logger.warning("RPC service error: %s returned %d", url, status_code, exc_info=True)
            msg = f"Service error {status_code} from {url}"
            raise RPCServiceError(msg) from e

        except httpx.ConnectError as e:
            logger.exception("RPC connection error for %s", url)
            msg = f"Failed to connect to {url}"
            raise RPCError(msg) from e

        except RetryError as e:
            logger.exception("RPC exhausted retries for %s", url)
            msg = f"Exhausted retries for {url}"
            raise RPCError(msg) from e

        except Exception as e:
            logger.exception("Unexpected RPC error for %s", url)
            msg = f"Unexpected error calling {url}: {e}"
            raise RPCError(msg) from e

    async def get(self, endpoint: str) -> dict[str, Any]:
        """
        Make a GET request to downstream service.

        Args:
            endpoint: API endpoint path

        Returns:
            Response JSON as dictionary

        Raises:
            RPCError: If request fails
        """
        url = f"{self.base_url}{endpoint}"

        try:
            logger.debug("RPC GET %s", url)

            response = await self._client.get(endpoint)
            response.raise_for_status()

            result = response.json()
            logger.debug("RPC GET %s succeeded", url)

        except httpx.TimeoutException as e:
            logger.exception("RPC timeout for %s", url)
            msg = f"Request to {url} timed out"
            raise RPCTimeoutError(msg) from e

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            logger.exception("RPC error: %s returned %d", url, status_code)
            msg = f"HTTP {status_code} from {url}"
            raise RPCError(msg) from e

        except Exception as e:
            logger.exception("Unexpected RPC error for %s", url)
            msg = f"Unexpected error calling {url}: {e}"
            raise RPCError(msg) from e
        else:
            return cast("dict[str, Any]", result)

    async def clear_cache(self, endpoint: str = "/clear_cache") -> bool:
        """
        Clear the cache on the downstream service.

        Args:
            endpoint: API endpoint path (default: /clear_cache)

        Returns:
            True if successful, False otherwise
        """
        try:
            response = await self._client.post(endpoint)
            response.raise_for_status()
        except Exception as e:
            logger.warning("Failed to clear cache on %s%s: %s", self.base_url, endpoint, e)
            return False
        else:
            return True
