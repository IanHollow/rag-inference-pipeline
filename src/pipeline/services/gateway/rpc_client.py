"""
RPC client for making requests to downstream services (Node 1 and Node 2).

Uses httpx.AsyncClient with Tenacity retry policies for robust communication.
"""

import logging
from typing import Any

import httpx
from tenacity import (
    RetryError,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


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
    """

    def __init__(
        self,
        base_url: str,
        timeout_seconds: float = 30.0,
        max_retries: int = 3,
    ) -> None:
        """
        Initialize RPC client.

        Args:
            base_url: Base URL for the service
            timeout_seconds: Request timeout in seconds
            max_retries: Maximum number of retry attempts for 5xx errors
        """
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

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
            "RPCClient initialized: base_url=%s, timeout=%.1fs, max_retries=%d",
            self.base_url,
            self.timeout_seconds,
            self.max_retries,
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
            logger.debug("RPC POST %s with payload keys: %s", url, list(payload.keys()))

            response = await self._client.post(
                endpoint,
                json=payload,
            )

            # Raise for 4xx/5xx status codes
            response.raise_for_status()

            result = response.json()
            logger.debug("RPC POST %s succeeded", url)
            return result

        except httpx.TimeoutException as e:
            logger.error("RPC timeout for %s after %.1fs", url, self.timeout_seconds)
            raise RPCTimeoutError(f"Request to {url} timed out") from e

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code

            # 5xx errors: service error
            if 500 <= status_code < 600:
                logger.warning("RPC service error: %s returned %d", url, status_code, exc_info=True)
                raise RPCServiceError(f"Service error {status_code} from {url}") from e

            # 4xx errors: client error (don't retry)
            logger.error("RPC client error: %s returned %d", url, status_code)
            raise RPCError(f"Client error {status_code} from {url}") from e

        except httpx.ConnectError as e:
            logger.exception("RPC connection error for %s", url)
            raise RPCError(f"Failed to connect to {url}") from e

        except RetryError as e:
            logger.exception("RPC exhausted retries for %s", url)
            raise RPCError(f"Exhausted retries for {url}") from e

        except Exception as e:
            logger.exception("Unexpected RPC error for %s", url)
            raise RPCError(f"Unexpected error calling {url}: {e}") from e

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
            return result

        except httpx.TimeoutException as e:
            logger.error("RPC timeout for %s", url)
            raise RPCTimeoutError(f"Request to {url} timed out") from e

        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            logger.error("RPC error: %s returned %d", url, status_code)
            raise RPCError(f"HTTP {status_code} from {url}") from e

        except Exception as e:
            logger.exception("Unexpected RPC error for %s", url)
            raise RPCError(f"Unexpected error calling {url}: {e}") from e
