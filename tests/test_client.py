"""Tests for scripts/client.py"""

import threading
import time
from unittest.mock import Mock, patch

import pytest
import requests

from scripts.client import send_request_async


@pytest.fixture
def mock_response_success() -> Mock:
    """Create a mock successful response."""
    response = Mock()
    response.status_code = 200
    response.json.return_value = {
        "generated_response": "This is a test response with helpful information.",
        "sentiment": "positive",
        "is_toxic": False,
        "request_id": "test_123",
    }
    return response


@pytest.fixture
def mock_response_error() -> Mock:
    """Create a mock error response."""
    response = Mock()
    response.status_code = 500
    response.text = "Internal Server Error"
    return response


def test_send_request_async_success(mock_response_success: Mock) -> None:
    """Test successful request handling."""
    with (
        patch("scripts.client.requests.post", return_value=mock_response_success),
        patch("scripts.client.results", {}),
        patch("scripts.client.results_lock", threading.Lock()),
        patch("builtins.print"),
    ):
        from scripts.client import results

        send_request_async("test_req_1", "Test query?", time.time())

        # Check that result was stored
        assert "test_req_1" in results
        assert results["test_req_1"]["success"] is True
        assert "result" in results["test_req_1"]
        assert "elapsed_time" in results["test_req_1"]
        assert results["test_req_1"]["result"]["sentiment"] == "positive"


def test_send_request_async_http_error(mock_response_error: Mock) -> None:
    """Test handling of HTTP error responses."""
    with (
        patch("scripts.client.requests.post", return_value=mock_response_error),
        patch("scripts.client.results", {}),
        patch("scripts.client.results_lock", threading.Lock()),
        patch("builtins.print"),
    ):
        from scripts.client import results

        send_request_async("test_req_2", "Test query?", time.time())

        # Check that error was stored
        assert "test_req_2" in results
        assert results["test_req_2"]["success"] is False
        assert "HTTP 500" in results["test_req_2"]["error"]


def test_send_request_async_timeout() -> None:
    """Test handling of request timeout."""
    with (
        patch("scripts.client.requests.post", side_effect=requests.exceptions.Timeout),
        patch("scripts.client.results", {}),
        patch("scripts.client.results_lock", threading.Lock()),
        patch("builtins.print"),
    ):
        from scripts.client import results

        send_request_async("test_req_3", "Test query?", time.time())

        # Check that timeout error was stored
        assert "test_req_3" in results
        assert results["test_req_3"]["success"] is False
        assert "Timeout" in results["test_req_3"]["error"]


def test_send_request_async_connection_error() -> None:
    """Test handling of connection errors."""
    with (
        patch("scripts.client.requests.post", side_effect=requests.exceptions.ConnectionError),
        patch("scripts.client.results", {}),
        patch("scripts.client.results_lock", threading.Lock()),
        patch("builtins.print"),
    ):
        from scripts.client import results

        send_request_async("test_req_4", "Test query?", time.time())

        # Check that connection error was stored
        assert "test_req_4" in results
        assert results["test_req_4"]["success"] is False
        assert "Connection error" in results["test_req_4"]["error"]


def test_send_request_async_generic_exception() -> None:
    """Test handling of generic exceptions."""
    with (
        patch("scripts.client.requests.post", side_effect=ValueError("Test error")),
        patch("scripts.client.results", {}),
        patch("scripts.client.results_lock", threading.Lock()),
        patch("builtins.print"),
    ):
        from scripts.client import results

        send_request_async("test_req_5", "Test query?", time.time())

        # Check that generic error was stored
        assert "test_req_5" in results
        assert results["test_req_5"]["success"] is False
        assert "Test error" in results["test_req_5"]["error"]


def test_send_request_async_payload_format(mock_response_success: Mock) -> None:
    """Test that request payload is formatted correctly."""
    with (
        patch("scripts.client.requests.post", return_value=mock_response_success) as mock_post,
        patch("scripts.client.results", {}),
        patch("scripts.client.results_lock", threading.Lock()),
        patch("builtins.print"),
    ):
        request_id = "test_req_6"
        query = "What is the refund policy?"

        send_request_async(request_id, query, time.time())

        # Check that post was called with correct payload
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert "json" in call_kwargs
        assert call_kwargs["json"]["request_id"] == request_id
        assert call_kwargs["json"]["query"] == query
        assert call_kwargs["timeout"] == 300


def test_send_request_async_thread_safety() -> None:
    """Test that multiple concurrent requests are thread-safe."""
    with (
        patch("scripts.client.requests.post") as mock_post,
        patch("scripts.client.results", {}),
        patch("scripts.client.results_lock", threading.Lock()),
        patch("builtins.print"),
    ):
        from scripts.client import results

        # Create mock response
        response = Mock()
        response.status_code = 200
        response.json.return_value = {
            "generated_response": "Test",
            "sentiment": "positive",
            "is_toxic": False,
        }
        mock_post.return_value = response

        # Send multiple requests concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(
                target=send_request_async, args=(f"req_{i}", f"Query {i}", time.time())
            )
            thread.start()
            threads.append(thread)

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Check that all results were stored
        assert len(results) == 5
        for i in range(5):
            assert f"req_{i}" in results


def test_server_url_from_environment() -> None:
    """Test that SERVER_URL uses NODE_0_IP environment variable."""
    with patch.dict("os.environ", {"NODE_0_IP": "192.168.1.100:9000"}):
        # Re-import to pick up new environment variable
        import importlib

        import scripts.client

        importlib.reload(scripts.client)

        from scripts.client import SERVER_URL

        assert "192.168.1.100:9000" in SERVER_URL
        assert SERVER_URL.startswith("http://")
        assert SERVER_URL.endswith("/query")


def test_send_request_async_stores_elapsed_time(mock_response_success: Mock) -> None:
    """Test that elapsed time is calculated and stored."""
    with (
        patch("scripts.client.requests.post", return_value=mock_response_success),
        patch("scripts.client.results", {}),
        patch("scripts.client.results_lock", threading.Lock()),
        patch("scripts.client.time.time", side_effect=[100.0, 100.5]),  # Mock time progression
        patch("builtins.print"),
    ):
        from scripts.client import results

        send_request_async("test_req_8", "Test query?", 100.0)

        # Check that elapsed_time was calculated
        assert "test_req_8" in results
        assert "elapsed_time" in results["test_req_8"]
        assert results["test_req_8"]["elapsed_time"] >= 0
