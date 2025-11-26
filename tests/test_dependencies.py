"""
Tests for the dependencies module.

Tests cover:
- get_registry function
- get_component function
- Error handling when registry is not initialized
"""

from unittest.mock import MagicMock

from fastapi import HTTPException
import pytest

from pipeline.component_registry import ComponentRegistry
from pipeline.dependencies import get_component, get_registry


class TestGetRegistry:
    """Tests for get_registry function."""

    def test_get_registry_success(self) -> None:
        """Test getting registry from request with initialized registry."""
        # Create mock request
        mock_request = MagicMock()
        mock_registry = ComponentRegistry()
        mock_request.app.state.registry = mock_registry

        result = get_registry(mock_request)

        assert result == mock_registry

    def test_get_registry_not_initialized(self) -> None:
        """Test that missing registry raises HTTPException."""
        mock_request = MagicMock()
        mock_request.app.state.registry = None

        with pytest.raises(HTTPException) as exc_info:
            get_registry(mock_request)

        assert exc_info.value.status_code == 500
        assert "not initialized" in exc_info.value.detail

    def test_get_registry_missing_attribute(self) -> None:
        """Test that missing registry attribute raises HTTPException."""
        mock_request = MagicMock()
        # Simulate missing registry attribute
        del mock_request.app.state.registry

        with pytest.raises(HTTPException) as exc_info:
            get_registry(mock_request)

        assert exc_info.value.status_code == 500


class TestGetComponent:
    """Tests for get_component function."""

    def test_get_component_success(self) -> None:
        """Test getting a component by name."""
        # Setup
        mock_registry = ComponentRegistry()
        mock_component = MagicMock()
        mock_registry.register("my_component", mock_component)

        mock_request = MagicMock()
        mock_request.app.state.registry = mock_registry
        mock_request.app.state.component_aliases = {}

        # Execute
        getter = get_component("my_component")
        result = getter(mock_request)

        assert result == mock_component

    def test_get_component_with_alias(self) -> None:
        """Test getting a component using an alias."""
        # Setup
        mock_registry = ComponentRegistry()
        mock_component = MagicMock()
        mock_registry.register("actual_name", mock_component)

        mock_request = MagicMock()
        mock_request.app.state.registry = mock_registry
        mock_request.app.state.component_aliases = {"alias_name": "actual_name"}

        # Execute
        getter = get_component("alias_name")
        result = getter(mock_request)

        assert result == mock_component

    def test_get_component_not_found(self) -> None:
        """Test that getting nonexistent component returns None."""
        # Setup
        mock_registry = ComponentRegistry()
        mock_request = MagicMock()
        mock_request.app.state.registry = mock_registry
        mock_request.app.state.component_aliases = {}

        # Execute
        getter = get_component("nonexistent")
        result = getter(mock_request)

        assert result is None

    def test_get_component_alias_not_in_aliases(self) -> None:
        """Test getting component when name is not in aliases dict."""
        # Setup
        mock_registry = ComponentRegistry()
        mock_component = MagicMock()
        mock_registry.register("direct_name", mock_component)

        mock_request = MagicMock()
        mock_request.app.state.registry = mock_registry
        mock_request.app.state.component_aliases = {"other_alias": "other_component"}

        # Execute - should fall back to direct name lookup
        getter = get_component("direct_name")
        result = getter(mock_request)

        assert result == mock_component

    def test_get_component_returns_callable(self) -> None:
        """Test that get_component returns a callable dependency."""
        getter = get_component("any_name")

        assert callable(getter)

    def test_get_component_with_empty_aliases(self) -> None:
        """Test getting component when aliases dict is empty."""
        # Setup
        mock_registry = ComponentRegistry()
        mock_component = MagicMock()
        mock_registry.register("component", mock_component)

        mock_request = MagicMock()
        mock_request.app.state.registry = mock_registry
        mock_request.app.state.component_aliases = {}

        # Execute
        getter = get_component("component")
        result = getter(mock_request)

        assert result == mock_component

    def test_get_component_multiple_calls_same_name(self) -> None:
        """Test that multiple calls with same name return consistent results."""
        # Setup
        mock_registry = ComponentRegistry()
        mock_component = MagicMock()
        mock_registry.register("shared", mock_component)

        mock_request = MagicMock()
        mock_request.app.state.registry = mock_registry
        mock_request.app.state.component_aliases = {}

        # Execute
        getter1 = get_component("shared")
        getter2 = get_component("shared")

        result1 = getter1(mock_request)
        result2 = getter2(mock_request)

        assert result1 == result2 == mock_component
