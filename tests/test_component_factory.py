"""
Tests for the component factory module.

Tests cover:
- Creating components of each type
- Factory function registration
- Error handling for unknown component types
- String vs enum component type resolution
"""

from unittest.mock import MagicMock, patch

import pytest

from pipeline.component_factory import (
    COMPONENT_FACTORIES,
    create_component,
    create_document_store,
    create_embedding_generator,
    create_faiss_store,
    create_gateway_orchestrator,
    create_llm_generator,
    create_reranker,
    create_sentiment_analyzer,
    create_toxicity_filter,
)
from pipeline.config import PipelineSettings
from pipeline.enums import ComponentType


class TestComponentFactories:
    """Test individual component factory functions."""

    @pytest.fixture
    def settings(self) -> PipelineSettings:
        """Create test settings."""
        return PipelineSettings(node_number=0)

    @patch("pipeline.component_factory.EmbeddingGenerator")
    def test_create_embedding_generator(
        self, mock_class: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test creating an embedding generator component."""
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        result = create_embedding_generator(settings, {})

        mock_class.assert_called_once_with(settings)
        assert result == mock_instance

    @patch("pipeline.component_factory.FAISSStore")
    def test_create_faiss_store(self, mock_class: MagicMock, settings: PipelineSettings) -> None:
        """Test creating a FAISS store component."""
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        result = create_faiss_store(settings, {})

        mock_class.assert_called_once_with(settings)
        assert result == mock_instance

    @patch("pipeline.component_factory.DocumentStore")
    def test_create_document_store(self, mock_class: MagicMock, settings: PipelineSettings) -> None:
        """Test creating a document store component."""
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        result = create_document_store(settings, {})

        mock_class.assert_called_once_with(settings)
        assert result == mock_instance

    @patch("pipeline.component_factory.Reranker")
    def test_create_reranker(self, mock_class: MagicMock, settings: PipelineSettings) -> None:
        """Test creating a reranker component."""
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        result = create_reranker(settings, {})

        mock_class.assert_called_once_with(settings)
        assert result == mock_instance

    @patch("pipeline.component_factory.LLMGenerator")
    def test_create_llm_generator(self, mock_class: MagicMock, settings: PipelineSettings) -> None:
        """Test creating an LLM generator component."""
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        result = create_llm_generator(settings, {})

        mock_class.assert_called_once_with(settings)
        assert result == mock_instance

    @patch("pipeline.component_factory.SentimentAnalyzer")
    def test_create_sentiment_analyzer(
        self, mock_class: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test creating a sentiment analyzer component."""
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        result = create_sentiment_analyzer(settings, {})

        mock_class.assert_called_once_with(settings)
        assert result == mock_instance

    @patch("pipeline.component_factory.ToxicityFilter")
    def test_create_toxicity_filter(
        self, mock_class: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test creating a toxicity filter component."""
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        result = create_toxicity_filter(settings, {})

        mock_class.assert_called_once_with(settings)
        assert result == mock_instance

    @patch("pipeline.component_factory.Orchestrator")
    def test_create_gateway_orchestrator(
        self, mock_class: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test creating a gateway orchestrator component."""
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        config = {
            "retrieval_url": "http://localhost:8001",
            "generation_url": "http://localhost:8002",
            "batch_size": 16,
            "batch_timeout": 0.05,
        }
        result = create_gateway_orchestrator(settings, config)

        mock_class.assert_called_once_with(
            retrieval_url="http://localhost:8001",
            generation_url="http://localhost:8002",
            batch_size=16,
            batch_timeout=0.05,
        )
        assert result == mock_instance

    @patch("pipeline.component_factory.Orchestrator")
    def test_create_gateway_orchestrator_default_config(
        self, mock_class: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test creating gateway orchestrator with default/missing config values."""
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        result = create_gateway_orchestrator(settings, {})

        mock_class.assert_called_once_with(
            retrieval_url=None,
            generation_url=None,
            batch_size=None,
            batch_timeout=None,
        )
        assert result == mock_instance


class TestCreateComponent:
    """Test the create_component function."""

    @pytest.fixture
    def settings(self) -> PipelineSettings:
        """Create test settings."""
        return PipelineSettings(node_number=0)

    @patch("pipeline.component_factory.EmbeddingGenerator")
    def test_create_component_with_enum(
        self, mock_class: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test creating component using ComponentType enum."""
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        result = create_component(ComponentType.EMBEDDING, settings)

        mock_class.assert_called_once_with(settings)
        assert result == mock_instance

    @patch("pipeline.component_factory.EmbeddingGenerator")
    def test_create_component_with_string_key(
        self, mock_class: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test creating component using string key that maps to factory."""
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        result = create_component("embedding_generator", settings)

        mock_class.assert_called_once_with(settings)
        assert result == mock_instance

    @patch("pipeline.component_factory.FAISSStore")
    def test_create_component_with_enum_value_string(
        self, mock_class: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test creating component using string that matches enum value."""
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        # "faiss" is ComponentType.FAISS.value
        result = create_component("faiss", settings)

        mock_class.assert_called_once_with(settings)
        assert result == mock_instance

    def test_create_component_unknown_type(self, settings: PipelineSettings) -> None:
        """Test that unknown component type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown component type"):
            create_component("nonexistent_component", settings)

    def test_create_component_with_none_config(self, settings: PipelineSettings) -> None:
        """Test that None config is handled gracefully."""
        with patch("pipeline.component_factory.EmbeddingGenerator") as mock_class:
            mock_class.return_value = MagicMock()
            result = create_component(ComponentType.EMBEDDING, settings, None)
            assert result is not None

    @patch("pipeline.component_factory.SentimentAnalyzer")
    def test_create_component_with_custom_config(
        self, mock_class: MagicMock, settings: PipelineSettings
    ) -> None:
        """Test creating component with custom config dict."""
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance

        custom_config = {"some_param": "value"}
        result = create_component(ComponentType.SENTIMENT, settings, custom_config)

        mock_class.assert_called_once_with(settings)
        assert result == mock_instance


class TestComponentFactoriesRegistry:
    """Test the COMPONENT_FACTORIES registry."""

    def test_all_component_types_registered(self) -> None:
        """Test that all ComponentType enum values have factories."""
        for component_type in ComponentType:
            assert component_type in COMPONENT_FACTORIES, (
                f"ComponentType.{component_type.name} not registered in COMPONENT_FACTORIES"
            )

    def test_string_aliases_registered(self) -> None:
        """Test that expected string aliases are registered."""
        expected_aliases = [
            "embedding_generator",
            "faiss_store",
            "llm_generator",
            "sentiment_analyzer",
            "toxicity_filter",
            "orchestrator",
        ]
        for alias in expected_aliases:
            assert alias in COMPONENT_FACTORIES, (
                f"String alias '{alias}' not registered in COMPONENT_FACTORIES"
            )

    def test_factories_are_callable(self) -> None:
        """Test that all factory values are callable."""
        for key, factory in COMPONENT_FACTORIES.items():
            assert callable(factory), f"Factory for {key} is not callable"
