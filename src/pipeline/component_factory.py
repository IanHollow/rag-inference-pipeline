from collections.abc import Callable
from typing import Any

from .components.document_store import DocumentStore
from .components.embedding import EmbeddingGenerator
from .components.faiss_store import FAISSStore
from .components.llm import LLMGenerator
from .components.reranker import Reranker
from .components.sentiment import SentimentAnalyzer
from .components.toxicity import ToxicityFilter
from .config import PipelineSettings
from .enums import ComponentType
from .services.gateway.orchestrator import Orchestrator


# Type alias for component factory function
ComponentFactory = Callable[[PipelineSettings, dict[str, Any]], Any]


def create_embedding_generator(
    settings: PipelineSettings, _config: dict[str, Any]
) -> EmbeddingGenerator:
    return EmbeddingGenerator(settings)


def create_faiss_store(settings: PipelineSettings, _config: dict[str, Any]) -> FAISSStore:
    return FAISSStore(settings)


def create_document_store(settings: PipelineSettings, _config: dict[str, Any]) -> DocumentStore:
    return DocumentStore(settings)


def create_reranker(settings: PipelineSettings, _config: dict[str, Any]) -> Reranker:
    return Reranker(settings)


def create_llm_generator(settings: PipelineSettings, _config: dict[str, Any]) -> LLMGenerator:
    return LLMGenerator(settings)


def create_sentiment_analyzer(
    settings: PipelineSettings, _config: dict[str, Any]
) -> SentimentAnalyzer:
    return SentimentAnalyzer(settings)


def create_toxicity_filter(settings: PipelineSettings, _config: dict[str, Any]) -> ToxicityFilter:
    return ToxicityFilter(settings)


def create_gateway_orchestrator(
    _settings: PipelineSettings, config: dict[str, Any]
) -> Orchestrator:
    return Orchestrator(
        retrieval_url=config.get("retrieval_url"),
        generation_url=config.get("generation_url"),
        batch_size=config.get("batch_size"),
        batch_timeout=config.get("batch_timeout"),
    )


COMPONENT_FACTORIES: dict[ComponentType | str, ComponentFactory] = {
    ComponentType.EMBEDDING: create_embedding_generator,
    "embedding_generator": create_embedding_generator,
    ComponentType.FAISS: create_faiss_store,
    "faiss_store": create_faiss_store,
    ComponentType.DOCUMENT_STORE: create_document_store,
    ComponentType.RERANKER: create_reranker,
    ComponentType.LLM: create_llm_generator,
    "llm_generator": create_llm_generator,
    ComponentType.SENTIMENT: create_sentiment_analyzer,
    "sentiment_analyzer": create_sentiment_analyzer,
    ComponentType.TOXICITY: create_toxicity_filter,
    "toxicity_filter": create_toxicity_filter,
    ComponentType.GATEWAY: create_gateway_orchestrator,
    "orchestrator": create_gateway_orchestrator,
}


def create_component(
    component_type: ComponentType | str,
    settings: PipelineSettings,
    config: dict[str, Any] | None = None,
) -> object:
    if config is None:
        config = {}

    # Try to resolve string to enum if possible, but allow string keys in map
    factory = COMPONENT_FACTORIES.get(component_type)

    # If not found, try to match by value if component_type is a string
    if not factory and isinstance(component_type, str):
        # Try to find enum by value
        try:
            enum_member = ComponentType(component_type)
            factory = COMPONENT_FACTORIES.get(enum_member)
        except ValueError:
            pass

    if not factory:
        msg = f"Unknown component type: {component_type}"
        raise ValueError(msg)
    return factory(settings, config)
