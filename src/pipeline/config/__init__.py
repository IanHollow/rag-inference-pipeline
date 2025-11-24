"""
Configuration management for the distributed ML pipeline.

This module uses Pydantic Settings to load configuration from environment variables
required by the project spec.
"""

import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ..enums import NodeRole, derive_node_role


def _default_cpu_threads() -> int:
    """Default to min(16, cpu_count) to avoid oversubscription on large nodes."""
    count = os.cpu_count() or 8
    return min(16, count)


class PipelineSettings(BaseSettings):
    """
    Central configuration for the ML pipeline.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # === Required Node Configuration ===
    total_nodes: int = Field(
        default=3,
        alias="TOTAL_NODES",
        description="Total number of nodes in the cluster",
    )

    node_number: int = Field(
        default=0,
        alias="NODE_NUMBER",
        description="This node's number (0-based index)",
    )

    node_0_ip: str = Field(
        default="localhost:8000",
        alias="NODE_0_IP",
        description="IP:port for node 0",
    )

    node_1_ip: str = Field(
        default="localhost:8001",
        alias="NODE_1_IP",
        description="IP:port for node 1",
    )

    node_2_ip: str = Field(
        default="localhost:8002",
        alias="NODE_2_IP",
        description="IP:port for node 2",
    )

    # === Profiling Configuration ===
    profiling_run_id: str = Field(
        default="default",
        alias="PROFILING_RUN_ID",
        description="Identifier for the current profiling run",
    )

    faiss_index_path: str = Field(
        default="faiss_index.bin",
        alias="FAISS_INDEX_PATH",
        description="Path to the pre-built FAISS index file",
    )

    documents_dir: str = Field(
        default="documents/",
        alias="DOCUMENTS_DIR",
        description="Directory containing the document database",
    )

    only_cpu: bool = Field(
        default=True,
        alias="ONLY_CPU",
        description="If true, only use CPU",
    )

    # === CPU Parallelism Configuration ===
    cpu_inference_threads: int = Field(
        default_factory=_default_cpu_threads,
        alias="CPU_INFERENCE_THREADS",
        description="Number of threads for CPU inference (torch, faiss)",
    )

    cpu_worker_threads: int = Field(
        default_factory=_default_cpu_threads,
        alias="CPU_WORKER_THREADS",
        description="Number of threads for worker pools",
    )

    max_parallel_generation: int = Field(
        default=4,
        alias="MAX_PARALLEL_GENERATION",
        description="Maximum number of concurrent generation requests",
    )

    enable_adaptive_batching: bool = Field(
        default=True,
        alias="ENABLE_ADAPTIVE_BATCHING",
        description="If true, use adaptive batching",
    )

    # === Caching Configuration ===
    gateway_response_cache_enabled: bool = Field(
        default=False,
        alias="GATEWAY_RESPONSE_CACHE_ENABLED",
        description="Enable gateway response caching",
    )

    cache_max_ttl: float = Field(
        default=60.0,
        alias="CACHE_MAX_TTL",
        description="Max TTL for caches in seconds",
    )

    disable_cache_for_profiling: bool = Field(
        default=False,
        alias="DISABLE_CACHE_FOR_PROFILING",
        description="Force disable caches during profiling",
    )

    gateway_cache_capacity: int = Field(
        default=1000,
        alias="GATEWAY_CACHE_CAPACITY",
        description="Capacity of the gateway response cache",
    )

    retrieval_cache_capacity: int = Field(
        default=1000,
        alias="RETRIEVAL_CACHE_CAPACITY",
        description="Capacity of the retrieval FAISS cache",
    )

    document_cache_capacity: int = Field(
        default=5000,
        alias="DOCUMENT_CACHE_CAPACITY",
        description="Capacity of the document store cache",
    )

    fuzzy_cache_matching: bool = Field(
        default=False,
        alias="FUZZY_CACHE_MATCHING",
        description="Enable fuzzy matching (token sort) for gateway cache",
    )

    # === Compression Configuration ===
    pipeline_rpc_compression: str = Field(
        default="none",
        alias="PIPELINE_RPC_COMPRESSION",
        description="Compression algorithm for RPC (none, zstd, lz4)",
    )

    pipeline_rpc_compression_level: int = Field(
        default=3,
        alias="PIPELINE_RPC_COMPRESSION_LEVEL",
        description="Compression level",
    )

    # === Document Payload Configuration ===
    documents_payload_mode: str = Field(
        default="full",
        alias="DOCUMENTS_PAYLOAD_MODE",
        description="Mode for document payload transfer: 'full', 'id_only', 'compressed'",
    )

    # === Pipeline Constants (Do Not Change) ===
    faiss_dim: int = Field(
        default=768,
        description="FAISS embedding dimension",
    )

    max_tokens: int = Field(
        default=128,
        description="Maximum tokens for LLM generation",
    )

    retrieval_k: int = Field(
        default=10,
        description="Number of documents to retrieve from FAISS",
    )

    truncate_length: int = Field(
        default=512,
        description="Maximum sequence length for truncation",
    )

    # === Role Configuration ===
    pipeline_role_profile: str = Field(
        default="",
        alias="PIPELINE_ROLE_PROFILE",
        description="Name of the role profile to use (overrides NODE_NUMBER based role)",
    )

    role_profile_override_path: str = Field(
        default="",
        alias="ROLE_PROFILE_OVERRIDE_PATH",
        description="Path to a JSON/YAML file containing a custom role profile",
    )

    rerank_top_n: int = Field(
        default=10,
        ge=1,
        description="Number of top documents to return after reranking",
    )

    # === Batching Configuration ===
    gateway_batch_size: int = Field(
        default=4,
        ge=1,
        description="Batch size for gateway request accumulation",
    )

    gateway_batch_timeout_ms: int = Field(
        default=100,
        ge=1,
        description="Max milliseconds to wait for a full batch at gateway",
    )

    retrieval_batch_size: int = Field(
        default=8,
        ge=1,
        description="Batch size for retrieval service operations",
    )

    generation_batch_size: int = Field(
        default=4,
        ge=1,
        description="Batch size for generation service operations",
    )

    # === Batching Tuning Knobs ===
    gateway_min_batch_size: int = Field(
        default=1,
        ge=1,
        description="Minimum batch size for gateway adaptive batching",
    )

    retrieval_max_batch_delay_ms: int = Field(
        default=50,
        ge=1,
        description="Max delay for retrieval batching",
    )

    generation_max_batch_delay_ms: int = Field(
        default=200,
        ge=1,
        description="Max delay for generation batching",
    )

    # === Model Names ===
    embedding_model_name: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description="Sentence transformer model for embeddings",
    )

    reranker_model_name: str = Field(
        default="BAAI/bge-reranker-base",
        description="Model for reranking retrieved documents",
    )

    llm_model_name: str = Field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        description="Large language model for response generation",
    )

    sentiment_model_name: str = Field(
        default="nlptown/bert-base-multilingual-uncased-sentiment",
        description="Model for sentiment analysis",
    )

    safety_model_name: str = Field(
        default="unitary/toxic-bert",
        description="Model for toxicity/safety filtering",
    )

    # === Server Configuration ===
    request_timeout_seconds: int = Field(
        default=300,
        ge=1,
        description="Maximum time to wait for a request to complete",
    )

    startup_wait_seconds: int = Field(
        default=300,
        ge=1,
        description="Maximum time allowed for service initialization",
    )

    # === Logging ===
    log_level: str = Field(
        default="INFO",
        alias="LOG_LEVEL",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    enable_tracing: bool = Field(
        default=True,
        alias="ENABLE_TRACING",
        description="If true, emit OpenTelemetry spans for pipeline stages",
    )

    enable_profiling: bool = Field(
        default=True,
        alias="ENABLE_PROFILING",
        description="If true, capture sampled psutil-based profiling snapshots",
    )

    profiling_sample_rate: float = Field(
        default=0.2,
        alias="PROFILING_SAMPLE_RATE",
        ge=0.0,
        le=1.0,
        description="Fraction of requests to capture profiling data for (0-1)",
    )

    otel_exporter_endpoint: str = Field(
        default="http://127.0.0.1:4317",
        alias="OTEL_EXPORTER_OTLP_ENDPOINT",
        description="OTLP gRPC endpoint for exporting traces",
    )

    otel_exporter_insecure: bool = Field(
        default=True,
        alias="OTEL_EXPORTER_OTLP_INSECURE",
        description="Use insecure (non-TLS) connection for OTLP exporter",
    )

    # Computed properties
    @property
    def role(self) -> NodeRole:
        """Derive this node's role from its node_number."""
        return derive_node_role(self.node_number)

    @property
    def node_ips(self) -> dict[int, str]:
        """Map of node numbers to their IP addresses."""
        return {
            0: self.node_0_ip,
            1: self.node_1_ip,
            2: self.node_2_ip,
        }

    @property
    def gateway_url(self) -> str:
        """Full HTTP URL for the gateway service (Node 0)."""
        return f"http://{self.node_0_ip}"

    @property
    def retrieval_url(self) -> str:
        """Full HTTP URL for the retrieval service (Node 1)."""
        return f"http://{self.node_1_ip}"

    @property
    def generation_url(self) -> str:
        """Full HTTP URL for the generation service (Node 2)."""
        return f"http://{self.node_2_ip}"

    @property
    def listen_host(self) -> str:
        """Host address for this node to listen on."""
        return self.node_ips[self.node_number].split(":")[0]

    @property
    def listen_port(self) -> int:
        """Port for this node to listen on."""
        ip_port = self.node_ips[self.node_number]
        return int(ip_port.split(":")[1]) if ":" in ip_port else 8000

    @field_validator("total_nodes")
    @classmethod
    def validate_total_nodes(cls, v: int) -> int:
        """Ensure total_nodes is exactly 3 (per project spec)."""
        if v != 3:
            raise ValueError(f"TOTAL_NODES must be 3 (got {v}).")
        return v

    @field_validator("node_number")
    @classmethod
    def validate_node_number(cls, v: int) -> int:
        """Ensure node_number is 0, 1, or 2."""
        if v not in {0, 1, 2}:
            raise ValueError(f"NODE_NUMBER must be 0, 1, or 2 (got {v})")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log_level is valid."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_upper = v.upper()
        if v_upper not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels} (got {v})")
        return v_upper


# Global settings instance
_settings: PipelineSettings | None = None


def get_settings() -> PipelineSettings:
    """
    Get the global PipelineSettings instance.

    This ensures settings are loaded exactly once and reused throughout the application.

    Returns:
        PipelineSettings: The global configuration instance
    """
    global _settings
    if _settings is None:
        _settings = PipelineSettings()
    return _settings
