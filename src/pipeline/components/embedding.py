"""
Embedding generation module for retrieval service.

Manages persistent SentenceTransformer model with warm cache and batch encoding.
"""

import builtins
from collections.abc import Callable
import hashlib
import logging
import threading
from typing import Any, TypeVar

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from ..config import PipelineSettings
from ..utils.cache import LRUCache

T = TypeVar("T", bound=Callable[..., Any])


def _noop_profile(func: T) -> T:
    return func


profile = getattr(builtins, "profile", _noop_profile)


logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Manages embedding generation with persistent model loading.
    """

    def __init__(self, settings: PipelineSettings) -> None:
        """
        Initialize the embedding generator.

        Args:
            settings: Pipeline configuration containing model name and device settings
        """
        self.settings = settings
        self.model_name = settings.embedding_model_name

        if settings.only_cpu:
            device_name = "cpu"
        elif torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"

        self.device = torch.device(device_name)
        self._model: SentenceTransformer | None = None
        self._is_loaded = False
        self._lock = threading.Lock()

        # Initialize cache
        self.cache = LRUCache[str, np.ndarray](
            capacity=10000, ttl=settings.cache_max_ttl, name="embedding_cache"
        )

    def load(self) -> None:
        """
        Load the SentenceTransformer model into memory.
        """
        if self._is_loaded:
            logger.info("Embedding model already loaded")
            return

        logger.info("Loading embedding model: %s", self.model_name)
        try:
            self._model = SentenceTransformer(self.model_name).to(self.device)
            self._model.eval()
            self._is_loaded = True

            # Warmup
            logger.info("Warming up embedding model...")
            # Use a longer text to trigger compilation for realistic sequence lengths
            dummy_text = "This is a test sentence for warmup. " * 10
            self._model.encode([dummy_text], convert_to_numpy=True)
            logger.info("Embedding model warmup complete")

            logger.info("Embedding model loaded successfully on %s", self.device)
        except Exception as e:
            logger.exception("Failed to load embedding model: %s", e)
            raise

    @profile
    def encode(self, texts: list[str]) -> np.ndarray:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of text strings to encode

        Returns:
            numpy array of shape (len(texts), embedding_dim) with normalized embeddings

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If texts is empty
        """
        if not self._is_loaded or self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        if not texts:
            raise ValueError("Cannot encode empty text list")

        logger.debug("Encoding batch of %d texts", len(texts))

        # If cache is disabled, encode all
        if getattr(self.settings, "disable_cache_for_profiling", True):
            return self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )

        # Check cache
        results: list[np.ndarray | None] = [None] * len(texts)
        missing_indices = []
        missing_texts = []

        with self._lock:
            for i, text in enumerate(texts):
                # Normalize text for cache key
                key = hashlib.sha256(text.strip().encode()).hexdigest()
                cached = self.cache.get(key)
                if cached is not None:
                    results[i] = cached
                else:
                    missing_indices.append(i)
                    missing_texts.append(text)

        # Encode missing
        if missing_texts:
            logger.debug("Cache miss for %d/%d texts", len(missing_texts), len(texts))
            try:
                embeddings = self._model.encode(
                    missing_texts,
                    batch_size=32,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                )

                with self._lock:
                    for i, embedding in zip(missing_indices, embeddings, strict=False):
                        results[i] = embedding
                        # Update cache
                        key = hashlib.sha256(texts[i].strip().encode()).hexdigest()
                        self.cache.put(key, embedding)
            except Exception as e:
                logger.exception("Failed to encode texts: %s", e)
                raise
        else:
            logger.debug("Cache hit for all %d texts", len(texts))

        return np.array(results)

    def clear_cache(self) -> None:
        """Clear the embedding cache safely."""
        with self._lock:
            self.cache.clear()

    def unload(self) -> None:
        """
        Unload the model from memory.
        """
        if self._is_loaded:
            logger.info("Unloading embedding model")
            self._model = None
            self._is_loaded = False
            # Force garbage collection
            import gc

            gc.collect()
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            elif self.device.type == "mps":
                torch.mps.empty_cache()

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._is_loaded

    def __repr__(self) -> str:
        """String representation of the generator."""
        status = "loaded" if self._is_loaded else "not loaded"
        return f"EmbeddingGenerator(model={self.model_name}, device={self.device}, status={status})"
