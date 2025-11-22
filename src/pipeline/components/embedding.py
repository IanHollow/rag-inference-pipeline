"""
Embedding generation module for retrieval service.

Manages persistent SentenceTransformer model with warm cache and batch encoding.
"""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer
import torch

from ..config import PipelineSettings

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Manages embedding generation with persistent model loading.

    The model is loaded once and kept in memory for the lifetime of the service.
    Supports batch encoding for efficiency.
    """

    def __init__(self, settings: PipelineSettings) -> None:
        """
        Initialize the embedding generator.

        Args:
            settings: Pipeline configuration containing model name and device settings
        """
        self.settings = settings
        self.model_name = settings.embedding_model_name
        self.device = torch.device("cpu" if settings.only_cpu else "cuda")
        self._model: SentenceTransformer | None = None
        self._is_loaded = False

    def load(self) -> None:
        """
        Load the SentenceTransformer model into memory.

        This should be called during service startup to warm the cache.
        """
        if self._is_loaded:
            logger.info("Embedding model already loaded")
            return

        logger.info("Loading embedding model: %s", self.model_name)
        try:
            self._model = SentenceTransformer(self.model_name).to(self.device)
            self._model.eval()
            self._is_loaded = True
            logger.info("Embedding model loaded successfully on %s", self.device)
        except Exception as e:
            logger.exception("Failed to load embedding model: %s", e)
            raise

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

        try:
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,
                convert_to_numpy=True,
                batch_size=len(texts),  # Process all texts in one batch
                show_progress_bar=False,
            )
            logger.debug("Generated embeddings with shape %s", embeddings.shape)
            return embeddings
        except Exception as e:
            logger.exception("Failed to encode texts: %s", e)
            raise

    def unload(self) -> None:
        """
        Unload the model from memory.

        This is typically called during service shutdown.
        """
        if self._is_loaded:
            logger.info("Unloading embedding model")
            self._model = None
            self._is_loaded = False
            # Force garbage collection
            import gc

            gc.collect()
            if not self.settings.only_cpu:
                torch.cuda.empty_cache()

    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded."""
        return self._is_loaded

    def __repr__(self) -> str:
        """String representation of the generator."""
        status = "loaded" if self._is_loaded else "not loaded"
        return f"EmbeddingGenerator(model={self.model_name}, device={self.device}, status={status})"
