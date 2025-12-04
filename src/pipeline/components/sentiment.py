"""
Sentiment analysis module for the generation service.

Manages nlptown/bert-base-multilingual-uncased-sentiment for sentiment classification.
"""

import gc
import logging
import time
from typing import TYPE_CHECKING, ClassVar

import torch
from transformers import pipeline as hf_pipeline

from pipeline.config import PipelineSettings


if TYPE_CHECKING:
    from transformers import TextClassificationPipeline


logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment analyzer using nlptown/bert-base-multilingual-uncased-sentiment.

    Loads the model once and keeps it hot in memory.
    Operates on truncated outputs to stay within memory limits.
    """

    # Sentiment mapping from model labels to required output format
    SENTIMENT_MAP: ClassVar[dict[str, str]] = {
        "1 star": "very negative",
        "2 stars": "negative",
        "3 stars": "neutral",
        "4 stars": "positive",
        "5 stars": "very positive",
    }

    def __init__(self, settings: PipelineSettings) -> None:
        """
        Initialize the sentiment analyzer.

        Args:
            settings: Pipeline configuration settings
        """
        self.settings = settings
        self.model_name = settings.sentiment_model_name

        if settings.only_cpu:
            device_name = "cpu"
        elif torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"

        self.device = torch.device(device_name)
        self.pipeline: TextClassificationPipeline | None = None
        self._loaded = False

        logger.info("Sentiment Analyzer initialized (device: %s)", self.device)

    def load(self) -> None:
        """Load the sentiment analysis model into memory."""
        if self._loaded:
            logger.warning("Sentiment model already loaded")
            return

        logger.info("Loading sentiment model: %s", self.model_name)
        start_time = time.time()

        try:
            # HuggingFace pipeline device: 0 for cuda, "mps" for Apple Silicon, -1 for CPU
            if self.device.type == "cuda":
                device_arg: int | str = 0
            elif self.device.type == "mps":
                device_arg = "mps"
            else:
                device_arg = -1

            # Use float16 for CUDA and MPS, float32 for CPU
            use_fp16 = self.device.type in ("cuda", "mps")
            model_dtype = torch.float16 if use_fp16 else torch.float32

            pipeline_result = hf_pipeline(
                task="text-classification",
                model=self.model_name,
                device=device_arg,
                dtype=model_dtype,
            )
            self.pipeline = pipeline_result

            self._loaded = True

            # Warmup
            logger.info("Warming up sentiment model...")
            self.pipeline("This is a test sentence for warmup.")
            logger.info("Sentiment model warmup complete")

            elapsed = time.time() - start_time
            logger.info("Sentiment model loaded in %.2f seconds", elapsed)

        except Exception:
            logger.exception("Failed to load sentiment model")
            raise

    def unload(self) -> None:
        """Unload the model from memory."""
        if not self._loaded:
            return

        logger.info("Unloading sentiment model")

        if self.pipeline is not None:
            # Delete the pipeline and its components
            del self.pipeline
            self.pipeline = None

        # Force garbage collection before clearing device cache
        gc.collect()

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

        self._loaded = False
        logger.info("Sentiment model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded

    def analyze(self, text: str) -> str:
        """
        Analyze sentiment of a single text.

        Args:
            text: Text to analyze

        Returns:
            Sentiment string: 'very negative', 'negative', 'neutral', 'positive', or 'very positive'

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._loaded or self.pipeline is None:
            msg = "Sentiment model not loaded"
            raise RuntimeError(msg)

        # Truncate text to avoid memory issues
        truncated_text = text[: self.settings.truncate_length]

        # Run sentiment analysis
        result = self.pipeline(truncated_text)[0]

        # Map to required output format
        raw_label = result["label"]
        return self.SENTIMENT_MAP.get(raw_label, "neutral")

    def analyze_batch(self, texts: list[str]) -> list[str]:
        """
        Analyze sentiment of a batch of texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of sentiment strings

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._loaded or self.pipeline is None:
            msg = "Sentiment model not loaded"
            raise RuntimeError(msg)

        # Truncate all texts
        truncated_texts = [text[: self.settings.truncate_length] for text in texts]

        # Run batch sentiment analysis
        results = self.pipeline(truncated_texts)

        # Map all results to required output format
        return [self.SENTIMENT_MAP.get(result["label"], "neutral") for result in results]
