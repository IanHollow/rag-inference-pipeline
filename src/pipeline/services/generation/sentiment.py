"""
Sentiment analysis module for the generation service.

Manages nlptown/bert-base-multilingual-uncased-sentiment for sentiment classification.
"""

import logging
import time
from typing import ClassVar

import torch
from transformers import TextClassificationPipeline, pipeline as hf_pipeline

from ...config import PipelineSettings

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
        self.model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            # Use text-classification task which is the same as sentiment-analysis which is an alias
            device_arg: int | str = 0 if self.device.type == "cuda" else -1
            pipeline_result = hf_pipeline(
                task="text-classification",
                model=self.model_name,
                device=device_arg,
            )
            self.pipeline = pipeline_result

            self._loaded = True

            elapsed = time.time() - start_time
            logger.info("Sentiment model loaded in %.2f seconds", elapsed)

        except Exception as e:
            logger.exception("Failed to load sentiment model: %s", e)
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

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        sentiment = self.SENTIMENT_MAP.get(raw_label, "neutral")

        return sentiment

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
        sentiments = [self.SENTIMENT_MAP.get(result["label"], "neutral") for result in results]

        return sentiments
