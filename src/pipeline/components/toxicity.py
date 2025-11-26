"""
Toxicity detection module for the generation service.

Manages unitary/toxic-bert for toxicity/safety filtering.
"""

import logging
import time

import torch
from transformers import TextClassificationPipeline, pipeline as hf_pipeline

from ..config import PipelineSettings

logger = logging.getLogger(__name__)


class ToxicityFilter:
    """
    Toxicity filter using unitary/toxic-bert.

    Loads the model once and keeps it hot in memory.
    Operates on truncated outputs to stay within memory limits.
    Returns boolean flags indicating toxicity.
    """

    def __init__(self, settings: PipelineSettings) -> None:
        """
        Initialize the toxicity filter.

        Args:
            settings: Pipeline configuration settings
        """
        self.settings = settings
        self.model_name = settings.safety_model_name

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
        self.threshold = 0.5  # Toxicity threshold

        logger.info("Toxicity Filter initialized (device: %s)", self.device)

    def load(self) -> None:
        """Load the toxicity detection model into memory."""
        if self._loaded:
            logger.warning("Toxicity model already loaded")
            return

        logger.info("Loading toxicity model: %s", self.model_name)
        start_time = time.time()

        try:
            # HuggingFace pipeline device: 0 for cuda, "mps" for Apple Silicon, -1 for CPU
            if self.device.type == "cuda":
                device_arg: int | str = 0
            elif self.device.type == "mps":
                device_arg = "mps"
            else:
                device_arg = -1

            self.pipeline = hf_pipeline(
                "text-classification",
                model=self.model_name,
                device=device_arg,
            )

            self._loaded = True

            # Warmup
            logger.info("Warming up toxicity model...")
            self.pipeline("This is a test sentence for warmup.")
            logger.info("Toxicity model warmup complete")

            elapsed = time.time() - start_time
            logger.info("Toxicity model loaded in %.2f seconds", elapsed)

        except Exception as e:
            logger.exception("Failed to load toxicity model: %s", e)
            raise

    def unload(self) -> None:
        """Unload the model from memory."""
        if not self._loaded:
            return

        logger.info("Unloading toxicity model")

        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

        self._loaded = False
        logger.info("Toxicity model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded

    def filter(self, text: str) -> str:
        """
        Check if text is toxic.

        Args:
            text: Text to analyze

        Returns:
            String "true" if toxic, "false" otherwise

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._loaded or self.pipeline is None:
            msg = "Toxicity model not loaded"
            raise RuntimeError(msg)

        # Truncate text to avoid memory issues
        truncated_text = text[: self.settings.truncate_length]

        # Run toxicity detection
        result = self.pipeline(truncated_text)[0]

        # Check if score exceeds threshold
        is_toxic = result["score"] > self.threshold

        return "true" if is_toxic else "false"

    def filter_batch(self, texts: list[str]) -> list[str]:
        """
        Check toxicity for a batch of texts.

        Args:
            texts: List of texts to analyze

        Returns:
            List of strings ("true" or "false")

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._loaded or self.pipeline is None:
            msg = "Toxicity model not loaded"
            raise RuntimeError(msg)

        # Truncate all texts
        truncated_texts = [text[: self.settings.truncate_length] for text in texts]

        # Run batch toxicity detection
        results = self.pipeline(truncated_texts)

        # Check all results against threshold
        toxicity_flags = [
            "true" if result["score"] > self.threshold else "false" for result in results
        ]

        return toxicity_flags

    def check(self, text: str) -> tuple[bool, float]:
        """
        Check if text is toxic and return score.

        Args:
            text: Text to analyze

        Returns:
            Tuple of (is_toxic, score)

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._loaded or self.pipeline is None:
            msg = "Toxicity model not loaded"
            raise RuntimeError(msg)

        # Truncate text to avoid memory issues
        truncated_text = text[: self.settings.truncate_length]

        # Run toxicity detection
        result = self.pipeline(truncated_text)[0]
        score = result["score"]

        # Check if score exceeds threshold
        is_toxic = score > self.threshold

        return is_toxic, score
