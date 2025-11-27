"""
LLM module for the generation service.

Manages Qwen/Qwen2.5-0.5B-Instruct for response generation.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
import logging
import time
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    from ..config import PipelineSettings
    from .schemas import RerankedDocument


logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Callable[..., Any])


def _noop_profile(func: T) -> T:
    return func


profile = getattr(builtins, "profile", _noop_profile)


class LLMGenerator:
    """
    LLM response generator using Qwen/Qwen2.5-0.5B-Instruct.

    Loads the model once and keeps it resident in memory.
    Generates responses with max_tokens=128 constraint.
    Uses torch autocast and tokenizer caching to stay within 16GB RAM.
    """

    def __init__(self, settings: PipelineSettings) -> None:
        """
        Initialize the LLM generator.

        Args:
            settings: Pipeline configuration settings
        """
        self.settings = settings
        self.model_name = settings.llm_model_name

        if settings.only_cpu:
            device_name = "cpu"
        elif torch.cuda.is_available():
            device_name = "cuda"
        elif torch.backends.mps.is_available():
            device_name = "mps"
        else:
            device_name = "cpu"

        self.device = torch.device(device_name)
        self.tokenizer: PreTrainedTokenizerBase | None = None
        self.model: PreTrainedModel | None = None
        self._loaded = False

        logger.info("LLM Generator initialized (device: %s)", self.device)

    def load(self) -> None:
        """Load the LLM model and tokenizer into memory."""
        if self._loaded:
            logger.warning("LLM model already loaded")
            return

        logger.info("Loading LLM model: %s", self.model_name)
        start_time = time.time()

        try:
            # Load tokenizer with caching
            tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
                self.model_name,
                use_fast=True,
            )

            # Ensure pad token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            self.tokenizer = tokenizer

            # Load model with appropriate dtype
            # Use float16 for CUDA and MPS (Apple Silicon), float32 for CPU
            use_fp16 = self.device.type in ("cuda", "mps")
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16 if use_fp16 else torch.float32,
                # TODO: look into re-enabling these options which require the accelerate python package
                # low_cpu_mem_usage=True,
                # device_map=str(self.device),
            )

            cast("nn.Module", model).to(self.device)
            model.eval()
            self.model = model
            self._loaded = True

            # Warmup
            logger.info("Warming up LLM...")
            if self.tokenizer:
                dummy_doc_content = (
                    "This is a test document content that simulates a real retrieved chunk. " * 5
                )
                warmup_messages = [
                    {
                        "role": "system",
                        "content": "When given Context and Question, reply as 'Answer: <final answer>' only.",
                    },
                    {
                        "role": "user",
                        "content": f"Context:\n- Doc 1: {dummy_doc_content}\n- Doc 2: {dummy_doc_content}\n- Doc 3: {dummy_doc_content}\n\nQuestion: What is this?\n\nAnswer:",
                    },
                ]
                warmup_text = self.tokenizer.apply_chat_template(
                    warmup_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                dummy_input = self.tokenizer(cast("str", warmup_text), return_tensors="pt").to(
                    self.device
                )
                dummy_input_ids = dummy_input["input_ids"]
                dummy_attention_mask = dummy_input.get("attention_mask")

                # Get generate method explicitly
                generate_fn = model.generate

                with torch.inference_mode():
                    generate_fn(
                        input_ids=dummy_input_ids,
                        attention_mask=dummy_attention_mask,
                        max_new_tokens=10,
                        pad_token_id=self.tokenizer.eos_token_id,
                        do_sample=False,
                        use_cache=True,
                    )

            logger.info("LLM warmup complete")
            elapsed = time.time() - start_time
            logger.info("LLM model loaded in %.2f seconds", elapsed)

        except Exception as e:
            logger.exception("Failed to load LLM model: %s", e)
            raise

    def unload(self) -> None:
        """Unload the model from memory."""
        if not self._loaded:
            return

        logger.info("Unloading LLM model")

        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        elif self.device.type == "mps":
            torch.mps.empty_cache()

        self._loaded = False
        logger.info("LLM model unloaded")

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._loaded

    def _build_prompt(
        self, query: str, reranked_docs: list[RerankedDocument]
    ) -> list[dict[str, str]]:
        """
        Build the prompt from query and reranked documents.

        Args:
            query: The user query
            reranked_docs: List of reranked documents (top-N)

        Returns:
            Formatted messages list for chat template
        """
        # Use top 3 documents for context
        top_docs = reranked_docs[:3]
        context = "\n".join([f"- {doc.title}: {doc.content[:200]}" for doc in top_docs])

        messages = [
            {
                "role": "system",
                "content": "When given Context and Question, reply as 'Answer: <final answer>' only.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            },
        ]

        return messages

    @profile
    def generate(
        self,
        query: str,
        reranked_docs: list[RerankedDocument],
    ) -> str:
        """
        Generate a response for a single query.

        Args:
            query: The user query
            reranked_docs: List of reranked documents

        Returns:
            Generated response string

        Raises:
            RuntimeError: If model is not loaded
        """
        if not self._loaded or self.model is None or self.tokenizer is None:
            msg = "LLM model not loaded"
            raise RuntimeError(msg)

        # Explicitly assert types for mypy
        model: PreTrainedModel = self.model
        tokenizer: PreTrainedTokenizerBase = self.tokenizer

        # Build prompt
        messages = self._build_prompt(query, reranked_docs)

        # Apply chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Tokenize
        model_inputs = tokenizer(
            cast("str", text),
            return_tensors="pt",
            return_attention_mask=True,
        ).to(self.device)

        # Cache input length before generation
        input_length = model_inputs.input_ids.shape[1]

        # Get generate method explicitly to work around transformers type stub issues
        generate_fn = model.generate

        # Use inference_mode for better performance (faster than no_grad)
        # No autocast needed since model is already loaded in float16 for CUDA/MPS
        with torch.inference_mode():
            generated_ids = generate_fn(  # type: ignore[operator]
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs.get("attention_mask"),
                max_new_tokens=self.settings.max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True,
            )

        # Extract only the generated tokens (slice directly on tensor)
        generated_tokens = generated_ids[0, input_length:]

        # Decode single sequence (faster than batch_decode for single item)
        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response

    def generate_batch(
        self,
        queries: list[str],
        reranked_docs_batch: list[list[RerankedDocument]],
    ) -> list[str]:
        """
        Generate responses for a batch of queries.

        Args:
            queries: List of query strings
            reranked_docs_batch: List of reranked document lists

        Returns:
            List of generated response strings

        Raises:
            RuntimeError: If model is not loaded
            ValueError: If queries and reranked_docs_batch have different lengths
        """
        if not self._loaded or self.model is None or self.tokenizer is None:
            msg = "LLM model not loaded"
            raise RuntimeError(msg)

        if len(queries) != len(reranked_docs_batch):
            msg = f"Queries ({len(queries)}) and reranked_docs ({len(reranked_docs_batch)}) must have same length"
            raise ValueError(msg)

        responses = []
        for query, reranked_docs in zip(queries, reranked_docs_batch, strict=False):
            response = self.generate(query, reranked_docs)
            responses.append(response)

        return responses
