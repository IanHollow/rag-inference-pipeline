"""
LLM module for the generation service.

Manages Qwen/Qwen2.5-0.5B-Instruct for response generation.
"""

from __future__ import annotations

import builtins
from collections.abc import Callable
import gc
import logging
import time
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


if TYPE_CHECKING:
    from torch import nn
    from transformers import PreTrainedTokenizerBase
    from transformers.modeling_utils import PreTrainedModel

    from pipeline.config import PipelineSettings

    from .schemas import RerankedDocument


logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Callable[..., Any])


def _noop_profile(func: T) -> T:
    return func


profile = getattr(builtins, "profile", _noop_profile)


def _clear_memory(device: torch.device) -> None:
    """Clear GPU/MPS memory caches to prevent fragmentation."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


class LLMGenerator:
    """
    LLM response generator using Qwen/Qwen2.5-0.5B-Instruct.

    Loads the model once and keeps it resident in memory.
    Generates responses with max_tokens=128 constraint.
    Uses torch autocast and tokenizer caching to stay within 16GB RAM.
    """

    # Generation counter for periodic memory cleanup
    _generation_count: int = 0
    _cleanup_interval: int = 50  # Clean memory every N generations

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

            # Determine optimal dtype based on device
            # - CUDA/MPS: float16 for best GPU performance
            # - CPU with BF16 support: bfloat16 for ~2x speedup on Intel AMX / newer CPUs
            # - CPU without BF16: float32 (safest fallback)
            if self.device.type in ("cuda", "mps"):
                model_dtype = torch.float16
                dtype_name = "float16"
            elif (
                self.device.type == "cpu"
                and hasattr(torch, "backends")
                and hasattr(torch.backends, "cpu")
                and getattr(torch.backends.cpu, "is_bf16_supported", lambda: False)()
            ):
                model_dtype = torch.bfloat16
                dtype_name = "bfloat16"
            else:
                model_dtype = torch.float32
                dtype_name = "float32"

            logger.info("Loading LLM with dtype=%s", dtype_name)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=model_dtype,
                low_cpu_mem_usage=True,  # Reduce memory during loading
            )

            cast("nn.Module", model).to(self.device)
            model.eval()

            # Enable memory-efficient attention if available (for compatible models)
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = True

            # Apply SDPA (Scaled Dot Product Attention) optimization for PyTorch 2.0+
            # This provides significant speedups on both CPU and GPU through Flash Attention
            if hasattr(model.config, "attn_implementation"):
                try:
                    # SDPA is the native PyTorch attention implementation with Flash Attention
                    model.config.attn_implementation = "sdpa"
                    logger.info("Enabled SDPA (Flash Attention) for LLM")
                except Exception as e:
                    logger.debug("SDPA not available for this model: %s", e)

            # Apply torch.compile for optimization (PyTorch 2.0+)
            # Only enabled for GPU - CPU inference with variable-length inputs
            # often gets slower due to graph capture and recompilation overhead
            if (
                self.settings.enable_torch_compile
                and hasattr(torch, "compile")
                and self.device.type in ("cuda", "mps")
            ):
                try:
                    # CUDA: reduce-overhead for CUDA graphs; MPS: default mode
                    compile_mode = "reduce-overhead" if self.device.type == "cuda" else "default"

                    logger.info("Applying torch.compile (mode=%s)", compile_mode)
                    model = torch.compile(  # type: ignore[assignment]
                        model,
                        mode=compile_mode,
                        fullgraph=False,  # Allow graph breaks for compatibility
                    )
                    logger.info("torch.compile applied successfully")
                except Exception as e:
                    logger.warning("torch.compile failed, continuing without: %s", e)
            elif self.device.type == "cpu" and self.settings.enable_torch_compile:
                logger.info("Skipping torch.compile on CPU (variable-length inputs cause overhead)")

            self.model = cast("PreTrainedModel", model)
            self._loaded = True

            # Clear any temporary allocations from loading
            _clear_memory(self.device)

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
                generate_fn = cast("PreTrainedModel", self.model.generate)

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

        except Exception:
            logger.exception("Failed to load LLM model")
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

        # Force garbage collection before clearing device cache
        gc.collect()

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

        return [
            {
                "role": "system",
                "content": "When given Context and Question, reply as 'Answer: <final answer>' only.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:",
            },
        ]

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

        # Explicit cleanup of intermediate tensors to prevent memory leaks
        del model_inputs, generated_ids, generated_tokens

        # Periodic memory cleanup to prevent fragmentation
        LLMGenerator._generation_count += 1
        if LLMGenerator._generation_count % LLMGenerator._cleanup_interval == 0:
            _clear_memory(self.device)

        return response

    @profile
    def generate_batch(
        self,
        queries: list[str],
        reranked_docs_batch: list[list[RerankedDocument]],
    ) -> list[str]:
        """
        Generate responses for a batch of queries using true batched inference.

        This method batches all queries into a single forward pass for efficiency,
        significantly reducing overhead compared to sequential generation.

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

        if not queries:
            return []

        # For single item, use the optimized single-query path
        if len(queries) == 1:
            return [self.generate(queries[0], reranked_docs_batch[0])]

        # Build all prompts
        all_texts: list[str] = []
        for query, reranked_docs in zip(queries, reranked_docs_batch, strict=False):
            messages = self._build_prompt(query, reranked_docs)
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            all_texts.append(cast("str", text))

        # Batch tokenize with padding
        model_inputs = self.tokenizer(
            all_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.settings.truncate_length,
            return_attention_mask=True,
        ).to(self.device)

        # Store input lengths for each sequence (before padding)
        # We need to track where each input ends to extract only new tokens
        input_lengths = (model_inputs.attention_mask.sum(dim=1)).tolist()

        generate_fn = self.model.generate

        with torch.inference_mode():
            generated_ids = generate_fn(  # type: ignore[operator]
                input_ids=model_inputs["input_ids"],
                attention_mask=model_inputs["attention_mask"],
                max_new_tokens=self.settings.max_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True,
            )

            # Decode each sequence, extracting only the generated tokens
            responses: list[str] = []
            for i, input_len in enumerate(input_lengths):
                # Extract only the generated portion (after input)
                generated_tokens = generated_ids[i, input_len:]
                response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
                responses.append(response)

        # Explicit cleanup of intermediate tensors
        del model_inputs, generated_ids

        # Update generation counter and periodic cleanup
        LLMGenerator._generation_count += len(queries)
        if LLMGenerator._generation_count % LLMGenerator._cleanup_interval == 0:
            _clear_memory(self.device)

        return responses
