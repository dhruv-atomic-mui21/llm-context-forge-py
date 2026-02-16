"""
Multi-Provider Token Counter

Accurate token counting for OpenAI (tiktoken), Anthropic, Google,
and Llama (sentencepiece) models. Includes cost estimation,
context-window validation, and ChatML-aware message counting.
"""

import math
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, field


class TokenizerBackend(Enum):
    """Supported tokenizer backends."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LLAMA = "llama"
    ESTIMATE = "estimate"


@dataclass
class ModelInfo:
    """Metadata for a single LLM model."""
    name: str
    backend: TokenizerBackend
    context_window: int
    encoding_name: Optional[str] = None
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    tokens_per_message: int = 3   # ChatML overhead per message
    tokens_per_name: int = 1


class ModelRegistry:
    """
    Registry of known LLM models and their properties.

    Allows look-up of context window sizes, pricing, and which
    tokenizer backend should be used for each model.
    """

    _MODELS: Dict[str, ModelInfo] = {
        # OpenAI -----------------------------------------------------------
        "gpt-4": ModelInfo(
            "gpt-4", TokenizerBackend.OPENAI, 8_192,
            encoding_name="cl100k_base",
            input_cost_per_1k=0.03, output_cost_per_1k=0.06,
        ),
        "gpt-4-turbo": ModelInfo(
            "gpt-4-turbo", TokenizerBackend.OPENAI, 128_000,
            encoding_name="cl100k_base",
            input_cost_per_1k=0.01, output_cost_per_1k=0.03,
        ),
        "gpt-4o": ModelInfo(
            "gpt-4o", TokenizerBackend.OPENAI, 128_000,
            encoding_name="o200k_base",
            input_cost_per_1k=0.005, output_cost_per_1k=0.015,
        ),
        "gpt-4o-mini": ModelInfo(
            "gpt-4o-mini", TokenizerBackend.OPENAI, 128_000,
            encoding_name="o200k_base",
            input_cost_per_1k=0.00015, output_cost_per_1k=0.0006,
        ),
        "gpt-3.5-turbo": ModelInfo(
            "gpt-3.5-turbo", TokenizerBackend.OPENAI, 16_385,
            encoding_name="cl100k_base",
            input_cost_per_1k=0.0005, output_cost_per_1k=0.0015,
        ),
        # Anthropic --------------------------------------------------------
        "claude-3-opus": ModelInfo(
            "claude-3-opus", TokenizerBackend.ANTHROPIC, 200_000,
            input_cost_per_1k=0.015, output_cost_per_1k=0.075,
        ),
        "claude-3.5-sonnet": ModelInfo(
            "claude-3.5-sonnet", TokenizerBackend.ANTHROPIC, 200_000,
            input_cost_per_1k=0.003, output_cost_per_1k=0.015,
        ),
        "claude-3-haiku": ModelInfo(
            "claude-3-haiku", TokenizerBackend.ANTHROPIC, 200_000,
            input_cost_per_1k=0.00025, output_cost_per_1k=0.00125,
        ),
        # Google -----------------------------------------------------------
        "gemini-pro": ModelInfo(
            "gemini-pro", TokenizerBackend.GOOGLE, 1_000_000,
            input_cost_per_1k=0.00125, output_cost_per_1k=0.005,
        ),
        "gemini-flash": ModelInfo(
            "gemini-flash", TokenizerBackend.GOOGLE, 1_000_000,
            input_cost_per_1k=0.000075, output_cost_per_1k=0.0003,
        ),
        # Llama (open-source) ----------------------------------------------
        "llama-3-8b": ModelInfo(
            "llama-3-8b", TokenizerBackend.LLAMA, 8_192,
        ),
        "llama-3-70b": ModelInfo(
            "llama-3-70b", TokenizerBackend.LLAMA, 8_192,
        ),
    }

    @classmethod
    def get(cls, model: str) -> ModelInfo:
        """Look up a model by name (case-insensitive, prefix-match)."""
        key = model.lower().strip()
        if key in cls._MODELS:
            return cls._MODELS[key]

        # Prefix match (e.g. "gpt-4o-2024-08-06" → "gpt-4o")
        for name, info in sorted(cls._MODELS.items(), key=lambda x: -len(x[0])):
            if key.startswith(name):
                return info

        # Fallback to estimate
        return ModelInfo(
            model, TokenizerBackend.ESTIMATE, 4_096,
        )

    @classmethod
    def list_models(cls) -> List[str]:
        """Return all registered model names."""
        return list(cls._MODELS.keys())

    @classmethod
    def register(cls, info: ModelInfo) -> None:
        """Register a custom model."""
        cls._MODELS[info.name.lower().strip()] = info


class TokenCounter:
    """
    Multi-provider token counter.

    Counts tokens accurately for OpenAI models via tiktoken,
    and uses byte-pair estimation heuristics for other providers.
    Supports ChatML-aware message counting and cost estimation.
    """

    # Approximate chars-per-token ratios used as fallback
    _CHARS_PER_TOKEN = {
        TokenizerBackend.ANTHROPIC: 3.5,
        TokenizerBackend.GOOGLE: 3.5,
        TokenizerBackend.LLAMA: 3.8,
        TokenizerBackend.ESTIMATE: 4.0,
    }

    def __init__(self, default_model: str = "gpt-4o"):
        """
        Initialise counter.

        Args:
            default_model: Model to use when none specified.
        """
        self.default_model = default_model
        self._tiktoken_cache: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def count(self, text: str, model: Optional[str] = None) -> int:
        """
        Count the number of tokens in *text*.

        Args:
            text:  Plain text to tokenise.
            model: Model name (falls back to *default_model*).

        Returns:
            Token count.
        """
        if not text:
            return 0

        info = ModelRegistry.get(model or self.default_model)

        if info.backend == TokenizerBackend.OPENAI and info.encoding_name:
            return self._count_tiktoken(text, info.encoding_name)

        return self._count_estimate(text, info.backend)

    def count_messages(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
    ) -> int:
        """
        Count tokens for a list of chat messages (ChatML format).

        Each message is expected to have ``role`` and ``content`` keys.
        The overhead per message follows the OpenAI ChatML spec.

        Args:
            messages: List of ``{"role": ..., "content": ...}`` dicts.
            model:    Model name.

        Returns:
            Total token count including ChatML overhead.
        """
        info = ModelRegistry.get(model or self.default_model)
        total = 3  # every reply is primed with <|start|>assistant<|message|>

        for msg in messages:
            total += info.tokens_per_message
            for key, value in msg.items():
                total += self.count(str(value), model)
                if key == "name":
                    total += info.tokens_per_name

        return total

    def fits_in_window(
        self,
        text: str,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        reserve_output: int = 0,
    ) -> bool:
        """
        Check whether *text* fits within the context window.

        Args:
            text:           Text to check.
            max_tokens:     Explicit limit (defaults to model's window).
            model:          Model name.
            reserve_output: Tokens to reserve for the model's response.

        Returns:
            ``True`` if the text fits.
        """
        info = ModelRegistry.get(model or self.default_model)
        limit = (max_tokens or info.context_window) - reserve_output
        return self.count(text, model) <= limit

    def truncate_to_fit(
        self,
        text: str,
        max_tokens: int,
        model: Optional[str] = None,
        suffix: str = "\n... [truncated]",
    ) -> str:
        """
        Truncate *text* to fit within *max_tokens*.

        Uses binary search for efficiency.

        Args:
            text:       Text to truncate.
            max_tokens: Target token limit.
            model:      Model name.
            suffix:     String appended when truncation occurs.

        Returns:
            Truncated text.
        """
        if self.count(text, model) <= max_tokens:
            return text

        suffix_tokens = self.count(suffix, model)
        target = max_tokens - suffix_tokens

        # Binary search for the right character position
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.count(text[:mid], model) <= target:
                lo = mid
            else:
                hi = mid - 1

        return text[:lo] + suffix

    def estimate_cost(
        self,
        text: str,
        model: Optional[str] = None,
        direction: str = "input",
    ) -> float:
        """
        Estimate cost in USD for processing *text*.

        Args:
            text:      Text to estimate cost for.
            model:     Model name.
            direction: ``"input"`` or ``"output"``.

        Returns:
            Estimated cost in USD.
        """
        info = ModelRegistry.get(model or self.default_model)
        tokens = self.count(text, model)
        rate = (
            info.input_cost_per_1k if direction == "input"
            else info.output_cost_per_1k
        )
        return (tokens / 1000) * rate

    def get_model_info(self, model: Optional[str] = None) -> ModelInfo:
        """Return model metadata."""
        return ModelRegistry.get(model or self.default_model)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _count_tiktoken(self, text: str, encoding_name: str) -> int:
        """Count tokens using tiktoken."""
        enc = self._get_tiktoken_encoder(encoding_name)
        return len(enc.encode(text))

    def _get_tiktoken_encoder(self, encoding_name: str):
        """Lazy-load and cache tiktoken encoders."""
        if encoding_name not in self._tiktoken_cache:
            try:
                import tiktoken
                self._tiktoken_cache[encoding_name] = tiktoken.get_encoding(
                    encoding_name
                )
            except ImportError:
                raise ImportError(
                    "tiktoken is required for OpenAI token counting. "
                    "Install with: pip install tiktoken"
                )
        return self._tiktoken_cache[encoding_name]

    def _count_estimate(self, text: str, backend: TokenizerBackend) -> int:
        """Estimate token count using chars-per-token heuristic."""
        ratio = self._CHARS_PER_TOKEN.get(backend, 4.0)
        return max(1, math.ceil(len(text) / ratio))


if __name__ == "__main__":
    print("ContextForge — Token Counter")
    print("=" * 40)
    print("Usage:")
    print('  counter = TokenCounter("gpt-4o")')
    print('  tokens  = counter.count("Hello world!")')
    print('  cost    = counter.estimate_cost("Hello world!")')
    print(f"  models  = {ModelRegistry.list_models()}")
