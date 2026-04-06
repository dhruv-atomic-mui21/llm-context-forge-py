"""
Multi-Provider Token Counter

Accurate token counting for OpenAI (tiktoken), Anthropic, Google,
and Llama (sentencepiece) models. Includes cost estimation,
context-window validation, and ChatML-aware message counting.
"""

import math
from typing import Any, Dict, List, Optional
from llm_context_forge.models import ModelRegistry, TokenizerBackend, ModelInfo

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
        self._hf_cache: Dict[str, Any] = {}
        self._mistral_tokenizer: Any = None
        self._anthropic_tokenizer: Any = None
        self._fallback_warned = False

        # Add logging for production fallbacks
        import logging
        self.logger = logging.getLogger("llm_context_forge.tokenizer")

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
        
        try:
            if info.backend == TokenizerBackend.HUGGINGFACE and info.encoding_name:
                return self._count_huggingface(text, info.encoding_name)
            elif info.backend == TokenizerBackend.MISTRAL:
                return self._count_mistral(text, info.name)
            elif info.backend == TokenizerBackend.ANTHROPIC:
                return self._count_anthropic(text)
        except (ImportError, Exception) as e:
            if not self._fallback_warned:
                self.logger.warning(f"Exact tokenizer unavailable for {info.name} ({e}). Using production-grade fallback.")
                self._fallback_warned = True

        return self._count_estimate(text, info.backend)

    def count_batch(self, texts: List[str], model: Optional[str] = None) -> List[int]:
        """Count tokens for multiple texts efficiently"""
        return [self.count(t, model) for t in texts]

    def count_with_warnings(self, text: str, model: Optional[str] = None, warn_threshold: float = 0.8) -> Dict[str, Any]:
        """Return token count + warnings if approaching limit"""
        tokens = self.count(text, model)
        info = ModelRegistry.get(model or self.default_model)
        warnings = []
        if tokens >= info.context_window:
            warnings.append(f"Exceeds context window ({tokens} / {info.context_window})")
        elif tokens >= (info.context_window * warn_threshold):
            warnings.append(f"Approaching context window limit ({tokens} / {info.context_window})")
        return {"tokens": tokens, "warnings": warnings}

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
        limit = (max_tokens if max_tokens is not None else info.context_window) - reserve_output
        return self.count(text, model) <= limit

    def truncate_to_fit(
        self,
        text: str,
        max_tokens: int,
        model: Optional[str] = None,
        suffix: str = "\\n... [truncated]",
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

    def _count_huggingface(self, text: str, encoding_name: str) -> int:
        """Count tokens using Hugging Face transformers."""
        if encoding_name not in self._hf_cache:
            from transformers import AutoTokenizer
            self._hf_cache[encoding_name] = AutoTokenizer.from_pretrained(encoding_name)
        return len(self._hf_cache[encoding_name].encode(text))

    def _count_mistral(self, text: str, model_name: str) -> int:
        """Count tokens using mistral-common."""
        if not self._mistral_tokenizer:
            from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
            # Load the default mistral tokenizer (v3)
            self._mistral_tokenizer = MistralTokenizer.v3(is_tekken=True)
        return len(self._mistral_tokenizer.encode_chat_completion(
            {"messages": [{"role": "user", "content": text}]}
        ).tokens)

    def _count_anthropic(self, text: str) -> int:
        """Count tokens using anthropic client SDK."""
        if not self._anthropic_tokenizer:
            import anthropic
            self._anthropic_tokenizer = anthropic.Client().get_tokenizer()
        return len(self._anthropic_tokenizer.encode(text))

    def _count_estimate(self, text: str, backend: TokenizerBackend) -> int:
        """
        Production-grade fallback token count estimation.
        Uses OpenAI's tiktoken cl100k_base with a safety scalar since
        most modern tokenizers use similar BPE models.
        """
        enc = self._get_tiktoken_encoder("cl100k_base")
        base_count = len(enc.encode(text))
        
        # Apply safety multiplier based on target backend
        multipliers = {
            TokenizerBackend.LLAMA: 1.05,
            TokenizerBackend.MISTRAL: 1.05,
            TokenizerBackend.GOOGLE: 1.0,
            TokenizerBackend.ANTHROPIC: 1.05,
        }
        mult = multipliers.get(backend, 1.1)
        
        return max(1, math.ceil(base_count * mult))


if __name__ == "__main__":
    print("LLM Context Forge — Token Counter")
    print("=" * 40)
    print("Usage:")
    print('  counter = TokenCounter("gpt-4o")')
    print('  tokens  = counter.count("Hello world!")')
    print('  cost    = counter.estimate_cost("Hello world!")')
    print(f"  models  = {ModelRegistry.list_models()}")
