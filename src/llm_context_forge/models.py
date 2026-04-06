"""
Model Registry and Metadata definitions.
"""
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass

class TokenizerBackend(Enum):
    """Supported tokenizer backends."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    LLAMA = "llama"
    HUGGINGFACE = "huggingface"
    MISTRAL = "mistral"
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
            "gemini-pro", TokenizerBackend.GOOGLE, 2_000_000,
            input_cost_per_1k=0.0035, output_cost_per_1k=0.0105,
        ),
        "gemini-flash": ModelInfo(
            "gemini-flash", TokenizerBackend.GOOGLE, 1_000_000,
            input_cost_per_1k=0.000075, output_cost_per_1k=0.0003,
        ),
        # Llama (open-source) ----------------------------------------------
        "llama-3-8b": ModelInfo(
            "llama-3-8b", TokenizerBackend.HUGGINGFACE, 8_192,
            encoding_name="meta-llama/Meta-Llama-3-8B",
            input_cost_per_1k=0.00005, output_cost_per_1k=0.00005,
        ),
        "llama-3-70b": ModelInfo(
            "llama-3-70b", TokenizerBackend.HUGGINGFACE, 8_192,
            encoding_name="meta-llama/Meta-Llama-3-70B",
            input_cost_per_1k=0.00054, output_cost_per_1k=0.00054,
        ),
        "llama-3.1-405b": ModelInfo(
            "llama-3.1-405b", TokenizerBackend.HUGGINGFACE, 128_000,
            encoding_name="meta-llama/Meta-Llama-3.1-405B",
            input_cost_per_1k=0.002, output_cost_per_1k=0.002,
        ),
        # Mistral
        "mistral-large": ModelInfo(
            "mistral-large", TokenizerBackend.MISTRAL, 128_000,
            input_cost_per_1k=0.003, output_cost_per_1k=0.009,
        ),
        # Cohere
        "command-r-plus": ModelInfo(
            "command-r-plus", TokenizerBackend.HUGGINGFACE, 128_000,
            encoding_name="CohereForAI/c4ai-command-r-plus",
            input_cost_per_1k=0.0025, output_cost_per_1k=0.010,
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
