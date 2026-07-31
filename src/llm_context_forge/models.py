"""
Model Registry and Metadata definitions with Pricing Integrity.
"""
import os
import sys
import datetime
import warnings
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import yaml

class PricingDataStaleWarning(UserWarning):
    """Warning emitted when pricing data for a model is older than 30 days."""
    pass

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
    """Metadata and pricing for a single LLM model."""
    name: str
    backend: TokenizerBackend
    context_window: int
    encoding_name: Optional[str] = None
    input_cost_per_1k: float = 0.0
    output_cost_per_1k: float = 0.0
    cached_input_cost_per_1k: float = 0.0
    tokens_per_message: int = 3   # ChatML overhead per message
    tokens_per_name: int = 1
    provider: str = "custom"
    source_url: Optional[str] = None
    date_retrieved: Optional[str] = None
    confidence: str = "official"   # official | third-party
    registry_version: str = "1.0.0"

class ModelRegistry:
    """
    Registry of known LLM models and their properties.

    Loads model definitions and pricing dynamically from versioned YAML data files.
    Emits staleness warnings for pricing older than 30 days and supports local overrides.
    """

    _MODELS: Dict[str, ModelInfo] = {}
    _WARNED_MODELS: set = set()
    _INITIALIZED: bool = False

    @classmethod
    def _map_backend(cls, provider: str, name: str) -> TokenizerBackend:
        p = provider.lower()
        n = name.lower()
        if "openai" in p or "gpt" in n or n.startswith("o1") or n.startswith("o3"):
            return TokenizerBackend.OPENAI
        elif "anthropic" in p or "claude" in n:
            return TokenizerBackend.ANTHROPIC
        elif "google" in p or "gemini" in n:
            return TokenizerBackend.GOOGLE
        elif "mistral" in p:
            return TokenizerBackend.MISTRAL
        elif "groq" in p or "together" in p or "llama" in n or "command" in n:
            return TokenizerBackend.HUGGINGFACE
        return TokenizerBackend.ESTIMATE

    @classmethod
    def _map_encoding(cls, name: str) -> Optional[str]:
        n = name.lower()
        if "gpt-4o" in n or n.startswith("o1") or n.startswith("o3"):
            return "o200k_base"
        elif "gpt" in n:
            return "cl100k_base"
        elif "llama-3" in n:
            return "meta-llama/Meta-Llama-3-8B"
        elif "command" in n:
            return "CohereForAI/c4ai-command-r-plus"
        return None

    @classmethod
    def initialize(cls, force_reload: bool = False) -> None:
        """Initialize or reload the model registry from bundled/override YAML file."""
        if cls._INITIALIZED and not force_reload:
            return

        cls._MODELS.clear()
        
        # Path 1: Default bundled YAML file
        package_dir = os.path.dirname(os.path.abspath(__file__))
        bundled_path = os.path.join(package_dir, "data", "pricing_registry.yaml")

        sources = []
        if os.path.exists(bundled_path):
            sources.append(bundled_path)

        # Path 2: Environment variable override
        env_path = os.environ.get("LLM_CONTEXT_FORGE_PRICING_FILE")
        if env_path and os.path.exists(env_path):
            sources.append(env_path)

        for filepath in sources:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw_data = yaml.safe_load(f) or {}
                
                for model_name, item in raw_data.items():
                    provider = item.get("provider", "unknown")
                    input_per_1m = float(item.get("input_per_1m", 0.0))
                    output_per_1m = float(item.get("output_per_1m", 0.0))
                    cached_input_per_1m = float(item.get("cached_input_per_1m", input_per_1m / 2.0))

                    info = ModelInfo(
                        name=model_name,
                        backend=cls._map_backend(provider, model_name),
                        context_window=int(item.get("context_window", 4096)),
                        encoding_name=cls._map_encoding(model_name),
                        input_cost_per_1k=input_per_1m / 1000.0,
                        output_cost_per_1k=output_per_1m / 1000.0,
                        cached_input_cost_per_1k=cached_input_per_1m / 1000.0,
                        provider=provider,
                        source_url=item.get("source_url"),
                        date_retrieved=item.get("date_retrieved"),
                        confidence=item.get("confidence", "official"),
                        registry_version=str(item.get("registry_version", "1.0.0")),
                    )
                    cls._MODELS[model_name.lower().strip()] = info
            except Exception as e:
                warnings.warn(f"Failed to load pricing data from {filepath}: {e}", UserWarning)

        cls._INITIALIZED = True

    @classmethod
    def _check_staleness(cls, info: ModelInfo) -> None:
        if not info.date_retrieved or info.name in cls._WARNED_MODELS:
            return

        try:
            retrieved_date = datetime.date.fromisoformat(info.date_retrieved)
            today = datetime.date.today()
            age_days = (today - retrieved_date).days

            if age_days > 30:
                cls._WARNED_MODELS.add(info.name)
                warnings.warn(
                    f"llm-context-forge: pricing for {info.name} was last verified {age_days} days ago. "
                    "Run `llm_context_forge pricing update` or set LLM_CONTEXT_FORGE_PRICING_FILE "
                    "to suppress this warning.",
                    PricingDataStaleWarning,
                    stacklevel=3
                )
        except Exception:
            pass

    @classmethod
    def get(cls, model: str) -> ModelInfo:
        """Look up a model by name (case-insensitive, prefix-match)."""
        if not cls._INITIALIZED:
            cls.initialize()

        key = model.lower().strip()

        # Direct match
        if key in cls._MODELS:
            info = cls._MODELS[key]
            cls._check_staleness(info)
            return info

        # Prefix match (e.g. "gpt-4o-2024-08-06" → "gpt-4o")
        for name, info in sorted(cls._MODELS.items(), key=lambda x: -len(x[0])):
            if key.startswith(name):
                cls._check_staleness(info)
                return info

        # Fallback to estimate
        return ModelInfo(
            name=model,
            backend=TokenizerBackend.ESTIMATE,
            context_window=4_096,
        )

    @classmethod
    def list_models(cls) -> List[str]:
        """Return all registered model names."""
        if not cls._INITIALIZED:
            cls.initialize()
        return list(cls._MODELS.keys())

    @classmethod
    def register(cls, info: ModelInfo) -> None:
        """Register a custom model."""
        if not cls._INITIALIZED:
            cls.initialize()
        cls._MODELS[info.name.lower().strip()] = info


# Ensure auto-initialization on module import
ModelRegistry.initialize()
