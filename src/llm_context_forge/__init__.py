"""
LLM Context Forge — Production-Grade LLMOps Infrastructure

Provides tools for managing LLM context windows, counting tokens
accurately, chunking documents intelligently, estimating costs,
and compressing prompts to fit within model limits.
"""

from llm_context_forge.models import ModelRegistry, TokenizerBackend, ModelInfo, PricingDataStaleWarning
from llm_context_forge.tokenizer import TokenCounter
from llm_context_forge.chunker import DocumentChunker, ChunkStrategy, Chunk
from llm_context_forge.context import ContextWindow, ContextBlock, Priority, ConversationManager
from llm_context_forge.compressor import ContextCompressor, CompressionStrategy, CompressionResult
from llm_context_forge.cost import CostCalculator, Cost, ConversationCost, BulkCostAnalysis
from llm_context_forge.pricing_provider import (
    PricingProvider,
    BundledYAMLPricingProvider,
    OpenRouterPricingProvider,
    update_pricing_registry,
)

__version__ = "1.0.0"

__all__ = [
    # Version
    "__version__",
    # Models & Pricing
    "ModelRegistry",
    "ModelInfo",
    "TokenizerBackend",
    "PricingDataStaleWarning",
    "PricingProvider",
    "BundledYAMLPricingProvider",
    "OpenRouterPricingProvider",
    "update_pricing_registry",
    # Tokenizer
    "TokenCounter",
    # Chunker
    "DocumentChunker",
    "ChunkStrategy",
    "Chunk",
    # Context
    "ContextWindow",
    "ContextBlock",
    "Priority",
    "ConversationManager",
    # Compressor
    "ContextCompressor",
    "CompressionStrategy",
    "CompressionResult",
    # Cost
    "CostCalculator",
    "Cost",
    "ConversationCost",
    "BulkCostAnalysis",
]
