"""
LayoutLM Forge — Production-Grade LLMOps Infrastructure

Provides tools for managing LLM context windows, counting tokens
accurately, chunking documents intelligently, estimating costs,
and compressing prompts to fit within model limits.
"""

from layoutlm_forge.models import ModelRegistry, TokenizerBackend, ModelInfo
from layoutlm_forge.tokenizer import TokenCounter
from layoutlm_forge.chunker import DocumentChunker, ChunkStrategy, Chunk
from layoutlm_forge.context import ContextWindow, ContextBlock, Priority, ConversationManager
from layoutlm_forge.compressor import ContextCompressor, CompressionStrategy, CompressionResult
from layoutlm_forge.cost import CostCalculator, Cost, ConversationCost, BulkCostAnalysis

__version__ = "0.1.0"

__all__ = [
    # Models
    "ModelRegistry",
    "ModelInfo",
    "TokenizerBackend",
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
