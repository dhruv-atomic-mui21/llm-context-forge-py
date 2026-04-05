"""
ContextForge — Production-Grade LLMOps Infrastructure

Provides tools for managing LLM context windows, counting tokens
accurately, chunking documents intelligently, estimating costs,
and compressing prompts to fit within model limits.
"""

from contextforge.models import ModelRegistry, TokenizerBackend, ModelInfo
from contextforge.tokenizer import TokenCounter
from contextforge.chunker import DocumentChunker, ChunkStrategy, Chunk
from contextforge.context import ContextWindow, ContextBlock, Priority, ConversationManager
from contextforge.compressor import ContextCompressor, CompressionStrategy, CompressionResult
from contextforge.cost import CostCalculator, Cost, ConversationCost, BulkCostAnalysis

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
