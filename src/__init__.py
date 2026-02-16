"""
ContextForge — LLM Context & Token Handling Toolkit

Provides tools for managing LLM context windows, counting tokens,
chunking documents intelligently, and compressing prompts.
"""

from src.tokenizer import TokenCounter, TokenizerBackend, ModelRegistry
from src.chunker import DocumentChunker, ChunkStrategy, Chunk
from src.context import ContextWindow, ContextBlock, Priority, ConversationManager
from src.compressor import ContextCompressor, CompressionStrategy, CompressionResult

__version__ = "1.0.0"

__all__ = [
    # Tokenizer
    "TokenCounter",
    "TokenizerBackend",
    "ModelRegistry",
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
]
