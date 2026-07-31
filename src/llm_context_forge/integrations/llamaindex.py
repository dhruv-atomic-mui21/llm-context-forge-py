"""
LlamaIndex Integration for LLM Context Forge.

Provides `ContextForgeNodeParser` subclassing `MetadataAwareTextSplitter`.
"""

from typing import List, Optional
from llm_context_forge.chunker import DocumentChunker, ChunkStrategy

try:
    from llama_index.core.node_parser import MetadataAwareTextSplitter
except ImportError:
    class MetadataAwareTextSplitter:  # type: ignore
        def __init__(self, **kwargs):
            pass


class ContextForgeNodeParser(MetadataAwareTextSplitter):
    """LlamaIndex-compatible NodeParser adapting chunking based on metadata."""

    include_metadata: bool = True
    include_prev_next_rel: bool = True

    def __init__(
        self,
        model: str = "gpt-4o",
        default_strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH,
        max_tokens: int = 500,
        overlap_tokens: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._chunker = DocumentChunker(default_model=model)
        self._default_strategy = default_strategy
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._model = model

    def split_text_metadata_aware(self, text: str, metadata_str: str) -> List[str]:
        """
        Adapt chunking strategy dynamically based on metadata contents.
        (e.g., code files -> CODE strategy; markdown -> HEURISTIC strategy).
        """
        meta_lower = metadata_str.lower()
        if "python" in meta_lower or "code" in meta_lower or "file_type: code" in meta_lower:
            strategy = ChunkStrategy.CODE
        elif "markdown" in meta_lower or "heading" in meta_lower or "md" in meta_lower:
            strategy = ChunkStrategy.HEURISTIC
        else:
            strategy = self._default_strategy

        chunks = self._chunker.chunk(
            text,
            strategy=strategy,
            max_tokens=self._max_tokens,
            overlap_tokens=self._overlap_tokens,
            model=self._model,
        )
        return [c.text for c in chunks]

    def split_text(self, text: str) -> List[str]:
        """Basic text splitting."""
        return self.split_text_metadata_aware(text, "")
