"""
LangChain Integration for LLM Context Forge.

Provides `ContextForgeTextSplitter` and `ContextForgeDocumentTransformer`
compatible with LangChain pipelines.
"""

from typing import List, Any, Optional
from llm_context_forge.chunker import DocumentChunker, ChunkStrategy

# Try importing LangChain base classes, otherwise provide standard standalone wrappers
try:
    from langchain_text_splitters import TextSplitter
except ImportError:
    class TextSplitter:  # type: ignore
        def __init__(self, **kwargs):
            pass

try:
    from langchain_core.documents import Document
    from langchain_core.documents.transformers import BaseDocumentTransformer
except ImportError:
    class Document:  # type: ignore
        def __init__(self, page_content: str, metadata: Optional[dict] = None):
            self.page_content = page_content
            self.metadata = metadata or {}

    class BaseDocumentTransformer:  # type: ignore
        pass


class ContextForgeTextSplitter(TextSplitter):
    """LangChain-compatible TextSplitter delegating to DocumentChunker."""

    def __init__(
        self,
        model: str = "gpt-4o",
        strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH,
        max_tokens: int = 500,
        overlap_tokens: int = 50,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._chunker = DocumentChunker(default_model=model)
        self._strategy = strategy
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._model = model

    def split_text(self, text: str) -> List[str]:
        """Split text into a list of chunk strings."""
        chunks = self._chunker.chunk(
            text,
            strategy=self._strategy,
            max_tokens=self._max_tokens,
            overlap_tokens=self._overlap_tokens,
            model=self._model,
        )
        return [c.text for c in chunks]


class ContextForgeDocumentTransformer(BaseDocumentTransformer):
    """LangChain-compatible DocumentTransformer with sync and async support."""

    def __init__(
        self,
        model: str = "gpt-4o",
        strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH,
        max_tokens: int = 500,
        overlap_tokens: int = 50,
    ):
        self._chunker = DocumentChunker(default_model=model)
        self._strategy = strategy
        self._max_tokens = max_tokens
        self._overlap_tokens = overlap_tokens
        self._model = model

    def transform_documents(self, documents: List[Any], **kwargs) -> List[Any]:
        """Transform input documents by chunking their page content."""
        transformed = []
        for doc in documents:
            text = getattr(doc, "page_content", str(doc))
            base_meta = getattr(doc, "metadata", {})
            chunks = self._chunker.chunk(
                text,
                strategy=self._strategy,
                max_tokens=self._max_tokens,
                overlap_tokens=self._overlap_tokens,
                model=self._model,
            )
            for chunk in chunks:
                meta = {**base_meta, **chunk.metadata, "chunk_index": chunk.index}
                transformed.append(Document(page_content=chunk.text, metadata=meta))
        return transformed

    async def atransform_documents(self, documents: List[Any], **kwargs) -> List[Any]:
        """Async version of transform_documents."""
        import anyio
        return await anyio.to_thread.run_sync(self.transform_documents, documents)
