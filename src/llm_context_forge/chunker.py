"""
Intelligent Document Chunker

Splits text into token-bounded chunks using multiple strategies:
  - Fixed-size (character / token)
  - Sentence-aware
  - Paragraph-aware
  - Heuristic (markdown headings / structure)
  - Semantic (embedding-based percentile drop via sentence-transformers)
  - Code-aware (function / class boundaries)

Supports configurable overlap, automatic merging of small chunks, and async execution.
"""

import re
import warnings
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import anyio


class ChunkStrategy(Enum):
    """Chunking strategy."""
    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    HEURISTIC = "heuristic"
    SEMANTIC = "semantic"
    CODE = "code"


@dataclass
class Chunk:
    """Represents a single chunk of text."""
    text: str
    index: int
    token_count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)


class DocumentChunker:
    """
    Intelligent document chunker.

    Splits documents into token-bounded pieces while respecting
    natural boundaries (sentences, paragraphs, headings, code blocks, or semantic embeddings).
    """

    # Regex patterns for boundary detection
    _SENTENCE_SPLIT = re.compile(r'(?<=[.!?])\s+')
    _PARAGRAPH_SPLIT = re.compile(r'\n\s*\n')
    _HEADING_SPLIT = re.compile(r'^(#{1,6}\s)', re.MULTILINE)
    _CODE_FENCE = re.compile(r'^```', re.MULTILINE)
    _FUNC_DEF = re.compile(
        r'^(?:def |class |async def |function |const |let |var )',
        re.MULTILINE,
    )

    def __init__(self, default_model: str = "gpt-4o"):
        """
        Initialise chunker.

        Args:
            default_model: Model used for token counting.
        """
        from llm_context_forge.tokenizer import TokenCounter
        self._counter = TokenCounter(default_model)
        self.default_model = default_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(
        self,
        text: str,
        strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH,
        max_tokens: int = 500,
        overlap_tokens: int = 50,
        model: Optional[str] = None,
        semantic_threshold_percentile: float = 95.0,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> List[Chunk]:
        """
        Chunk *text* using the given strategy.

        Args:
            text:                          Text to chunk.
            strategy:                      Splitting strategy.
            max_tokens:                    Max tokens per chunk.
            overlap_tokens:                Token overlap between consecutive chunks.
            model:                         Model for token counting.
            semantic_threshold_percentile: Percentile drop threshold for SEMANTIC strategy.
            embedding_model:               HuggingFace sentence-transformer model name.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        model = model or self.default_model

        # Handle deprecation / strategy resolution
        resolved_strategy = strategy
        if strategy == ChunkStrategy.SEMANTIC or str(strategy).lower() == "semantic":
            # Check if sentence_transformers is available
            try:
                import sentence_transformers  # noqa: F401
                # True semantic chunking path
                return self._semantic_embedding_chunk(
                    text,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    model=model,
                    percentile=semantic_threshold_percentile,
                    embedding_model=embedding_model,
                )
            except ImportError:
                warnings.warn(
                    "llm-context-forge: ChunkStrategy.SEMANTIC (markdown header splitting) "
                    "has been renamed to ChunkStrategy.HEURISTIC. Falling back to HEURISTIC strategy. "
                    "Install `pip install llm-context-forge[semantic]` to enable embedding-based semantic chunking.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                resolved_strategy = ChunkStrategy.HEURISTIC

        segments = self._split_by_strategy(text, resolved_strategy)
        return self._assemble_chunks(segments, max_tokens, overlap_tokens, model)

    async def achunk(
        self,
        text: str,
        strategy: ChunkStrategy = ChunkStrategy.PARAGRAPH,
        max_tokens: int = 500,
        overlap_tokens: int = 50,
        model: Optional[str] = None,
        semantic_threshold_percentile: float = 95.0,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> List[Chunk]:
        """Async counterpart for chunking documents."""
        return await anyio.to_thread.run_sync(
            self.chunk,
            text,
            strategy,
            max_tokens,
            overlap_tokens,
            model,
            semantic_threshold_percentile,
            embedding_model,
        )

    def chunk_code(
        self,
        code: str,
        language: str = "python",
        max_tokens: int = 500,
    ) -> List[Chunk]:
        """
        Chunk source code respecting function/class boundaries.
        """
        blocks = self._split_code_blocks(code, language)
        return self._assemble_chunks(blocks, max_tokens, overlap_tokens=0)

    def chunk_markdown(
        self,
        md: str,
        max_tokens: int = 500,
        overlap_tokens: int = 50,
    ) -> List[Chunk]:
        """
        Chunk markdown respecting headings and code fences.
        """
        sections = self._split_markdown_sections(md)
        return self._assemble_chunks(sections, max_tokens, overlap_tokens)

    def merge_small_chunks(
        self,
        chunks: List[Chunk],
        min_tokens: int = 100,
    ) -> List[Chunk]:
        """
        Merge consecutive small chunks until each meets *min_tokens*.
        """
        if not chunks:
            return []

        merged: List[Chunk] = []
        buffer_text = chunks[0].text
        buffer_tokens = chunks[0].token_count

        for chunk in chunks[1:]:
            if buffer_tokens < min_tokens:
                buffer_text += "\n\n" + chunk.text
                buffer_tokens += chunk.token_count
            else:
                merged.append(Chunk(
                    text=buffer_text,
                    index=len(merged),
                    token_count=buffer_tokens,
                ))
                buffer_text = chunk.text
                buffer_tokens = chunk.token_count

        # Flush remaining buffer
        merged.append(Chunk(
            text=buffer_text,
            index=len(merged),
            token_count=buffer_tokens,
        ))

        return merged

    # ------------------------------------------------------------------
    # Strategy-based splitting
    # ------------------------------------------------------------------

    def _split_by_strategy(
        self,
        text: str,
        strategy: ChunkStrategy,
    ) -> List[str]:
        """Split text into raw segments based on strategy."""
        if strategy == ChunkStrategy.FIXED:
            return self._split_fixed(text)
        elif strategy == ChunkStrategy.SENTENCE:
            return self._split_sentences(text)
        elif strategy == ChunkStrategy.PARAGRAPH:
            return self._split_paragraphs(text)
        elif strategy in (ChunkStrategy.HEURISTIC, ChunkStrategy.SEMANTIC):
            return self._split_markdown_sections(text)
        elif strategy == ChunkStrategy.CODE:
            return self._split_code_blocks(text)
        else:
            return self._split_paragraphs(text)

    def _split_fixed(self, text: str, chars: int = 1000) -> List[str]:
        """Split into fixed-size character blocks."""
        return [text[i:i + chars] for i in range(0, len(text), chars)]

    def _split_sentences(self, text: str) -> List[str]:
        """Split on sentence boundaries."""
        parts = self._SENTENCE_SPLIT.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _split_paragraphs(self, text: str) -> List[str]:
        """Split on paragraph breaks (double newlines)."""
        parts = self._PARAGRAPH_SPLIT.split(text)
        return [p.strip() for p in parts if p.strip()]

    def _split_markdown_sections(self, text: str) -> List[str]:
        """Split markdown on headings while preserving heading text."""
        lines = text.split('\n')
        sections: List[str] = []
        current: List[str] = []

        for line in lines:
            if self._HEADING_SPLIT.match(line) and current:
                sections.append('\n'.join(current))
                current = []
            current.append(line)

        if current:
            sections.append('\n'.join(current))

        return [s.strip() for s in sections if s.strip()]

    def _split_code_blocks(
        self,
        code: str,
        language: str = "python",
    ) -> List[str]:
        """Split source code on function/class definitions."""
        lines = code.split('\n')
        blocks: List[str] = []
        current: List[str] = []

        for line in lines:
            if self._FUNC_DEF.match(line) and current:
                blocks.append('\n'.join(current))
                current = []
            current.append(line)

        if current:
            blocks.append('\n'.join(current))

        return [b for b in blocks if b.strip()]

    # ------------------------------------------------------------------
    # True Semantic Chunking (Greg Kamradt's Percentile Drop Algorithm)
    # ------------------------------------------------------------------

    def _semantic_embedding_chunk(
        self,
        text: str,
        max_tokens: int,
        overlap_tokens: int,
        model: str,
        percentile: float = 95.0,
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> List[Chunk]:
        """Embedding-based semantic chunking using cosine similarity drops."""
        from sentence_transformers import SentenceTransformer
        import numpy as np

        sentences = self._split_sentences(text)
        if len(sentences) <= 1:
            return self._assemble_chunks(sentences, max_tokens, overlap_tokens, model)

        st_model = SentenceTransformer(embedding_model)
        embeddings = st_model.encode(sentences)

        # Compute cosine similarities between adjacent sentence embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        normed = embeddings / norms

        similarities = [
            float(np.dot(normed[i], normed[i + 1]))
            for i in range(len(normed) - 1)
        ]

        # Calculate threshold percentile drop
        cutoff = np.percentile(similarities, 100.0 - percentile)

        grouped_segments: List[str] = []
        current_group: List[str] = [sentences[0]]

        for i, sim in enumerate(similarities):
            if sim < cutoff:
                grouped_segments.append(" ".join(current_group))
                current_group = [sentences[i + 1]]
            else:
                current_group.append(sentences[i + 1])

        if current_group:
            grouped_segments.append(" ".join(current_group))

        return self._assemble_chunks(grouped_segments, max_tokens, overlap_tokens, model)

    # ------------------------------------------------------------------
    # Chunk assembly
    # ------------------------------------------------------------------

    def _assemble_chunks(
        self,
        segments: List[str],
        max_tokens: int,
        overlap_tokens: int = 0,
        model: Optional[str] = None,
    ) -> List[Chunk]:
        """Assemble segments into token-bounded chunks with overlap."""
        chunks: List[Chunk] = []
        buffer: List[str] = []
        buffer_tokens = 0

        for segment in segments:
            seg_tokens = self._counter.count(segment, model)

            # If a single segment exceeds max, force-split it
            if seg_tokens > max_tokens:
                # Flush buffer first
                if buffer:
                    text = '\n\n'.join(buffer)
                    chunks.append(Chunk(
                        text=text,
                        index=len(chunks),
                        token_count=buffer_tokens,
                    ))
                    buffer, buffer_tokens = [], 0

                # Force-split the large segment
                sub_chunks = self._force_split(segment, max_tokens, model)
                for sc in sub_chunks:
                    chunks.append(Chunk(
                        text=sc,
                        index=len(chunks),
                        token_count=self._counter.count(sc, model),
                    ))
                continue

            # Would adding this segment exceed the limit?
            if buffer_tokens + seg_tokens > max_tokens:
                text = '\n\n'.join(buffer)
                chunks.append(Chunk(
                    text=text,
                    index=len(chunks),
                    token_count=buffer_tokens,
                ))

                # Keep overlap from the end of the flushed buffer
                if overlap_tokens > 0 and buffer:
                    overlap_text = self._get_overlap(text, overlap_tokens, model)
                    buffer = [overlap_text]
                    buffer_tokens = self._counter.count(overlap_text, model)
                else:
                    buffer, buffer_tokens = [], 0

            buffer.append(segment)
            buffer_tokens += seg_tokens

        # Flush remaining
        if buffer:
            text = '\n\n'.join(buffer)
            chunks.append(Chunk(
                text=text,
                index=len(chunks),
                token_count=buffer_tokens,
            ))

        return chunks

    def _force_split(
        self,
        text: str,
        max_tokens: int,
        model: Optional[str] = None,
    ) -> List[str]:
        """Force-split a text that exceeds max_tokens."""
        sentences = self._split_sentences(text)
        if len(sentences) > 1:
            return [s for s in sentences if s.strip()]

        avg_chars = int(max_tokens * 4)  # ~4 chars per token
        return [text[i:i + avg_chars] for i in range(0, len(text), avg_chars)]

    def _get_overlap(
        self,
        text: str,
        overlap_tokens: int,
        model: Optional[str] = None,
    ) -> str:
        """Extract the last *overlap_tokens* worth of text from *text*."""
        words = text.split()
        overlap_text = ""
        for word in reversed(words):
            candidate = word + " " + overlap_text if overlap_text else word
            if self._counter.count(candidate, model) > overlap_tokens:
                break
            overlap_text = candidate
        return overlap_text.strip()
