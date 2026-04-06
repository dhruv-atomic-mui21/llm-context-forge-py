"""
Intelligent Document Chunker

Splits text into token-bounded chunks using multiple strategies:
  - Fixed-size (character / token)
  - Sentence-aware
  - Paragraph-aware
  - Semantic (markdown headings / code fences)
  - Code-aware (function / class boundaries)

Supports configurable overlap and automatic merging of small chunks.
"""

import re
from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


class ChunkStrategy(Enum):
    """Chunking strategy."""
    FIXED = "fixed"
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
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
    natural boundaries (sentences, paragraphs, headings, code blocks).
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
    ) -> List[Chunk]:
        """
        Chunk *text* using the given strategy.

        Args:
            text:           Text to chunk.
            strategy:       Splitting strategy.
            max_tokens:     Max tokens per chunk.
            overlap_tokens: Token overlap between consecutive chunks.
            model:          Model for token counting.

        Returns:
            List of Chunk objects.
        """
        if not text or not text.strip():
            return []

        model = model or self.default_model

        segments = self._split_by_strategy(text, strategy)
        return self._assemble_chunks(segments, max_tokens, overlap_tokens, model)

    def chunk_code(
        self,
        code: str,
        language: str = "python",
        max_tokens: int = 500,
    ) -> List[Chunk]:
        """
        Chunk source code respecting function/class boundaries.

        Args:
            code:      Source code.
            language:  Programming language hint.
            max_tokens: Max tokens per chunk.

        Returns:
            List of Chunk objects.
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

        Args:
            md:             Markdown text.
            max_tokens:     Max tokens per chunk.
            overlap_tokens: Token overlap.

        Returns:
            List of Chunk objects.
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

        Args:
            chunks:     List of chunks.
            min_tokens: Minimum token count after merging.

        Returns:
            Merged chunk list.
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
        elif strategy == ChunkStrategy.SEMANTIC:
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
        # Split by sentences first, then fall back to characters
        sentences = self._split_sentences(text)
        if len(sentences) > 1:
            return [s for s in sentences if s.strip()]

        # Character-level split as last resort
        avg_chars = int(max_tokens * 4)  # ~4 chars per token
        return [text[i:i + avg_chars] for i in range(0, len(text), avg_chars)]

    def _get_overlap(
        self,
        text: str,
        overlap_tokens: int,
        model: Optional[str] = None,
    ) -> str:
        """Extract the last *overlap_tokens* worth of text from *text*."""
        # Walk backwards from end
        words = text.split()
        overlap_text = ""
        for word in reversed(words):
            candidate = word + " " + overlap_text if overlap_text else word
            if self._counter.count(candidate, model) > overlap_tokens:
                break
            overlap_text = candidate
        return overlap_text.strip()


if __name__ == "__main__":
    print("LLM Context Forge — Document Chunker")
    print("=" * 40)
    print("Usage:")
    print("  chunker = DocumentChunker()")
    print("  chunks  = chunker.chunk(text, ChunkStrategy.PARAGRAPH, max_tokens=500)")
    print("  code_chunks = chunker.chunk_code(source, 'python', max_tokens=500)")
