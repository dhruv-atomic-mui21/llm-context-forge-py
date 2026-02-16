"""
Tests for DocumentChunker
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chunker import DocumentChunker, ChunkStrategy, Chunk


class TestChunkCreation:
    """Tests for Chunk dataclass."""

    def test_chunk_char_count(self):
        c = Chunk(text="Hello world", index=0, token_count=2)
        assert c.char_count == 11

    def test_chunk_metadata_default(self):
        c = Chunk(text="test", index=0, token_count=1)
        assert c.metadata == {}


class TestDocumentChunker:
    """Tests for DocumentChunker."""

    def setup_method(self):
        self.chunker = DocumentChunker("gpt-4o")

    def test_chunk_empty_text(self):
        chunks = self.chunker.chunk("")
        assert chunks == []

    def test_chunk_whitespace_only(self):
        chunks = self.chunker.chunk("   \n\n   ")
        assert chunks == []

    def test_chunk_short_text_single_chunk(self):
        chunks = self.chunker.chunk("Hello world.", max_tokens=500)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."

    def test_chunk_paragraph_strategy(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunks = self.chunker.chunk(text, ChunkStrategy.PARAGRAPH, max_tokens=500)
        assert len(chunks) >= 1

    def test_chunk_sentence_strategy(self):
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        chunks = self.chunker.chunk(text, ChunkStrategy.SENTENCE, max_tokens=10)
        assert len(chunks) >= 1

    def test_chunk_fixed_strategy(self):
        text = "x" * 5000
        chunks = self.chunker.chunk(text, ChunkStrategy.FIXED, max_tokens=100)
        assert len(chunks) > 1

    def test_chunk_indices_sequential(self):
        text = "Para one.\n\nPara two.\n\nPara three.\n\nPara four."
        chunks = self.chunker.chunk(text, ChunkStrategy.PARAGRAPH, max_tokens=10)
        for i, chunk in enumerate(chunks):
            assert chunk.index == i

    def test_chunk_respects_max_tokens(self):
        text = "word " * 1000
        chunks = self.chunker.chunk(text, ChunkStrategy.SENTENCE, max_tokens=50)
        for chunk in chunks:
            # Allow some tolerance for edge cases
            assert chunk.token_count <= 60

    def test_chunk_code(self):
        code = """
def foo():
    return 1

def bar():
    return 2

class Baz:
    def method(self):
        return 3
"""
        chunks = self.chunker.chunk_code(code, "python", max_tokens=500)
        assert len(chunks) >= 1

    def test_chunk_markdown(self):
        md = """# Heading 1

Some content here.

## Heading 2

More content here.

## Heading 3

Even more content.
"""
        chunks = self.chunker.chunk_markdown(md, max_tokens=500)
        assert len(chunks) >= 1


class TestMergeSmallChunks:
    """Tests for chunk merging."""

    def setup_method(self):
        self.chunker = DocumentChunker("gpt-4o")

    def test_merge_empty_list(self):
        assert self.chunker.merge_small_chunks([]) == []

    def test_merge_all_small(self):
        chunks = [
            Chunk(text="a", index=0, token_count=1),
            Chunk(text="b", index=1, token_count=1),
            Chunk(text="c", index=2, token_count=1),
        ]
        merged = self.chunker.merge_small_chunks(chunks, min_tokens=5)
        assert len(merged) == 1  # All merged into one

    def test_merge_preserves_large(self):
        chunks = [
            Chunk(text="large block " * 50, index=0, token_count=100),
            Chunk(text="small", index=1, token_count=1),
        ]
        merged = self.chunker.merge_small_chunks(chunks, min_tokens=10)
        assert len(merged) == 2

    def test_merge_reindexes(self):
        chunks = [
            Chunk(text="a", index=0, token_count=10),
            Chunk(text="b", index=1, token_count=10),
        ]
        merged = self.chunker.merge_small_chunks(chunks, min_tokens=5)
        for i, c in enumerate(merged):
            assert c.index == i


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
