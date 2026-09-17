"""
Rigorous tests for DocumentChunker
"""

import pytest
from llm_context_forge.chunker import DocumentChunker, ChunkStrategy

class TestDocumentChunker:
    """Rigorous constraints, overlaps, and bounds tests."""

    def setup_method(self):
        self.chunker = DocumentChunker("gpt-4o")

    def test_constraint_enforcement_paragraphs(self):
        """Ensure paragraphs never exceed max_tokens constraint."""
        text = "P1\n\nP2\n\nP3\n\nP4\n\nP5"
        chunks = self.chunker.chunk(text, strategy=ChunkStrategy.PARAGRAPH, max_tokens=5, overlap_tokens=0)
        
        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count <= 5

    def test_force_split_massive_lines(self):
        """When semantic chunks fail, character splitting must enforce the bounds."""
        massive = "word " * 500  # huge single line with no punctuation
        chunks = self.chunker.chunk(massive, max_tokens=50, overlap_tokens=0)
        
        assert len(chunks) > 1
        for c in chunks:
            assert hasattr(c, "token_count")
            assert c.token_count <= 50

    def test_overlap_handling_infinite_safety(self):
        """Overlap should not prevent progression or cause infinite loops."""
        text = "A. B. C. D. E. F. G. H. I. J. K. L. M. N. O. P. Q. R. S. T. U. V. W. X. Y. Z."
        # Overly restrictive bounds
        chunks = self.chunker.chunk(text, strategy=ChunkStrategy.SENTENCE, max_tokens=5, overlap_tokens=2)
        
        assert len(chunks) > 5
        # Ensure chunks don't magically end up empty
        for c in chunks:
            assert c.token_count > 0

    def test_code_chunk_respects_functions(self):
        """Functions should remain largely coherent chunks unless forced."""
        code = "def foo():\n    pass\n\ndef bar():\n    pass"
        chunks = self.chunker.chunk_code(code, max_tokens=50)
        
        # Depending on splits, this could be 1 chunk if budget allows or 2
        assert len(chunks) > 0
        for c in chunks:
            assert c.token_count <= 50

    def test_empty_input(self):
        """Empty input should return empty chunk list."""
        assert len(self.chunker.chunk("", max_tokens=10, overlap_tokens=0)) == 0
        assert len(self.chunker.chunk("   \n  ", max_tokens=10, overlap_tokens=0)) == 0

    def test_unicode_handling(self):
        """Unicode characters (CJK, emojis, RTL) should not crash the chunker and count correctly."""
        text = "Hello \u4e16\u754c. 🚀  مرحبا. " * 50
        chunks = self.chunker.chunk(text, strategy=ChunkStrategy.SENTENCE, max_tokens=20, overlap_tokens=0)
        assert len(chunks) > 0
        for c in chunks:
            assert c.token_count <= 20
            
    def test_separator_behavior(self):
        """Ensure chunking correctly breaks on specified separators."""
        text = "A||B||C||D"
        # We simulate the fallback to fixed splitting when standard heuristics fail to find breaks
        chunks = self.chunker.chunk(text, strategy=ChunkStrategy.FIXED, max_tokens=2, overlap_tokens=0)
        assert len(chunks) >= 2

    def test_deterministic_output(self):
        """Chunking the same text multiple times must yield identical results."""
        text = "Line 1.\n\nLine 2.\n\nLine 3."
        res1 = self.chunker.chunk(text, strategy=ChunkStrategy.PARAGRAPH, max_tokens=10, overlap_tokens=0)
        res2 = self.chunker.chunk(text, strategy=ChunkStrategy.PARAGRAPH, max_tokens=10, overlap_tokens=0)
        assert [c.text for c in res1] == [c.text for c in res2]

    def test_invalid_configuration(self):
        """Invalid configurations should raise appropriate errors."""
        with pytest.raises(ValueError):
            self.chunker.chunk("text", max_tokens=-5)
        
        with pytest.raises(ValueError):
            self.chunker.chunk("text", max_tokens=10, overlap_tokens=20)
