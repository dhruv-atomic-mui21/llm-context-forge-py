"""
Tests for ContextCompressor
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.compressor import ContextCompressor, CompressionStrategy, CompressionResult


class TestCompressionResult:
    """Tests for CompressionResult dataclass."""

    def test_ratio_calculation(self):
        r = CompressionResult("hi", 100, 50, "extractive")
        assert r.ratio == 0.5

    def test_ratio_zero_original(self):
        r = CompressionResult("", 0, 0, "extractive")
        assert r.ratio == 1.0

    def test_savings_pct(self):
        r = CompressionResult("hi", 100, 25, "extractive")
        assert r.savings_pct == 75.0


class TestContextCompressor:
    """Tests for ContextCompressor."""

    def setup_method(self):
        self.compressor = ContextCompressor("gpt-4o")
        self.long_text = (
            "The quick brown fox jumps over the lazy dog. "
            "Machine learning is transforming the technology landscape. "
            "Natural language processing enables computers to understand text. "
            "Deep learning models require large amounts of training data. "
            "Transformers have revolutionized the field of NLP. "
            "Attention mechanisms allow models to focus on relevant parts. "
            "Pre-trained models can be fine-tuned for specific tasks. "
            "Context windows limit the amount of text models can process. "
            "Token counting is essential for managing API costs. "
            "Prompt engineering helps get better results from LLMs. "
        ) * 5

    def test_compress_already_fits(self):
        result = self.compressor.compress("Hello", target_tokens=1000)
        assert result.text == "Hello"
        assert result.ratio == 1.0

    def test_compress_extractive(self):
        result = self.compressor.compress(
            self.long_text, target_tokens=50,
            strategy=CompressionStrategy.EXTRACTIVE,
        )
        assert result.compressed_tokens <= 60  # some tolerance
        assert result.ratio < 1.0
        assert result.savings_pct > 0

    def test_compress_truncate(self):
        result = self.compressor.compress(
            self.long_text, target_tokens=50,
            strategy=CompressionStrategy.TRUNCATE,
        )
        assert result.compressed_tokens <= 50
        assert result.original_tokens > 50

    def test_compress_middle_out(self):
        result = self.compressor.compress(
            self.long_text, target_tokens=80,
            strategy=CompressionStrategy.MIDDLE_OUT,
        )
        assert "[... middle content removed" in result.text
        assert result.compressed_tokens <= 90

    def test_compress_map_reduce(self):
        result = self.compressor.compress(
            self.long_text, target_tokens=80,
            strategy=CompressionStrategy.MAP_REDUCE,
        )
        assert result.compressed_tokens <= 90
        assert result.ratio < 1.0

    def test_compress_empty_text(self):
        result = self.compressor.compress("", target_tokens=100)
        assert result.text == ""


class TestExtractKeySentences:
    """Tests for key sentence extraction."""

    def setup_method(self):
        self.compressor = ContextCompressor("gpt-4o")

    def test_extract_from_short_text(self):
        text = "One sentence. Two sentence."
        sentences = self.compressor.extract_key_sentences(text, n=5)
        assert len(sentences) == 2  # fewer than n

    def test_extract_correct_count(self):
        text = (
            "First sentence is here. Second sentence follows. "
            "Third sentence appears. Fourth sentence exists. "
            "Fifth sentence concludes."
        )
        sentences = self.compressor.extract_key_sentences(text, n=3)
        assert len(sentences) == 3

    def test_extract_preserves_order(self):
        text = "Alpha. Beta. Gamma. Delta. Epsilon."
        sentences = self.compressor.extract_key_sentences(text, n=2)
        # Sentences should be in original order
        original = ["Alpha.", "Beta.", "Gamma.", "Delta.", "Epsilon."]
        indices = [original.index(s) for s in sentences if s in original]
        assert indices == sorted(indices)


class TestMiddleOut:
    """Tests for middle-out compression."""

    def setup_method(self):
        self.compressor = ContextCompressor("gpt-4o")

    def test_short_text_unchanged(self):
        text = "Short text."
        result = self.compressor.middle_out(text, max_tokens=100)
        assert result == text

    def test_long_text_has_marker(self):
        text = "word " * 500
        result = self.compressor.middle_out(text, max_tokens=30)
        assert "[... middle content removed" in result


class TestConversationCompression:
    """Tests for conversation compression."""

    def setup_method(self):
        self.compressor = ContextCompressor("gpt-4o")

    def test_empty_conversation(self):
        result = self.compressor.compress_conversation([], target_tokens=100)
        assert result == []

    def test_short_conversation_unchanged(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ]
        result = self.compressor.compress_conversation(messages, target_tokens=10000)
        assert len(result) == 2

    def test_long_conversation_compressed(self):
        messages = [
            {"role": "system", "content": "You are helpful."},
        ]
        for i in range(20):
            messages.append({"role": "user", "content": f"Long message {i}. " * 30})
            messages.append({"role": "assistant", "content": f"Response {i}. " * 30})

        result = self.compressor.compress_conversation(
            messages, target_tokens=200, preserve_recent=2,
        )
        # Should have summary + recent messages
        assert len(result) < len(messages)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
