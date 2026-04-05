"""
Tests for ContextCompressor
"""

import pytest
from layoutlm_forge.compressor import ContextCompressor, CompressionStrategy, CompressionResult

class TestCompressionResult:
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
    def setup_method(self):
        self.compressor = ContextCompressor("gpt-4o")
        from tests.fixtures.sample_data import LONG_TEXT
        self.long_text = LONG_TEXT

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
    def setup_method(self):
        self.compressor = ContextCompressor("gpt-4o")

    def test_extract_from_short_text(self):
        text = "One sentence. Two sentence."
        sentences = self.compressor.extract_key_sentences(text, n=5)
        assert len(sentences) == 2

    def test_extract_correct_count(self):
        text = (
            "First sentence is here. Second sentence follows. "
            "Third sentence appears. Fourth sentence exists. "
            "Fifth sentence concludes."
        )
        sentences = self.compressor.extract_key_sentences(text, n=3)
        assert len(sentences) == 3

class TestMiddleOut:
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
        assert len(result) < len(messages)
