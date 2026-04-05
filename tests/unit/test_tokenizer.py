"""
Tests for TokenCounter
"""

import pytest
from contextforge.tokenizer import TokenCounter

class TestTokenCounter:
    """Tests for TokenCounter."""

    def test_count_empty_string(self):
        counter = TokenCounter("gpt-4o")
        assert counter.count("") == 0

    def test_count_returns_positive(self):
        counter = TokenCounter("gpt-4o")
        tokens = counter.count("Hello world, this is a test.")
        assert tokens > 0

    def test_count_longer_text_more_tokens(self):
        counter = TokenCounter("gpt-4o")
        short = counter.count("hi")
        long = counter.count("This is a significantly longer sentence for testing.")
        assert long > short

    def test_count_with_anthropic_estimate(self):
        counter = TokenCounter("claude-3.5-sonnet")
        tokens = counter.count("Hello world, this is a test.")
        assert tokens > 0

    def test_count_with_estimate_backend(self):
        counter = TokenCounter("unknown-model")
        tokens = counter.count("Test text here.")
        assert tokens > 0

    def test_count_messages(self):
        counter = TokenCounter("gpt-4o")
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]
        tokens = counter.count_messages(messages)
        assert tokens > 0

    def test_count_messages_empty(self):
        counter = TokenCounter("gpt-4o")
        tokens = counter.count_messages([])
        assert tokens == 3  # base overhead

    def test_fits_in_window_small_text(self):
        counter = TokenCounter("gpt-4o")
        assert counter.fits_in_window("Hello") is True

    def test_fits_in_window_with_limit(self):
        counter = TokenCounter("gpt-4o")
        assert counter.fits_in_window("Hello", max_tokens=0) is False

    def test_fits_in_window_with_reserve(self):
        counter = TokenCounter("gpt-4o")
        assert counter.fits_in_window("Hello", max_tokens=4, reserve_output=4) is False

    def test_truncate_short_text_unchanged(self):
        counter = TokenCounter("gpt-4o")
        text = "Hi"
        result = counter.truncate_to_fit(text, max_tokens=100)
        assert result == text

    def test_truncate_long_text(self):
        counter = TokenCounter("gpt-4o")
        text = "word " * 1000
        result = counter.truncate_to_fit(text, max_tokens=20)
        assert counter.count(result) <= 20

    def test_estimate_cost(self):
        counter = TokenCounter("gpt-4o")
        cost = counter.estimate_cost("Hello world!")
        assert cost >= 0.0

    def test_estimate_cost_output(self):
        counter = TokenCounter("gpt-4o")
        input_cost = counter.estimate_cost("Hello", direction="input")
        output_cost = counter.estimate_cost("Hello", direction="output")
        # Output typically costs more
        assert output_cost >= input_cost

    def test_get_model_info(self):
        counter = TokenCounter("gpt-4o")
        info = counter.get_model_info()
        assert info.name == "gpt-4o"
        assert info.context_window == 128_000
        
    def test_count_batch(self):
        counter = TokenCounter("gpt-4o")
        counts = counter.count_batch(["hi", "hello world!"])
        assert len(counts) == 2
        assert counts[0] > 0
        assert counts[1] > counts[0]
        
    def test_count_with_warnings(self):
        counter = TokenCounter("gpt-4o")
        res = counter.count_with_warnings("hi", warn_threshold=0.0)
        assert len(res["warnings"]) > 0
