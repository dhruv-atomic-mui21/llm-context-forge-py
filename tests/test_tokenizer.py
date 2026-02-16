"""
Tests for TokenCounter and ModelRegistry
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer import TokenCounter, TokenizerBackend, ModelRegistry, ModelInfo


class TestModelRegistry:
    """Tests for the model registry."""

    def test_list_models_not_empty(self):
        models = ModelRegistry.list_models()
        assert len(models) > 0

    def test_get_known_model(self):
        info = ModelRegistry.get("gpt-4o")
        assert info.name == "gpt-4o"
        assert info.backend == TokenizerBackend.OPENAI
        assert info.context_window == 128_000

    def test_get_unknown_model_returns_estimate(self):
        info = ModelRegistry.get("nonexistent-model-xyz")
        assert info.backend == TokenizerBackend.ESTIMATE
        assert info.context_window == 4_096

    def test_prefix_match(self):
        info = ModelRegistry.get("gpt-4o-2024-08-06")
        assert info.name == "gpt-4o"

    def test_register_custom_model(self):
        custom = ModelInfo(
            name="my-custom-model",
            backend=TokenizerBackend.ESTIMATE,
            context_window=32_000,
        )
        ModelRegistry.register(custom)
        info = ModelRegistry.get("my-custom-model")
        assert info.context_window == 32_000

    def test_anthropic_models_exist(self):
        info = ModelRegistry.get("claude-3.5-sonnet")
        assert info.backend == TokenizerBackend.ANTHROPIC
        assert info.context_window == 200_000

    def test_google_models_exist(self):
        info = ModelRegistry.get("gemini-pro")
        assert info.backend == TokenizerBackend.GOOGLE

    def test_llama_models_exist(self):
        info = ModelRegistry.get("llama-3-8b")
        assert info.backend == TokenizerBackend.LLAMA


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
        assert counter.fits_in_window("Hello", max_tokens=1) is False

    def test_fits_in_window_with_reserve(self):
        counter = TokenCounter("gpt-4o")
        assert counter.fits_in_window("Hello", max_tokens=5, reserve_output=4) is False

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
