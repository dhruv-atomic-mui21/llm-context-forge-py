"""
Tests for ModelRegistry
"""

import pytest
from llm_context_forge.models import ModelRegistry, TokenizerBackend, ModelInfo

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
        assert info.backend == TokenizerBackend.HUGGINGFACE
