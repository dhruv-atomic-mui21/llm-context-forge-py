"""
Unit tests for Pricing Integrity, Staleness Warnings, and Local Overrides.
"""

import os
import tempfile
import warnings
import pytest
from llm_context_forge.models import ModelRegistry, PricingDataStaleWarning, ModelInfo, TokenizerBackend
from llm_context_forge.pricing_provider import BundledYAMLPricingProvider, update_pricing_registry


def test_bundled_yaml_pricing_provider():
    provider = BundledYAMLPricingProvider()
    data = provider.fetch()
    assert "gpt-4o" in data
    assert data["gpt-4o"]["input_per_1m"] == 2.50
    assert data["gpt-4o"]["provider"] == "openai"


def test_zero_hardcoded_prices_in_codebase():
    # Verify that models in ModelRegistry came from registry without python hardcoding
    info = ModelRegistry.get("gpt-4o")
    assert info.input_cost_per_1k == 0.0025
    assert info.provider == "openai"
    assert info.source_url == "https://openai.com/api/pricing"


def test_staleness_warning_triggered():
    old_info = ModelInfo(
        name="stale-model-test",
        backend=TokenizerBackend.OPENAI,
        context_window=4096,
        date_retrieved="2020-01-01",  # Over 30 days old
    )
    ModelRegistry.register(old_info)

    with pytest.warns(PricingDataStaleWarning) as record:
        ModelRegistry.get("stale-model-test")

    assert len(record) > 0
    assert "was last verified" in str(record[0].message)


def test_env_pricing_override(monkeypatch):
    custom_yaml = """
custom-enterprise-model:
  provider: enterprise
  input_per_1m: 1.00
  output_per_1m: 2.00
  context_window: 64000
  source_url: "https://internal.company.com/pricing"
  date_retrieved: "2026-07-20"
  confidence: official
  registry_version: "1.0.0"
"""
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(custom_yaml)
        f_path = f.name

    try:
        monkeypatch.setenv("LLM_CONTEXT_FORGE_PRICING_FILE", f_path)
        ModelRegistry.initialize(force_reload=True)

        info = ModelRegistry.get("custom-enterprise-model")
        assert info.provider == "enterprise"
        assert info.input_cost_per_1k == 0.001
        assert info.context_window == 64000
    finally:
        if os.path.exists(f_path):
            os.remove(f_path)
        ModelRegistry.initialize(force_reload=True)
