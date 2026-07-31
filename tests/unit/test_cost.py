"""
Rigorous tests for CostCalculator.
"""

import pytest
from llm_context_forge.cost import CostCalculator
from llm_context_forge.models import ModelRegistry

class TestCostCalculator:
    """Concrete mathematical logic validators for billing operations."""

    def setup_method(self):
        self.calc = CostCalculator("gpt-4o")

    def test_mathematical_validity_inputs(self):
        """Test math engine determinism based on manual knowns."""
        info = ModelRegistry.get("gpt-4o")
        assert info.input_cost_per_1k == 0.0025
        
        # 100k input tokens = $0.25
        cost = self.calc.estimate_prompt(100_000, "gpt-4o")
        assert abs(cost.usd - 0.25) < 0.0001
        
    def test_mathematical_validity_outputs(self):
        cost = self.calc.estimate_completion(100_000, "gpt-4o")
        # 100k generated tokens = $1.00
        assert abs(cost.usd - 1.00) < 0.0001
        
    def test_conversation_aggregates(self):
        msgs = [{"role": "user", "content": "Test"}] # roughly 4 tokens input
        cost = self.calc.estimate_conversation(msgs, assumed_output_tokens=500)
        
        # 500 outputs * 0.010 / 1000 = 0.005
        assert cost.input_usd > 0
        assert abs(cost.output_usd - 0.005) < 0.0001
        assert abs(cost.total_usd - (cost.input_usd + cost.output_usd)) < 0.000001

