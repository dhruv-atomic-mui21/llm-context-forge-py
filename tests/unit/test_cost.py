"""
Tests for CostCalculator
"""
import pytest
from contextforge.cost import CostCalculator, Cost, ConversationCost, BulkCostAnalysis

class TestCostCalculator:
    def setup_method(self):
        self.calc = CostCalculator("gpt-4o")
        self.text = "Hello world, this is a test snippet."

    def test_estimate_prompt(self):
        cost = self.calc.estimate_prompt(self.text)
        assert isinstance(cost, Cost)
        assert cost.tokens > 0
        assert cost.usd > 0
        assert cost.model == "gpt-4o"

    def test_estimate_completion(self):
        cost = self.calc.estimate_completion(self.text)
        assert isinstance(cost, Cost)
        assert cost.tokens > 0
        assert cost.usd > 0
        
    def test_estimate_conversation(self):
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        cost = self.calc.estimate_conversation(messages)
        assert isinstance(cost, ConversationCost)
        assert cost.input_tokens > 0
        assert cost.total_usd > 0
        assert cost.input_usd > 0
        assert cost.output_usd > 0

    def test_bulk_estimate(self):
        docs = ["Doc 1", "Doc 2", "Doc 3"]
        cost = self.calc.bulk_estimate(docs)
        assert isinstance(cost, BulkCostAnalysis)
        assert cost.num_documents == 3
        assert cost.total_tokens > 0
        assert cost.total_usd > 0

    def test_compare_models(self):
        docs = ["Hello world"]
        comparison = self.calc.compare_models(docs, ["gpt-4o", "claude-3.5-sonnet"])
        assert "gpt-4o" in comparison
        assert "claude-3.5-sonnet" in comparison
        assert comparison["gpt-4o"].total_tokens > 0
