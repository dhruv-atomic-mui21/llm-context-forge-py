"""
Cost Calculator Engine

Provides multi-model cost estimation with real pricing data.
Estimate token costs before sending data to LLM providers.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
from layoutlm_forge.models import ModelRegistry
from layoutlm_forge.tokenizer import TokenCounter


@dataclass
class Cost:
    """Estimated cost for a single operation."""
    usd: float
    tokens: int
    model: str


@dataclass
class ConversationCost:
    """Estimated conversation cost breakdown."""
    total_usd: float
    input_usd: float
    output_usd: float
    input_tokens: int
    output_tokens: int
    model: str


@dataclass
class BulkCostAnalysis:
    """Estimated bulk cost analysis across documents."""
    total_usd: float
    total_tokens: int
    model: str
    num_documents: int


class CostCalculator:
    """
    Multi-model cost estimation engine.

    Pre-flight cost checking before making expensive API calls.
    Supports single prompts, conversations, bulk documents, and
    cross-model comparisons with real provider pricing.
    """

    def __init__(self, default_model: str = "gpt-4o"):
        self.default_model = default_model
        self.counter = TokenCounter(default_model)

    def estimate_prompt(
        self,
        tokens_or_text: Union[str, int],
        model: Optional[str] = None,
    ) -> Cost:
        """
        Estimate cost for prompt (input) tokens.

        Args:
            tokens_or_text: Raw text or pre-counted token integer.
            model:          Model name.

        Returns:
            Cost dataclass with USD amount, token count, and model name.
        """
        m = model or self.default_model
        info = ModelRegistry.get(m)

        if isinstance(tokens_or_text, str):
            tokens = self.counter.count(tokens_or_text, m)
        else:
            tokens = tokens_or_text

        usd = (tokens / 1000) * info.input_cost_per_1k
        return Cost(usd=usd, tokens=tokens, model=m)

    def estimate_completion(
        self,
        tokens_or_text: Union[str, int],
        model: Optional[str] = None,
    ) -> Cost:
        """
        Estimate cost for completion (output) tokens.

        Args:
            tokens_or_text: Raw text or pre-counted token integer.
            model:          Model name.

        Returns:
            Cost dataclass.
        """
        m = model or self.default_model
        info = ModelRegistry.get(m)

        if isinstance(tokens_or_text, str):
            tokens = self.counter.count(tokens_or_text, m)
        else:
            tokens = tokens_or_text

        usd = (tokens / 1000) * info.output_cost_per_1k
        return Cost(usd=usd, tokens=tokens, model=m)

    def estimate_conversation(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        assumed_output_tokens: int = 500,
    ) -> ConversationCost:
        """
        Estimate cost of an entire conversation including assumed output.

        Args:
            messages:              ChatML message list.
            model:                 Model name.
            assumed_output_tokens: Expected output length.

        Returns:
            ConversationCost with input/output breakdown.
        """
        m = model or self.default_model
        input_tokens = self.counter.count_messages(messages, m)

        input_cost = self.estimate_prompt(input_tokens, m).usd
        output_cost = self.estimate_completion(assumed_output_tokens, m).usd

        return ConversationCost(
            total_usd=input_cost + output_cost,
            input_usd=input_cost,
            output_usd=output_cost,
            input_tokens=input_tokens,
            output_tokens=assumed_output_tokens,
            model=m,
        )

    def bulk_estimate(
        self,
        documents: List[str],
        model: Optional[str] = None,
    ) -> BulkCostAnalysis:
        """
        Analyse cost implications of processing multiple documents.

        Args:
            documents: List of document texts.
            model:     Model name.

        Returns:
            BulkCostAnalysis with totals.
        """
        m = model or self.default_model
        total_tokens = sum(self.counter.count(doc, m) for doc in documents)
        total_usd = self.estimate_prompt(total_tokens, m).usd

        return BulkCostAnalysis(
            total_usd=total_usd,
            total_tokens=total_tokens,
            model=m,
            num_documents=len(documents),
        )

    def compare_models(
        self,
        texts: List[str],
        models: List[str],
    ) -> Dict[str, BulkCostAnalysis]:
        """
        Compare costs across multiple models for the same dataset.

        Args:
            texts:  List of texts to process.
            models: List of model names to compare.

        Returns:
            Dict mapping model name to BulkCostAnalysis.
        """
        return {m: self.bulk_estimate(texts, model=m) for m in models}
