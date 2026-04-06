"""
Cost Optimization Example

Shows how to compare pricing across multiple Large Language Models
for the exact same dataset to optimize LLMOps expenses.
"""

from llm_context_forge.cost import CostCalculator
from llm_context_forge.models import ModelRegistry

def main():
    # Documents we plan to process
    docs = [
        "This is an initial exploratory text about a random topic.",
        "A secondary corpus of information regarding server logs.",
        "And a third chunk containing user chat history to be summarized."
    ]

    # Initialize calculator
    calc = CostCalculator()

    # We want to compare the cost of running this batch on different models
    model_choices = ["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "gemini-flash"]

    print("--- Cost Comparison for Batch Processing ---")
    print(f"Processing {len(docs)} documents...\n")

    results = calc.compare_models(docs, model_choices)

    for model, analysis in results.items():
        info = ModelRegistry.get(model)
        print(f"Model: {model}")
        print(f"  Backend:       {info.backend.value}")
        print(f"  Total Tokens:  {analysis.total_tokens}")
        print(f"  Total USD:     ${analysis.total_usd:.6f}")
        print()

if __name__ == "__main__":
    main()
