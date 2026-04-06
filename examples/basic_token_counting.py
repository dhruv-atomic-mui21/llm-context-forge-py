"""
Basic Token Counting Example

Shows how to count tokens and estimate capacity/cost for a given text.
"""

from llm_context_forge.tokenizer import TokenCounter

def main():
    target_model = "gpt-4o"
    counter = TokenCounter(target_model)

    sample_text = (
        "LLM Context Forge is a production-grade library for LLMOps. "
        "It provides modular solutions for managing context windows."
    )

    # 1. Count tokens
    tokens = counter.count(sample_text)
    print(f"Token count for '{target_model}': {tokens}")

    # 2. Check fit inside window
    info = counter.get_model_info()
    fits = counter.fits_in_window(sample_text)
    print(f"Fits in context window ({info.context_window}): {fits}")

    # 3. Cost estimation
    cost = counter.estimate_cost(sample_text, direction="input")
    print(f"Estimated cost: ${cost:.6f}")

if __name__ == "__main__":
    main()
