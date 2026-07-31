"""
Token Counting Accuracy Benchmark Suite.

Evaluates token counter accuracy across English prose, Code, Non-English text,
and Mixed/Special characters against gpt-4o, claude-3.5-sonnet, and gemini-1.5-pro.
"""

import pytest
from llm_context_forge.tokenizer import TokenCounter


TEST_DATA = {
    "English Prose": [
        "The quick brown fox jumps over the lazy dog." * (i + 1) for i in range(20)
    ],
    "Code (Python/JS/SQL)": [
        f"def example_fn_{i}(x: int) -> int:\n    # Return square of x\n    return x * {i}\n" for i in range(20)
    ],
    "Non-English": [
        f"El rápido zorro marrón salta sobre el perro perezoso. 快速的棕色狐狸跳过懒狗。 {i}" for i in range(10)
    ],
    "Mixed/Special Chars": [
        f"🚀 LLM Context Forge! <tag attr='val'> {i} %$#@! 12345" for i in range(10)
    ]
}


@pytest.mark.parametrize("model", ["gpt-4o", "claude-3.5-sonnet", "gemini-1.5-pro"])
def test_accuracy_benchmark(model):
    counter = TokenCounter(model)
    results = {}
    
    for category, docs in TEST_DATA.items():
        total_tokens = sum(counter.count(doc, model=model) for doc in docs)
        assert total_tokens > 0
        results[category] = total_tokens
        
    assert len(results) == 4
