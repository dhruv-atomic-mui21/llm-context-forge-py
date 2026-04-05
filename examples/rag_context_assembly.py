"""
RAG Context Assembly Example

Shows how to pack context dynamically based on priorities,
guaranteeing system instructions fit while dropping lower priority RAG chunks if necessary.
"""

from layoutlm_forge.context import ContextWindow, Priority

def main():
    window = ContextWindow("gpt-4-turbo")

    # 1. Critical instructions (must be present)
    window.add_block(
        "You are an expert Q&A engine. Base your answers strictly on the provided context.",
        priority=Priority.CRITICAL,
        label="system_prompt"
    )

    # 2. User's actual question (should definitely be included)
    window.add_block(
        "User question: Explain how LayoutLM Forge calculates costs.",
        priority=Priority.HIGH,
        label="user_query"
    )

    # 3. Dynamic RAG snippets (might not fit depending on length)
    # Pretend these were retrieved from a vector database
    rag_snippets = [
        "Snippet 1: The CostCalculator engine is in layoutlm_forge/cost.py",
        "Snippet 2: " + ("It takes model pricing from ModelRegistry... " * 50),
        "Snippet 3: Supported models include GPT-4o, Claude Opus.",
    ]

    for i, snip in enumerate(rag_snippets):
        # Medium/Low priority so they get dropped if token limits are exceeded
        window.add_block(snip, priority=Priority.MEDIUM, label=f"rag_retrieval_{i}")

    # Let's pretend we have a very small token budget for testing (e.g. 200 tokens)
    budget = 200
    assembled = window.assemble(max_tokens=budget)

    print("--- Final Assembled RAG Prompt ---")
    print(assembled)
    print("\n--- Statistics ---")
    usage = window.usage()
    print(f"Blocks included: {usage['num_included']}")
    print(f"Blocks excluded: {usage['num_excluded']} (due to budget limit)")

if __name__ == "__main__":
    main()
