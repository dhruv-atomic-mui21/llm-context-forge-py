# Getting Started

## Installation

LLM Context Forge relies on pure Python and a few essential dependencies (like `tiktoken` for OpenAI tokenization and `typer` for the CLI).

```bash
pip install llm_context_forge
```

## Quick Start (Python)

```python
from llm_context_forge.tokenizer import TokenCounter

counter = TokenCounter("gpt-4o")

# Count tokens
tokens = counter.count("Hello, world!")
print(f"Tokens: {tokens}")

# Estimate cost
cost = counter.estimate_cost("Your prompt", model="gpt-4o", direction="input")
print(f"Cost: ${cost:.6f}")
```

For more Python examples, please check the `examples/` folder in the repository!

## Quick Start (CLI)

LLM Context Forge provides a rich CLI to test and explore its capabilities quickly.

```bash
# Count tokens
llm_context_forge count "Hello world" --model gpt-4o

# Estimate cost
llm_context_forge cost my_document.txt --model claude-3-opus

# Start API server
llm_context_forge serve --port 8000
```
