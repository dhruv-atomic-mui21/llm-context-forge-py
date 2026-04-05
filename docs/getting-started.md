# Getting Started

## Installation

LayoutLM Forge relies on pure Python and a few essential dependencies (like `tiktoken` for OpenAI tokenization and `typer` for the CLI).

```bash
pip install layoutlm_forge
```

## Quick Start (Python)

```python
from layoutlm_forge.tokenizer import TokenCounter

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

LayoutLM Forge provides a rich CLI to test and explore its capabilities quickly.

```bash
# Count tokens
layoutlm_forge count "Hello world" --model gpt-4o

# Estimate cost
layoutlm_forge cost my_document.txt --model claude-3-opus

# Start API server
layoutlm_forge serve --port 8000
```
