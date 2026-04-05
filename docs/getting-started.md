# Getting Started

## Installation

ContextForge relies on pure Python and a few essential dependencies (like `tiktoken` for OpenAI tokenization and `typer` for the CLI).

```bash
pip install contextforge
```

## Quick Start (Python)

```python
from contextforge.tokenizer import TokenCounter

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

ContextForge provides a rich CLI to test and explore its capabilities quickly.

```bash
# Count tokens
contextforge count "Hello world" --model gpt-4o

# Estimate cost
contextforge cost my_document.txt --model claude-3-opus

# Start API server
contextforge serve --port 8000
```
