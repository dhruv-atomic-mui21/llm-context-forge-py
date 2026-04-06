<div align="center">
  <h1> LLM Context Forge</h1>
  <p><b>Production-Grade LLMOps Infrastructure for Context Window Management</b></p>
  <p><i>Deterministic token counting · Intelligent chunking · Priority-based context assembly · Cost estimation — the foundation every AI application needs.</i></p>

  [![Tests](https://github.com/dhruv-atomic-mui21/llm_context_forge/workflows/Tests/badge.svg)](https://github.com/dhruv-atomic-mui21/llm_context_forge/actions)
  [![PyPI](https://img.shields.io/pypi/v/llm_context_forge.svg)](https://pypi.org/project/llm_context_forge/)
  [![Python](https://img.shields.io/pypi/pyversions/llm_context_forge.svg)](https://pypi.org/project/llm_context_forge/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Downloads](https://img.shields.io/pypi/dm/llm_context_forge.svg)](https://pypi.org/project/llm_context_forge/)
</div>
---

> **Note**: This package is a general-purpose LLM context management toolkit and is **not related to Microsoft's LayoutLM multimodal models**.

## Why LLM Context Forge?

Every production AI application eventually hits the same infrastructure problems:

| Problem | Impact | LLM Context Forge Solution |
|---------|--------|-----------------------|
|  Context window overflow | Silent failures, truncated responses | Priority-based assembly with overflow tracking |
|  Inaccurate token counting | Budget overruns, dropped requests | Deterministic counting via tiktoken + heuristic fallbacks |
|  Naive text splitting | Broken semantics, degraded LLM reasoning | 5 chunking strategies (sentence, paragraph, semantic, code, fixed) |
|  Unpredictable API costs | Surprise bills, no cost governance | Pre-flight cost estimation across 15+ models |
|  Oversized prompts | Wasted tokens, slow responses | 4 compression strategies (extractive, truncate, middle-out, map-reduce) |

## Installation

```bash
pip install llm-context-forge
```

With API server support:
```bash
pip install "llm-context-forge[api]"
```

## Quick Start

### Token Counting

```python
from llm_context_forge import TokenCounter

counter = TokenCounter("gpt-4o")
tokens = counter.count("Hello, world!")
print(f"Tokens: {tokens}")  # Tokens: 4

# Check context window fit
fits = counter.fits_in_window("Your prompt...", reserve_output=500)

# Estimate cost before sending
cost = counter.estimate_cost("Your prompt...", direction="input")
print(f"Cost: ${cost:.6f}")
```

### Intelligent Chunking

```python
from llm_context_forge import DocumentChunker, ChunkStrategy

chunker = DocumentChunker("gpt-4o")

# Chunk respecting paragraph boundaries
chunks = chunker.chunk(
    long_document,
    strategy=ChunkStrategy.PARAGRAPH,
    max_tokens=500,
    overlap_tokens=50,
)

# Specialized chunkers
code_chunks = chunker.chunk_code(source_code, language="python")
md_chunks = chunker.chunk_markdown(readme_text)
```

### Priority-Based Context Assembly

The core pattern for RAG applications — guarantee critical context fits while gracefully dropping lower-priority content:

```python
from llm_context_forge import ContextWindow, Priority

window = ContextWindow("gpt-4o")

# System instructions — always included
window.add_block("You are a legal assistant.", Priority.CRITICAL, "system")

# User query — high priority
window.add_block("What is the statute of limitations?", Priority.HIGH, "query")

# RAG search results — included if space permits
window.add_block(search_result_1, Priority.MEDIUM, "rag_1")
window.add_block(search_result_2, Priority.LOW, "rag_2")

# Assemble: packs highest-priority blocks first
prompt = window.assemble(max_tokens=4096)

# See what was included/dropped
usage = window.usage()
print(f"Included: {usage['num_included']} blocks ({usage['included_tokens']} tokens)")
print(f"Dropped:  {usage['num_excluded']} blocks")
```

### Cost Estimation

```python
from llm_context_forge import CostCalculator

calc = CostCalculator("gpt-4o")

# Single prompt cost
cost = calc.estimate_prompt("Your prompt text here")
print(f"Input cost: ${cost.usd:.6f}")

# Compare models
comparison = calc.compare_models(
    texts=["Document 1...", "Document 2..."],
    models=["gpt-4o", "gpt-4o-mini", "claude-3.5-sonnet", "gemini-flash"],
)
for model, analysis in comparison.items():
    print(f"{model}: ${analysis.total_usd:.6f} for {analysis.total_tokens} tokens")
```

### Context Compression

```python
from llm_context_forge import ContextCompressor, CompressionStrategy

compressor = ContextCompressor("gpt-4o")

# Extractive: keeps most important sentences via TF-IDF scoring
result = compressor.compress(long_text, target_tokens=200)
print(f"Compressed: {result.original_tokens} → {result.compressed_tokens} tokens")
print(f"Savings: {result.savings_pct:.1f}%")

# Middle-out: preserves start and end, removes middle
result = compressor.compress(log_text, target_tokens=300, strategy=CompressionStrategy.MIDDLE_OUT)
```

### Conversation Management

```python
from llm_context_forge import ConversationManager

manager = ConversationManager("gpt-4o")

manager.add_message("system", "You are a helpful Python tutor.")
manager.add_message("user", "Explain decorators")
manager.add_message("assistant", "Decorators are...")
# ... many more turns ...

# Auto-trim older messages to fit budget, preserving system prompt
trimmed = manager.get_context(max_tokens=4096, preserve_system=True)
```

## Supported Models

| Provider | Models | Token Counting | Pricing |
|----------|--------|:--------------:|:-------:|
| **OpenAI** | GPT-4, GPT-4 Turbo, GPT-4o, GPT-4o-mini, GPT-3.5 Turbo | ✅ `tiktoken` | ✅ |
| **Anthropic** | Claude 3 Opus, Claude 3.5 Sonnet, Claude 3 Haiku | ✅ `anthropic` | ✅ |
| **Google** | Gemini Pro, Gemini Flash | ✅ `transformers` | ✅ |
| **Meta** | Llama 3 8B, Llama 3 70B, Llama 3.1 405B | ✅ `transformers` | ✅ |
| **Mistral** | Mistral Large | ✅ `mistral-common` | ✅ |
| **Cohere** | Command R+ | ✅ `transformers` | ✅ |

### Production-Grade Tokenizer Fallback

In production environments, external tokenizer packages (`transformers`, `mistral-common`) might fail to download or initialize due to network errors. `llm-context-forge` provides a robust, production-grade fallback:
- If a native tokenizer fails to load, the system degrades to OpenAI's fast `cl100k_base` (`tiktoken`).
- Since most modern LLMs utilize similar Byte-Pair Encoding (BPE), `cl100k_base` offers a highly accurate baseline.
- `llm-context-forge` automatically applies structural safety multipliers (e.g. `1.05x`) specifically tuned to each backend before throwing an overflow warning. 
- A one-time warning is emitted via standard python logging to notify infrastructure teams of the fallback engagement.

Register custom models:
```python
from llm_context_forge import ModelRegistry, ModelInfo, TokenizerBackend

ModelRegistry.register(ModelInfo(
    name="my-fine-tuned-model",
    backend=TokenizerBackend.OPENAI,
    context_window=16_384,
    encoding_name="cl100k_base",
    input_cost_per_1k=0.002,
    output_cost_per_1k=0.006,
))
```

## CLI

```bash
# Count tokens
llm_context_forge count "Hello world" --model gpt-4o

# Chunk a document
llm_context_forge chunk document.md --strategy semantic --max-tokens 500

# Estimate cost
llm_context_forge cost document.txt --model claude-3.5-sonnet

# List all models
llm_context_forge models

# Health check
llm_context_forge doctor

# Start API server
llm_context_forge serve --port 8000

# Interactive demo
llm_context_forge demo
```

## REST API

Start the server and access interactive docs at `http://localhost:8000/docs`:

```bash
pip install "llm_context_forge[api]"
llm_context_forge serve
```

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | System health + version |
| `/api/v1/tokens/count` | POST | Count tokens |
| `/api/v1/tokens/validate` | POST | Check context window fit |
| `/api/v1/chunks/` | POST | Chunk text |
| `/api/v1/context/assemble` | POST | Priority-based assembly |
| `/api/v1/compress/` | POST | Compress text |
| `/api/v1/cost/estimate` | POST | Estimate cost |

## Architecture

```
llm_context_forge/
├── models.py        # Model registry (15+ models, pricing, backends)
├── tokenizer.py     # Multi-provider token counter (tiktoken + heuristics)
├── chunker.py       # 5-strategy document chunker with overlap
├── context.py       # Priority-based context assembly + conversation manager
├── compressor.py    # 4-strategy compression engine (TF-IDF, middle-out, etc.)
├── cost.py          # Cost estimation engine with model comparison
├── cli/main.py      # Typer CLI with Rich output
└── api/             # FastAPI server with versioned routes
```

## Docker

```bash
docker build -t llm_context_forge .
docker-compose up
```

## Development

```bash
git clone https://github.com/dhruv-atomic-mui21/llm_context_forge.git
cd llm_context_forge
pip install -e ".[dev]"
pytest
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development workflow guidelines.

## License

MIT — see [LICENSE](LICENSE).
