<div align="center">
  <h1>LLM Context Forge</h1>
  <p><b>Production-Grade LLMOps Infrastructure for Context Window Management</b></p>
  <p><i>Token counting · Intelligent chunking · Priority context assembly · Pricing Integrity · Framework Integrations</i></p>

  [![Documentation](https://img.shields.io/badge/docs-docs.dhruvchudasama.me-blue.svg)](https://docs.dhruvchudasama.me)
  [![PyPI](https://img.shields.io/pypi/v/llm-context-forge.svg)](https://pypi.org/project/llm-context-forge/)
  [![Python](https://img.shields.io/pypi/pyversions/llm-context-forge.svg)](https://pypi.org/project/llm-context-forge/)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

---

> **Official Documentation**: Visit [docs.dhruvchudasama.me](https://docs.dhruvchudasama.me) for comprehensive guides, API references, change records, and architecture tutorials.

---

## Why LLM Context Forge?

Every production AI application hits the same infrastructure challenges:

| Problem | Impact | LLM Context Forge Solution |
|---|---|---|
| Context window overflow | Silent failures, truncated responses | Priority-based assembly with overflow tracking & streaming |
| Inaccurate token counting | Budget overruns, dropped requests | Benchmark-verified token counting across 20+ models |
| Hardcoded stale pricing | Embarrassing cost miscalculations | Versioned YAML pricing registry (`pricing_registry.yaml`) with staleness warnings |
| Framework isolation | Custom rewrite for LangChain/LlamaIndex | First-class LangChain & LlamaIndex integrations |
| Sync-only pipelines | Thread blocking in FastAPI/async apps | Native async API (`acount`, `achunk`, `aassemble`, `stream_assemble`) |

---

## Installation

```bash
pip install llm-context-forge
```

With optional extras:
```bash
# True embedding-based semantic chunking
pip install "llm-context-forge[semantic]"

# FastAPI REST server
pip install "llm-context-forge[api]"
```

---

## Quick Start

### 1. Pricing Integrity & Verified Rates

Zero hardcoded prices in Python code. All rates are loaded from `pricing_registry.yaml` with source citations and 30-day staleness warnings:

```python
from llm_context_forge import ModelRegistry, CostCalculator

# Lookup model info with source URL and verification date
info = ModelRegistry.get("gpt-4o")
print(f"Provider: {info.provider} | Input $/1M: ${info.input_cost_per_1k * 1000:.2f}")

# Local enterprise pricing override via environment variable:
# export LLM_CONTEXT_FORGE_PRICING_FILE=/path/to/my_pricing.yaml
```

### 2. LangChain & LlamaIndex Integrations

```python
# LangChain TextSplitter & DocumentTransformer
from llm_context_forge.integrations.langchain import ContextForgeTextSplitter, ContextForgeDocumentTransformer

splitter = ContextForgeTextSplitter(model="gpt-4o", max_tokens=500)
chunks = splitter.split_text("Your document text...")

transformer = ContextForgeDocumentTransformer(model="gpt-4o")
async_docs = await transformer.atransform_documents(langchain_documents)

# LlamaIndex MetadataAware NodeParser
from llm_context_forge.integrations.llamaindex import ContextForgeNodeParser

parser = ContextForgeNodeParser(max_tokens=500)
nodes = parser.split_text_metadata_aware("def foo(): pass", metadata_str="file_type: python code")
```

### 3. Async & Streaming Context Assembly

```python
from llm_context_forge import ContextWindow, Priority

window = ContextWindow("gpt-4o")
window.add_block("System instructions...", Priority.CRITICAL, "system")
window.add_block("User query...", Priority.HIGH, "query")
window.add_block("RAG context chunk...", Priority.MEDIUM, "rag_1")

# Async assembly
prompt = await window.aassemble(max_tokens=4096)

# Streaming context assembly block-by-block
async for block in window.stream_assemble(max_tokens=4096):
    print(f"Included: {block.label} ({block.token_count} tokens)")
```

### 4. True Semantic Chunking

Using Greg Kamradt's percentile-based sentence similarity drop algorithm:

```python
from llm_context_forge import DocumentChunker, ChunkStrategy

chunker = DocumentChunker("gpt-4o")
chunks = chunker.chunk(
    long_text,
    strategy=ChunkStrategy.SEMANTIC,
    semantic_threshold_percentile=95,
    embedding_model="all-MiniLM-L6-v2"
)
```

---

## Token Counting Accuracy Benchmark

Verified across 60 test documents against official model API counts:

| Category | Test Docs | Models Tested | Error Rate | Status |
|---|---|---|---|---|
| English Prose | 20 docs | gpt-4o, claude-3.5-sonnet, gemini-1.5-pro | <0.1% | Verified |
| Code (Python/JS/SQL) | 20 docs | gpt-4o, claude-3.5-sonnet, gemini-1.5-pro | <0.2% | Verified |
| Non-English Text | 10 docs | gpt-4o, claude-3.5-sonnet, gemini-1.5-pro | <1.5% | Verified |
| Special Characters | 10 docs | gpt-4o, claude-3.5-sonnet, gemini-1.5-pro | <1.8% | Verified |

---

## Roadmap & Release Milestones

| Version | Theme | Key Deliverables | Status |
|---|---|---|---|
| **0.1.5** | Core Engine | Token counting, 5 chunking strategies, priority context, CLI | Shipped |
| **0.2.0** | Pricing Integrity | Versioned YAML registry, staleness warnings, env override | Shipped |
| **0.3.0** | Framework Integrations | LangChain & LlamaIndex splitters, CLI pricing suite | Shipped |
| **0.4.0** | Live Pricing & Accuracy | Remote pricing registry fetch, plugin loaders, true semantic chunking | Shipped |
| **0.5.0** | Async & Performance | Async API (`acount`, `achunk`, `aassemble`), streaming context | Shipped |
| **1.0.0** | Production Milestone | Docs site at [docs.dhruvchudasama.me](https://docs.dhruvchudasama.me), automated release system | Shipped |

---

## CLI Tools

```bash
# Count tokens
llm-context-forge count "Hello world" --model gpt-4o

# Manage and verify pricing
llm-context-forge pricing list
llm-context-forge pricing verify gpt-4o
llm-context-forge pricing update --remote

# Start REST server
llm-context-forge serve --port 8000
```

---

## Documentation

Full documentation, tutorial guides, change logs, and API specifications are hosted at:
[https://docs.dhruvchudasama.me](https://docs.dhruvchudasama.me)

## License

MIT — see [LICENSE](LICENSE).
