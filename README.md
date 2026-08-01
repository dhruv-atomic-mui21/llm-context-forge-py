<div align="center">
  <h1>LLM Context Forge</h1>
  <p><b>Production-Grade Context Window Infrastructure for LLM Applications</b></p>
  <p><i>Token counting · Intelligent chunking · Priority context assembly · Cost estimation · Framework integrations</i></p>

  [![PyPI](https://img.shields.io/pypi/v/llm-context-forge.svg?style=flat-square&color=e63946)](https://pypi.org/project/llm-context-forge/)
  [![npm](https://img.shields.io/npm/v/llm-context-forge.svg?style=flat-square&color=e63946)](https://www.npmjs.com/package/llm-context-forge)
  [![Documentation](https://img.shields.io/badge/docs-docs.dhruvchudasama.me-blue.svg?style=flat-square)](https://docs.dhruvchudasama.me)
  [![Velox Ecosystem](https://img.shields.io/badge/site-velox.satyaneev.me-emerald.svg?style=flat-square)](https://velox.satyaneev.me)
  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
</div>

---

> **Production-grade context window management: token counting, chunking, compression, and priority context packing for LLM apps.**
>
> 📌 **Official Documentation Hub**: [docs.dhruvchudasama.me](https://docs.dhruvchudasama.me)  
> ⚡ **PyPI Package**: [pypi.org/project/llm-context-forge](https://pypi.org/project/llm-context-forge/)  
> 🔗 **Product Suite**: [velox.satyaneev.me](https://velox.satyaneev.me)

---

## When to use `llm-context-forge` vs Alternatives

Most developers start with raw token counters or naive chunking until prompts drop silently or context limits crash their API calls in production:

| Scenario | Raw `tiktoken` / Naive `len(text)//4` | LangChain / LlamaIndex Defaults | `llm-context-forge` |
|---|---|---|---|
| **Token Accuracy** | `len()/4` is off by 15–30%; `tiktoken` lacks model wrappers | Requires heavy dependencies & specific abstractions | **Deterministic exact counting** across OpenAI, Claude, Gemini, & Llama |
| **Context Assembly** | Hardcoded array slices; system prompts get trimmed | Truncates arbitrarily without priority awareness | **Priority packing** (CRITICAL → HIGH → MEDIUM → LOW); system prompt always preserved |
| **Chunking Strategy** | Fixed character splits break sentences & code blocks | Basic recursive splitters without semantic boundary detection | **5 Smart Strategies** (Sentence, Paragraph, Semantic, Code, Fixed) |
| **Pricing Integrity** | Hardcoded stale prices in python code | No built-in cost verification | **Versioned pricing registry** with staleness warnings & local overrides |
| **Async & Streaming** | Sync-only thread blocking | Complex async setups | Native `acount`, `achunk`, `aassemble`, and `stream_assemble` |

---

## 30-Second Quickstart (Copy & Paste)

```bash
pip install llm-context-forge
```

```python
from llm_context_forge import ContextWindow, Priority, TokenCounter

# 1. Exact token counting
counter = TokenCounter("gpt-4o")
print(f"Exact tokens: {counter.count('Production prompt here...')}")

# 2. Priority-based context packing (Prevents context overflow)
window = ContextWindow("gpt-4o")
window.add_block("System: You are an expert AI software engineer.", Priority.CRITICAL, "system")
window.add_block("User query: Refactor this database schema", Priority.HIGH, "query")
window.add_block("RAG chunk 1...", Priority.MEDIUM, "rag_doc_1")
window.add_block("RAG chunk 2...", Priority.LOW, "rag_doc_2")

# Assemble safely under token budget
prompt = window.assemble(max_tokens=4096)
stats = window.usage()
print(f"Tokens used: {stats.tokens_used} | Excluded lower-priority blocks: {stats.excluded}")
```

---

## Features & Ecosystem Integrations

* **LangChain Integration**: `ContextForgeTextSplitter` and `ContextForgeDocumentTransformer`
* **LlamaIndex Integration**: Metadata-aware `ContextForgeNodeParser`
* **True Semantic Chunking**: Percentile-based sentence embedding drop algorithm
* **CLI Suite**: Command-line token counting, pricing checks, and REST API server (`llm-context-forge serve`)

```bash
# Count tokens instantly from CLI
llm-context-forge count "Hello world prompt" --model gpt-4o

# Verify pricing registry
llm-context-forge pricing verify gpt-4o
```

---

## Production Notes & Safety Multipliers

* **Safety Buffer Multiplier**: Default 5% safety margin on context windows prevents edge-case token overflow on non-ASCII characters or tool definitions.
* **Overflow Telemetry**: Emits structured log warnings when lower-priority blocks are dropped, ensuring your observability stack catches context budget exhaustion.
* **Pricing Staleness Guarantee**: Automatically alerts developers if pricing data hasn't been refreshed in over 30 days.

---

## Links & Ecosystem

* 📖 **Docs Site**: [docs.dhruvchudasama.me](https://docs.dhruvchudasama.me)
* 📦 **PyPI**: [pypi.org/project/llm-context-forge/](https://pypi.org/project/llm-context-forge/)
* ⚡ **Velox Site**: [velox.satyaneev.me](https://velox.satyaneev.me)
* 🐛 **Issue Tracker**: [GitHub Issues](https://github.com/dhruv-atomic-mui21/llm-context-forge/issues)

## License

[MIT](LICENSE) © Dhruv Chudasama

