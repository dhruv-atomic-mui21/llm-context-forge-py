# LLM Context Forge Documentation

Welcome to the official documentation for **LLM Context Forge** hosted at [docs.dhruvchudasama.me](https://docs.dhruvchudasama.me).

`llm-context-forge` provides production-grade LLMOps infrastructure for context window management, token counting, document chunking, cost estimation, and prompt compression.

---

## Key Features

- 💰 **Pricing Integrity**: Versioned YAML pricing registry (`pricing_registry.yaml`), runtime staleness warnings (>30 days), and local enterprise overrides via `LLM_CONTEXT_FORGE_PRICING_FILE`.
- 🧩 **Integrations**: Native splitters and transformers for **LangChain** (`ContextForgeTextSplitter`, `ContextForgeDocumentTransformer`) and **LlamaIndex** (`ContextForgeNodeParser`).
- ⚡ **Async & Streaming**: Full async support (`acount()`, `achunk()`, `aassemble()`) and streaming context block assembly (`stream_assemble()`).
- 🎯 **Semantic Chunking**: Embedding-based percentile drop algorithm via `sentence-transformers` (`pip install llm-context-forge[semantic]`).
- 🌐 **Live Remote Pricing**: Remote registry fetch with disk caching and ETag headers.

---

## Quick Installation

```bash
pip install llm-context-forge
```

For framework integrations and optional extras:
```bash
pip install llm-context-forge[semantic,api]
```

Visit the full documentation sections in the navigation bar to learn more.
