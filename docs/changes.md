# Summary of Changes and Architectural Upgrades

This document details all technical changes, architectural refactoring, and new feature additions implemented across the `llm-context-forge` release roadmap leading up to version **1.0.0**.

---

## 1. Version 0.2.0 — Pricing Integrity Release

### Architectural Changes
- **Zero Hardcoded Prices**: Removed all hardcoded price floats from Python modules (`models.py` and `cost.py`).
- **Versioned Pricing Registry**: Created `pricing_registry.yaml` bundling verified model pricing, source URLs, retrieval dates, and confidence ratings for 20+ models across OpenAI, Anthropic, Google, Mistral, Cohere, and Groq.
- **Staleness Warnings**: Added `PricingDataStaleWarning` runtime check that alerts infrastructure teams when model pricing data exceeds 30 days of age without breaking production execution.
- **Environment Overrides**: Added `LLM_CONTEXT_FORGE_PRICING_FILE` environment variable support allowing enterprise teams to supply custom internal pricing YAML files.
- **Provider Refactoring**: Removed "Meta" as a provider concept; replaced with host-specific model identifiers (e.g., `llama-3.1-8b-groq`, `llama-3.1-70b-together`).
- **Semantic Strategy Renaming**: Renamed heading-based markdown splitting from `ChunkStrategy.SEMANTIC` to `ChunkStrategy.HEURISTIC` with a backward-compatible deprecation warning.

---

## 2. Version 0.3.0 — Framework Integrations Release

### Architectural Changes
- **LangChain Integration (`llm_context_forge.integrations.langchain`)**:
  - `ContextForgeTextSplitter`: Subclasses LangChain's `TextSplitter` delegating splitting logic to `DocumentChunker`.
  - `ContextForgeDocumentTransformer`: Subclasses `BaseDocumentTransformer` implementing both `transform_documents()` and `atransform_documents()`.
- **LlamaIndex Integration (`llm_context_forge.integrations.llamaindex`)**:
  - `ContextForgeNodeParser`: Subclasses `MetadataAwareTextSplitter` dynamically selecting chunking strategies based on document metadata (e.g. code vs markdown).
- **CLI Pricing Management Suite**: Added `llm-context-forge pricing list`, `verify`, and `update` commands.

---

## 3. Version 0.4.0 — Live Pricing & Accuracy Release

### Architectural Changes
- **Extensible Pricing Loaders**: Created `PricingProvider` interface supporting `BundledYAMLPricingProvider` and `OpenRouterPricingProvider`.
- **Remote Pricing Sync**: Implemented `update_pricing_registry()` supporting remote JSON fetching with local disk cache (`~/.llm_context_forge/pricing_cache.json`) and ETag headers.
- **True Embedding Semantic Chunking**: Added `pip install llm-context-forge[semantic]` extra implementing Greg Kamradt's percentile drop algorithm via `sentence-transformers`.
- **Token Accuracy Benchmark Suite**: Added automated benchmark suite (`tests/performance/test_accuracy_benchmark.py`) testing token counting accuracy across English prose, Code, Non-English text, and Special characters against official API counts (<2% error rate).

---

## 4. Version 0.5.0 — Async & Performance Release

### Architectural Changes
- **Native Async API**: Implemented non-blocking async counterparts offloading CPU tokenization to worker threads via `anyio.to_thread.run_sync`:
  - `TokenCounter.acount()`
  - `DocumentChunker.achunk()`
  - `ContextWindow.aassemble()`
- **Streaming Context Assembly**: Implemented `ContextWindow.stream_assemble()` yielding context blocks as an async generator in priority order.

---

## 5. Version 1.0.0 — Production Milestone & Scheduled Releases

### Architectural Changes
- **Documentation Site**: Configured MkDocs Material documentation hosted at [docs.dhruvchudasama.me](https://docs.dhruvchudasama.me).
- **Multi-Python CI Matrix**: Built `.github/workflows/ci.yml` running automated tests across Python 3.8, 3.9, 3.10, 3.11, and 3.12 with zero hardcoded pricing lint checks.
- **Automated Scheduled Release System**: Created `.github/workflows/scheduled-release.yml` executing on a monthly/cron schedule to verify pricing freshness, run test matrices, build distribution artifacts, and release tagged versions automatically.
