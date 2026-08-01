# Changelog


## [1.0.1] - 2026-08-01

### Updated
- Automated monthly pricing registry refresh and release.
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-31

### Added
- Production release milestone with documentation site at `https://docs.dhruvchudasama.me`.
- Multi-Python GitHub Actions CI matrix (Python 3.8, 3.9, 3.10, 3.11, 3.12).
- Scheduled automated releases workflow (`.github/workflows/scheduled-release.yml`).
- Zero hardcoded prices lint enforcement.

## [0.5.0] - 2026-07-31

### Added
- Async methods across public API: `TokenCounter.acount()`, `DocumentChunker.achunk()`, `ContextWindow.aassemble()`.
- Streaming context assembly generator: `ContextWindow.stream_assemble()`.
- Performance benchmark test suite for tokenization, chunking, and context assembly throughput.

## [0.4.0] - 2026-07-31

### Added
- Dynamic remote pricing registry update function `update_pricing_registry()` with local file cache (`~/.llm_context_forge/pricing_cache.json`) and ETag validation.
- Extensible `PricingProvider` interface with `BundledYAMLPricingProvider` and `OpenRouterPricingProvider`.
- True embedding-based semantic chunking powered by `sentence-transformers` via Greg Kamradt's percentile drop algorithm (`pip install llm-context-forge[semantic]`).
- Token counting accuracy benchmark test suite across English prose, Code, Non-English text, and Special Characters.

## [0.3.0] - 2026-07-31

### Added
- LangChain integration subpackage: `ContextForgeTextSplitter` and `ContextForgeDocumentTransformer` (with `transform_documents` & async `atransform_documents`).
- LlamaIndex integration subpackage: `ContextForgeNodeParser` (subclassing `MetadataAwareTextSplitter`).
- CLI pricing management commands: `llm-context-forge pricing list`, `llm-context-forge pricing verify <model>`, `llm-context-forge pricing update [--remote]`.

## [0.2.0] - 2026-07-31

### Added
- Versioned YAML pricing registry (`pricing_registry.yaml`) with 20+ verified model pricing entries.
- Runtime staleness warning (`PricingDataStaleWarning`) firing when model pricing data is >30 days old.
- Support for `LLM_CONTEXT_FORGE_PRICING_FILE` environment variable for enterprise local pricing overrides.
- Renamed `ChunkStrategy.SEMANTIC` to `ChunkStrategy.HEURISTIC` for markdown structure splitting with backward-compatible deprecation warning.

### Removed
- Removed hardcoded pricing definitions from Python source code.
- Removed Meta as a provider concept in favor of provider-specific open-source host entries (e.g. `llama-3.1-8b-groq`).

## [0.1.5] - 2026-07-15

### Added
- Initial public beta with core `TokenCounter`, `DocumentChunker`, `ContextWindow`, `ContextCompressor`, and `CostCalculator`.
- FastAPI server with `/docs` interactive endpoint.
- CLI commands and Docker container setup.
