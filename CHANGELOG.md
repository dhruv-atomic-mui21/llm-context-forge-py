# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-09-01

### Added
- **Multimodal Vision Token Sizing**: Added `VisionTokenCounter` and `count_image_tokens` supporting OpenAI (GPT-4o detail tiers & tiles), Anthropic (Claude 3.5 1568px bounds), and Google Gemini (768px patches).
- Added `pillow>=10.0.0` core dependency with support for reading image dimensions directly from file paths, byte buffers, and stream objects without full pixel decompression.
- Formalized `[project.optional-dependencies]` in `pyproject.toml` including `integrations` (`langchain-core`, `llama-index-core`).
- **Credibility Repair Release**: Aligned all marketing and technical claims with actual codebase behavior.
- Added rigorous test coverage for edge cases: empty input, unicode handling, separators, overlap behavior, invalid config, and deterministic output.
- Added `docs/benchmarks.md` detailing methodology, environment, input corpus, and limitations of performance assertions.
- Implemented parameter validation (e.g., negative `max_tokens`, invalid `overlap_tokens`) across chunking utilities.
- Added explicit Markdown-based Heuristic chunking documentation to clarify non-embedding behavior.

### Changed
- Refactored `README.md` to remove unverified claims regarding "true semantic chunking", "sovereign", and "zero telemetry" without measurement boundaries.
- Replaced word "securely" with "deterministically" regarding chunk splitting logic.
- Accurately described the 5% token safety multiplier as applying to estimation fallback only, not the entire context window.

### Removed
- Removed hardcoded pricing definitions from Python source code, replacing with versioned `pricing_registry.yaml`.

## [0.1.5] - 2026-07-15

### Added
- Initial public beta with core `TokenCounter`, `DocumentChunker`, `ContextWindow`, `ContextCompressor`, and `CostCalculator`.
- FastAPI server with `/docs` interactive endpoint.
- CLI commands and Docker container setup.
