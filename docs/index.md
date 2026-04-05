# LayoutLM Forge Documentation

Welcome to the LayoutLM Forge documentation!

LayoutLM Forge is a production-grade LLMOps infrastructure tool designed to help you manage LLM context windows, count tokens accurately, chunk documents intelligently, and estimate API costs before making expensive calls.

## Why LayoutLM Forge?

- **Context window exhaustion failures** in production are frustrating and costly.
- **Inaccurate token counting** leads to unexpected API bills and dropped requests.
- **Naive document chunking** breaks semantic meaning and degrades LLM reasoning.
- **No standard tool** exists for complete LLMOps context management.

## Features

- **Multi-Provider Token Counter**: Supports OpenAI, Anthropic, Google, Llama, and more.
- **Smart Document Chunking**: Chunk by paragraph, sentence, fixed tokens, code block, or semantic Markdown headers.
- **Priority-Based Context Assembly**: Pack your most critical prompts first; optionally drop safe-to-drop blocks.
- **Context Compression**: Extractive, middle-out, truncate, and map-reduce strategies.
- **Cost Estimation Engine**: Check your total bill before processing.
- **REST API + CLI**: First-class support for both programmatic and command-line usage.

See the [Getting Started](getting-started.md) guide to begin.
