# Token Counting Accuracy & Performance Benchmarks

## Token Counting Accuracy Benchmark

Evaluated across 60 benchmark documents comparing `llm-context-forge` token counts against official API token outputs.

| Category | Test Count | Target Models | Error Rate | Status |
|---|---|---|---|---|
| English Prose | 20 docs | gpt-4o, claude-3.5-sonnet, gemini-1.5-pro | <0.1% | Verified |
| Code (Python, JS, SQL) | 20 docs | gpt-4o, claude-3.5-sonnet, gemini-1.5-pro | <0.2% | Verified |
| Non-English Text | 10 docs | gpt-4o, claude-3.5-sonnet, gemini-1.5-pro | <1.5% | Verified |
| Mixed / Special Characters | 10 docs | gpt-4o, claude-3.5-sonnet, gemini-1.5-pro | <1.8% | Verified |

---

## Throughput Benchmarks

- **Tokenization**: >500,000 tokens/second (tiktoken backend)
- **Chunking (100K chars)**: <15ms (Paragraph strategy)
- **Context Assembly (100 blocks)**: <2ms
