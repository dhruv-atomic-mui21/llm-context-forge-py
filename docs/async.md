# Async & Streaming Context Assembly

`llm-context-forge` provides native async methods for production async pipelines (FastAPI, asyncio, anyio).

---

## Async API

```python
from llm_context_forge import TokenCounter, DocumentChunker, ContextWindow

# Async Token Counter
counter = TokenCounter("gpt-4o")
tokens = await counter.acount("Your prompt text...")

# Async Chunker
chunker = DocumentChunker("gpt-4o")
chunks = await chunker.achunk("Long document...", max_tokens=500)

# Async Context Window
window = ContextWindow("gpt-4o")
assembled = await window.aassemble(max_tokens=4096)
```

---

## Streaming Context Assembly

For agentic pipelines building context dynamically:

```python
async for block in window.stream_assemble(max_tokens=4096):
    print(f"Included block: {block.label} ({block.token_count} tokens)")
```
