# CLI Reference

The `llm_context_forge` command-line tool is built with Typer for a rich, user-friendly experience.

## Commands

### `llm_context_forge count`
Count tokens in a text snippet or a file.
```bash
llm_context_forge count "Hello world!" --model gpt-4o
```

### `llm_context_forge chunk`
Breaks documents into smaller token-bounded pieces.
```bash
llm_context_forge chunk my_text.txt --strategy paragraph --max-tokens 500
```

### `llm_context_forge assemble`
Assemble context from a JSON file of blocks, respecting priority limits.
```bash
llm_context_forge assemble data.json --max-tokens 4096
```

### `llm_context_forge compress`
Compress text.
```bash
llm_context_forge compress data.txt --target 200 --strategy extractive
```

### `llm_context_forge cost`
Estimate cost for a text payload.
```bash
llm_context_forge cost "Hello world" --model claude-3.5-sonnet
```

### `llm_context_forge models`
List all registered models, their backend, context window limits, and pricing.

### `llm_context_forge doctor`
Verify your system environment, checking if you have `tiktoken` installed.

### `llm_context_forge serve`
Run the FastAPI server.
```bash
llm_context_forge serve --host 0.0.0.0 --port 8080
```

### `llm_context_forge demo`
Run an interactive demo showcasing token counting, chunking, assembling, and compressing.
