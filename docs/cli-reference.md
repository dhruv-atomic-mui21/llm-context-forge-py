# CLI Reference

The `layoutlm_forge` command-line tool is built with Typer for a rich, user-friendly experience.

## Commands

### `layoutlm_forge count`
Count tokens in a text snippet or a file.
```bash
layoutlm_forge count "Hello world!" --model gpt-4o
```

### `layoutlm_forge chunk`
Breaks documents into smaller token-bounded pieces.
```bash
layoutlm_forge chunk my_text.txt --strategy paragraph --max-tokens 500
```

### `layoutlm_forge assemble`
Assemble context from a JSON file of blocks, respecting priority limits.
```bash
layoutlm_forge assemble data.json --max-tokens 4096
```

### `layoutlm_forge compress`
Compress text.
```bash
layoutlm_forge compress data.txt --target 200 --strategy extractive
```

### `layoutlm_forge cost`
Estimate cost for a text payload.
```bash
layoutlm_forge cost "Hello world" --model claude-3.5-sonnet
```

### `layoutlm_forge models`
List all registered models, their backend, context window limits, and pricing.

### `layoutlm_forge doctor`
Verify your system environment, checking if you have `tiktoken` installed.

### `layoutlm_forge serve`
Run the FastAPI server.
```bash
layoutlm_forge serve --host 0.0.0.0 --port 8080
```

### `layoutlm_forge demo`
Run an interactive demo showcasing token counting, chunking, assembling, and compressing.
