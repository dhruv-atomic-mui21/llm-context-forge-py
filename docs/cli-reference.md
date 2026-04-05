# CLI Reference

The `contextforge` command-line tool is built with Typer for a rich, user-friendly experience.

## Commands

### `contextforge count`
Count tokens in a text snippet or a file.
```bash
contextforge count "Hello world!" --model gpt-4o
```

### `contextforge chunk`
Breaks documents into smaller token-bounded pieces.
```bash
contextforge chunk my_text.txt --strategy paragraph --max-tokens 500
```

### `contextforge assemble`
Assemble context from a JSON file of blocks, respecting priority limits.
```bash
contextforge assemble data.json --max-tokens 4096
```

### `contextforge compress`
Compress text.
```bash
contextforge compress data.txt --target 200 --strategy extractive
```

### `contextforge cost`
Estimate cost for a text payload.
```bash
contextforge cost "Hello world" --model claude-3.5-sonnet
```

### `contextforge models`
List all registered models, their backend, context window limits, and pricing.

### `contextforge doctor`
Verify your system environment, checking if you have `tiktoken` installed.

### `contextforge serve`
Run the FastAPI server.
```bash
contextforge serve --host 0.0.0.0 --port 8080
```

### `contextforge demo`
Run an interactive demo showcasing token counting, chunking, assembling, and compressing.
