# API Reference

LayoutLM Forge includes a FastAPI application to serve these tools over HTTP.

By default, the server runs on port 8000. Start it via CLI:

```bash
layoutlm_forge serve --port 8000
```

## Endpoints

### `GET /health`
Returns system health, version, and the number of supported model providers.

### `POST /api/v1/tokens/count`
Count tokens for a given string.

**Request:**
```json
{
  "text": "Hello world!",
  "model": "gpt-4o"
}
```

### `POST /api/v1/cost/estimate`
Estimate input cost for processing text.

**Request:**
```json
{
  "text": "Hello world!",
  "model": "gpt-4o"
}
```

### `POST /api/v1/chunks/`
Chunk text with strategies like `paragraph`, `sentence`, `fixed`, `semantic`, and `code`.

### `POST /api/v1/context/assemble`
Pass a list of priority-labelled blocks.

**Request:**
```json
{
  "blocks": [
    {"content": "System prompt", "priority": "CRITICAL"},
    {"content": "Irrelevant chat", "priority": "LOW"}
  ],
  "max_tokens": 100
}
```

### `POST /api/v1/compress/`
Compress long text into a specific target budget.

**Strategies**: `extractive`, `truncate`, `middle_out`, `map_reduce`
