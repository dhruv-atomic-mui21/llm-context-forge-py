# 🧠 ContextForge — LLM Context & Token Handling Toolkit

Manage LLM context windows, count tokens accurately, chunk documents intelligently, and compress prompts — solving the real pain-points of human–LLM interactions.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![tiktoken](https://img.shields.io/badge/tiktoken-0.5+-orange.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-teal.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- ✅ **Multi-Provider Token Counter** — Accurate counting for OpenAI (tiktoken), Anthropic, Google, and Llama models
- ✅ **Smart Document Chunking** — Sentence, paragraph, semantic, and code-aware splitting with overlap
- ✅ **Priority-Based Context Assembly** — Tag blocks with priorities; the manager greedily packs the window
- ✅ **Context Compression** — Extractive, truncate, middle-out, and map-reduce strategies
- ✅ **Conversation Manager** — Auto-trim old messages while preserving recent context
- ✅ **Cost Estimation** — Estimate API costs before making calls
- ✅ **REST API** — FastAPI endpoints for all features
- ✅ **CLI** — Command-line interface for quick operations

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/contextforge.git
cd contextforge
pip install -r requirements.txt
```

### Token Counting

```python
from src.tokenizer import TokenCounter

counter = TokenCounter("gpt-4o")

# Count tokens
tokens = counter.count("Hello, world!")
print(f"Tokens: {tokens}")

# Check if text fits in context window
fits = counter.fits_in_window("Your prompt here", max_tokens=4096)

# Estimate cost
cost = counter.estimate_cost("Your prompt", model="gpt-4o", direction="input")
print(f"Cost: ${cost:.6f}")

# Count chat messages (ChatML-aware)
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Hello!"},
]
total = counter.count_messages(messages)
```

### Smart Chunking

```python
from src.chunker import DocumentChunker, ChunkStrategy

chunker = DocumentChunker()

# Paragraph-aware chunking
chunks = chunker.chunk(long_text, ChunkStrategy.PARAGRAPH, max_tokens=500)

# Code-aware chunking
code_chunks = chunker.chunk_code(source_code, language="python", max_tokens=500)

# Markdown-aware chunking
md_chunks = chunker.chunk_markdown(readme_text, max_tokens=500)

for chunk in chunks:
    print(f"Chunk {chunk.index}: {chunk.token_count} tokens")
```

### Context Window Management

```python
from src.context import ContextWindow, Priority

window = ContextWindow()

# Add blocks with priorities
window.add_block("You are a helpful assistant.", Priority.CRITICAL, "system")
window.add_block("Project: Python web app using Flask.", Priority.HIGH, "project")
window.add_block("Here is the full changelog...", Priority.LOW, "changelog")
window.add_block("Optional: code style guide...", Priority.OPTIONAL, "style")

# Assemble — highest priority first, respecting token limit
context = window.assemble(max_tokens=4096)

# Check what was included/excluded
usage = window.usage()
print(f"Included: {usage['included_tokens']} tokens")
print(f"Excluded: {usage['excluded_tokens']} tokens")
```

### Context Compression

```python
from src.compressor import ContextCompressor, CompressionStrategy

compressor = ContextCompressor()

# Extractive compression (keeps key sentences)
result = compressor.compress(text, target_tokens=500, strategy=CompressionStrategy.EXTRACTIVE)
print(f"Compressed: {result.original_tokens} → {result.compressed_tokens} tokens")
print(f"Savings: {result.savings_pct:.1f}%")

# Middle-out (keeps start + end, removes middle)
result = compressor.compress(text, target_tokens=500, strategy=CompressionStrategy.MIDDLE_OUT)

# Compress conversation history
compressed_msgs = compressor.compress_conversation(messages, target_tokens=2000)
```

### Conversation Manager

```python
from src.context import ConversationManager

conv = ConversationManager("gpt-4o")

conv.add_message("system", "You are helpful.")
conv.add_message("user", "What is Python?")
conv.add_message("assistant", "Python is a programming language...")
conv.add_message("user", "Tell me more.")

# Auto-trim to fit in context window
messages = conv.get_context(max_tokens=4096)
print(conv.token_usage())
```

## 🖥️ API Server

```bash
python main.py --api --port 8000
```

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | API info |
| `/health` | GET | Health check |
| `/tokens/count` | POST | Count tokens for text + model |
| `/tokens/validate` | POST | Check if text fits in window |
| `/chunk` | POST | Chunk text with strategy |
| `/context/assemble` | POST | Assemble blocks by priority |
| `/compress` | POST | Compress text to target tokens |

### Example API Request

```bash
curl -X POST "http://localhost:8000/tokens/count" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world!", "model": "gpt-4o"}'
```

## 📁 Project Structure

```
contextforge/
├── api/
│   ├── __init__.py
│   └── app.py              # FastAPI application
├── src/
│   ├── __init__.py
│   ├── tokenizer.py        # Multi-provider token counter
│   ├── chunker.py          # Intelligent document chunker
│   ├── context.py          # Context window manager
│   └── compressor.py       # Context compression engine
├── tests/
│   ├── test_tokenizer.py
│   ├── test_chunker.py
│   ├── test_context.py
│   ├── test_compressor.py
│   └── test_api.py
├── .gitignore
├── CONTRIBUTING.md
├── LICENSE
├── main.py                 # CLI entry point
├── README.md
└── requirements.txt
```

## 🤖 Supported Models

| Provider | Models | Context Window |
|---|---|---|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-4, gpt-3.5-turbo | 8K – 128K |
| **Anthropic** | claude-3-opus, claude-3.5-sonnet, claude-3-haiku | 200K |
| **Google** | gemini-pro, gemini-flash | 1M |
| **Llama** | llama-3-8b, llama-3-70b | 8K |
| **Custom** | Register your own via `ModelRegistry.register()` | Any |

## 🔧 CLI Usage

```bash
# Count tokens
python main.py count "Hello world" --model gpt-4

# Chunk a document
python main.py chunk document.txt --strategy sentence --max-tokens 500

# Assemble context from JSON blocks
python main.py assemble blocks.json --max-tokens 4096

# Compress text
python main.py compress document.txt --target 1000 --strategy extractive

# List supported models
python main.py models

# Start API server
python main.py --api --port 8000

# Run demo
python main.py --demo
```

## 🧪 Testing

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ -v --cov=src --cov-report=html
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [tiktoken](https://github.com/openai/tiktoken) for OpenAI tokenisation
- [FastAPI](https://fastapi.tiangolo.com/) for the REST API framework
