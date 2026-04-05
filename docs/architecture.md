# Architecture

ContextForge is split into modular components that can be used independently or together through the API and CLI.

## Core Modules

1. **`contextforge.models`**: Stores model metadata, cost schema, and backend mapping (OpenAI, Anthropic, Google, etc.).
2. **`contextforge.tokenizer`**: Uses `tiktoken` for OpenAI models and heuristic estimations for others to rapidly count tokens.
3. **`contextforge.chunker`**: Employs Regex and token counting to securely chunk documents based on semantics (paragraphs, markdown nodes, functions/classes).
4. **`contextforge.context`**: A Priority queue-style greedy packer. It guarantees that `CRITICAL` or `HIGH` priority blocks fit inside the window before packing `LOW` priority blocks.
5. **`contextforge.compressor`**: Token budget optimizer. Uses heuristics like TF-IDF or simple map-reduce text truncation to forcefully fit contexts into smaller token constraints. 

## Flow

1. You **Chunk** a large document.
2. You feed the chunks into the **Context** assembler, adding priority tags (so that old data is dropped if token limit is reached).
3. If necessary, you run **Compression** on the chunks to squeeze out more space.
4. Before issuing the API request to the LLM, you use **Cost** to estimate the total expense of your context string.
