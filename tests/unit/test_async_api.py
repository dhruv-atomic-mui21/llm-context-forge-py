"""
Unit tests for Async and Streaming APIs across TokenCounter, DocumentChunker, and ContextWindow.
"""

import pytest
from llm_context_forge.tokenizer import TokenCounter
from llm_context_forge.chunker import DocumentChunker, ChunkStrategy
from llm_context_forge.context import ContextWindow, Priority


@pytest.mark.asyncio
async def test_async_token_counter():
    counter = TokenCounter("gpt-4o")
    count = await counter.acount("Hello async world!")
    assert count > 0


@pytest.mark.asyncio
async def test_async_document_chunker():
    chunker = DocumentChunker("gpt-4o")
    text = "Line 1.\n\nLine 2.\n\nLine 3."
    chunks = await chunker.achunk(text, strategy=ChunkStrategy.PARAGRAPH, max_tokens=10)
    assert len(chunks) >= 1


@pytest.mark.asyncio
async def test_async_context_window_assembly():
    window = ContextWindow("gpt-4o")
    window.add_block("Critical block content", Priority.CRITICAL, "sys")
    window.add_block("High block content", Priority.HIGH, "doc")
    
    assembled = await window.aassemble(max_tokens=100)
    assert "Critical block content" in assembled

    blocks = []
    async for block in window.stream_assemble(max_tokens=100):
        blocks.append(block)
    assert len(blocks) == 2
    assert blocks[0].label == "sys"
