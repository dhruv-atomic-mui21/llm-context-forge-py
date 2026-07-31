"""
Performance benchmarks for Tokenizer and Chunker throughput.
Run with: pytest tests/performance/ --benchmark
"""

import pytest
from llm_context_forge.tokenizer import TokenCounter
from llm_context_forge.chunker import DocumentChunker, ChunkStrategy


def test_benchmark_tokenizer(request):
    from tests.fixtures.sample_data import LONG_TEXT
    counter = TokenCounter("gpt-4o")
    
    def count_tokens():
        return counter.count(LONG_TEXT)

    if "benchmark" in request.fixturenames:
        benchmark = request.getfixturevalue("benchmark")
        result = benchmark(count_tokens)
    else:
        result = count_tokens()
        
    assert result > 0


def test_benchmark_chunker(request):
    from tests.fixtures.sample_data import LONG_TEXT
    chunker = DocumentChunker("gpt-4o")
    
    def chunk_doc():
        return chunker.chunk(LONG_TEXT, ChunkStrategy.SENTENCE, max_tokens=50)

    if "benchmark" in request.fixturenames:
        benchmark = request.getfixturevalue("benchmark")
        result = benchmark(chunk_doc)
    else:
        result = chunk_doc()
        
    assert len(result) > 0
