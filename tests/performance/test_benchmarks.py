"""
Performance benchmarks 
Run with: pytest tests/performance/ --benchmark
"""

import pytest
from contextforge.tokenizer import TokenCounter
from contextforge.chunker import DocumentChunker, ChunkStrategy

def test_benchmark_tokenizer(benchmark):
    from tests.fixtures.sample_data import LONG_TEXT
    counter = TokenCounter("gpt-4o")
    
    def count_tokens():
        return counter.count(LONG_TEXT)
        
    result = benchmark(count_tokens)
    assert result > 0

def test_benchmark_chunker(benchmark):
    from tests.fixtures.sample_data import LONG_TEXT
    chunker = DocumentChunker("gpt-4o")
    
    def chunk_doc():
        return chunker.chunk(LONG_TEXT, ChunkStrategy.SENTENCE, max_tokens=50)
        
    result = benchmark(chunk_doc)
    assert len(result) > 0
