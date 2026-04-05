"""
Integration tests for a full pipeline flow
"""

import pytest
from contextforge.chunker import DocumentChunker, ChunkStrategy
from contextforge.context import ContextWindow, Priority
from contextforge.compressor import ContextCompressor, CompressionStrategy
from contextforge.cost import CostCalculator

def test_full_pipeline_flow():
    from tests.fixtures.sample_data import LONG_TEXT
    
    # 1. Chunk document
    chunker = DocumentChunker("gpt-4o")
    chunks = chunker.chunk(LONG_TEXT, strategy=ChunkStrategy.SENTENCE, max_tokens=100)
    assert len(chunks) > 1

    # 2. Compress the first chunk
    compressor = ContextCompressor("gpt-4o")
    comp_res = compressor.compress(chunks[0].text, target_tokens=50, strategy=CompressionStrategy.EXTRACTIVE)
    assert comp_res.compressed_tokens <= 60

    # 3. Assemble into context
    window = ContextWindow("gpt-4o")
    window.add_block(comp_res.text, Priority.CRITICAL, "summary")
    window.add_block(chunks[-1].text, Priority.LOW, "details")
    assembled = window.assemble(max_tokens=200)
    assert len(assembled) > 0

    # 4. Calculate cost
    calc = CostCalculator("gpt-4o")
    cost = calc.estimate_prompt(assembled)
    assert cost.usd > 0
