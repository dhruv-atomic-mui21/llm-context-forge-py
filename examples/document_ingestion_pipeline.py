"""
Document Ingestion Pipeline Example

Shows how to ingest a long document, chunk it securely, 
compress the chunks to save tokens, and estimate the cost.
"""

from contextforge.chunker import DocumentChunker, ChunkStrategy
from contextforge.compressor import ContextCompressor, CompressionStrategy
from contextforge.cost import CostCalculator

def main():
    # Simulate a long document (e.g. parsed PDF or markdown file)
    long_doc = (
        "# Introduction\n\n"
        "This is an important module that handles parsing. "
        "It includes multiple steps such as feature extraction and tokenization.\n\n"
        "# Process\n\n"
        "We first normalize the text. "
        "Then we use the model to extract embedding vectors. " * 30
    )

    model = "gpt-4o"

    # Step 1: Chunking
    chunker = DocumentChunker(model)
    chunks = chunker.chunk_markdown(long_doc, max_tokens=100)
    print(f"Document split into {len(chunks)} chunks.")

    # Step 2: Compression
    compressor = ContextCompressor(model)
    compressed_chunks = []
    
    for i, c in enumerate(chunks):
        # We'll compress each chunk to max 50 tokens
        comp_res = compressor.compress(c.text, target_tokens=50, strategy=CompressionStrategy.MAP_REDUCE)
        compressed_chunks.append(comp_res.text)
        print(f"  Chunk {i}: {c.token_count} tokens -> {comp_res.compressed_tokens} tokens")

    # Step 3: Cost Estimation
    calc = CostCalculator(model)
    original_cost = calc.bulk_estimate([c.text for c in chunks])
    compressed_cost = calc.bulk_estimate(compressed_chunks)

    print("\n--- Savings ---")
    print(f"Original total cost:   ${original_cost.total_usd:.6f}")
    print(f"Compressed total cost: ${compressed_cost.total_usd:.6f}")

if __name__ == "__main__":
    main()
