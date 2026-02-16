#!/usr/bin/env python3
"""
ContextForge — LLM Context & Token Handling Toolkit

Command-line interface for token counting, document chunking,
context assembly, and compression.
"""

import argparse
import json
import sys
from pathlib import Path


def cmd_count(args):
    """Count tokens in text or a file."""
    from src.tokenizer import TokenCounter

    text = _read_input(args.text)
    counter = TokenCounter(args.model)
    tokens = counter.count(text, args.model)
    info = counter.get_model_info(args.model)
    cost = counter.estimate_cost(text, args.model, "input")

    print(f"Model:          {args.model}")
    print(f"Tokens:         {tokens:,}")
    print(f"Context window: {info.context_window:,}")
    print(f"Fits:           {'✅ Yes' if tokens <= info.context_window else '❌ No'}")
    print(f"Est. cost:      ${cost:.6f}")


def cmd_chunk(args):
    """Chunk text from a file."""
    from src.chunker import DocumentChunker, ChunkStrategy

    text = _read_input(args.input)
    strategy_map = {s.value: s for s in ChunkStrategy}
    strategy = strategy_map.get(args.strategy)
    if strategy is None:
        print(f"Error: Invalid strategy '{args.strategy}'. Choose from: {list(strategy_map.keys())}")
        sys.exit(1)

    chunker = DocumentChunker(args.model)
    chunks = chunker.chunk(
        text,
        strategy=strategy,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap,
        model=args.model,
    )

    if args.format == "json":
        data = [
            {"index": c.index, "tokens": c.token_count, "text": c.text}
            for c in chunks
        ]
        print(json.dumps(data, indent=2))
    else:
        for c in chunks:
            print(f"\n{'='*60}")
            print(f"Chunk {c.index} ({c.token_count} tokens)")
            print(f"{'='*60}")
            print(c.text[:500] + ("..." if len(c.text) > 500 else ""))

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"Total tokens: {sum(c.token_count for c in chunks):,}")


def cmd_assemble(args):
    """Assemble context from a JSON file of blocks."""
    from src.context import ContextWindow, Priority

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    priority_map = {p.name: p for p in Priority}

    window = ContextWindow(args.model)
    for block in data:
        p = priority_map.get(block.get("priority", "MEDIUM").upper(), Priority.MEDIUM)
        window.add_block(
            content=block["content"],
            priority=p,
            label=block.get("label", ""),
        )

    assembled = window.assemble(max_tokens=args.max_tokens)
    usage = window.usage()

    if args.format == "json":
        print(json.dumps({"assembled": assembled, "usage": usage}, indent=2))
    else:
        print(assembled)
        print(f"\n--- Usage ---")
        print(f"Included: {usage['included_tokens']:,} tokens ({usage['num_included']} blocks)")
        print(f"Excluded: {usage['excluded_tokens']:,} tokens ({usage['num_excluded']} blocks)")


def cmd_compress(args):
    """Compress text from a file."""
    from src.compressor import ContextCompressor, CompressionStrategy

    text = _read_input(args.input)
    strategy_map = {s.value: s for s in CompressionStrategy}
    strategy = strategy_map.get(args.strategy)
    if strategy is None:
        print(f"Error: Invalid strategy '{args.strategy}'. Choose from: {list(strategy_map.keys())}")
        sys.exit(1)

    compressor = ContextCompressor(args.model)
    result = compressor.compress(text, args.target, strategy, args.model)

    if args.format == "json":
        print(json.dumps({
            "text": result.text,
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "ratio": round(result.ratio, 4),
            "savings_pct": round(result.savings_pct, 2),
        }, indent=2))
    else:
        print(result.text)
        print(f"\n--- Compression Stats ---")
        print(f"Original:   {result.original_tokens:,} tokens")
        print(f"Compressed: {result.compressed_tokens:,} tokens")
        print(f"Savings:    {result.savings_pct:.1f}%")


def cmd_models(args):
    """List all registered models."""
    from src.tokenizer import ModelRegistry

    models = ModelRegistry.list_models()
    print(f"{'Model':<20} {'Backend':<12} {'Window':>10} {'Input $/1K':>12} {'Output $/1K':>12}")
    print("-" * 70)
    for name in models:
        info = ModelRegistry.get(name)
        print(
            f"{info.name:<20} {info.backend.value:<12} "
            f"{info.context_window:>10,} "
            f"${info.input_cost_per_1k:>10.5f} "
            f"${info.output_cost_per_1k:>10.5f}"
        )


def cmd_api(args):
    """Start the API server."""
    import uvicorn
    from api.app import app

    print(f"Starting ContextForge API at http://{args.host}:{args.port}")
    print(f"Docs: http://{args.host}:{args.port}/docs")
    uvicorn.run(app, host=args.host, port=args.port)


def cmd_demo(_args):
    """Run a quick demo showcasing all features."""
    print("=" * 60)
    print("  ContextForge — LLM Context & Token Handling Toolkit")
    print("=" * 60)
    print()

    # 1. Token counting
    from src.tokenizer import TokenCounter, ModelRegistry
    counter = TokenCounter("gpt-4o")
    sample = "ContextForge helps you manage LLM context windows, count tokens accurately, chunk documents intelligently, and compress prompts to fit within model limits."
    tokens = counter.count(sample)
    cost = counter.estimate_cost(sample)
    print("📊 Token Counting")
    print(f"   Text:   \"{sample[:60]}...\"")
    print(f"   Tokens: {tokens}")
    print(f"   Cost:   ${cost:.6f}")
    print()

    # 2. Chunking
    from src.chunker import DocumentChunker, ChunkStrategy
    chunker = DocumentChunker()
    long_text = (sample + " ") * 20
    chunks = chunker.chunk(long_text, ChunkStrategy.SENTENCE, max_tokens=50)
    print("✂️  Smart Chunking")
    print(f"   Input:  {counter.count(long_text)} tokens")
    print(f"   Chunks: {len(chunks)} (max 50 tokens each)")
    print()

    # 3. Context assembly
    from src.context import ContextWindow, Priority
    window = ContextWindow()
    window.add_block("You are a helpful assistant.", Priority.CRITICAL, "system")
    window.add_block("The user's project is a Python web app.", Priority.HIGH, "project_info")
    window.add_block("Here is the full changelog...", Priority.LOW, "changelog")
    assembled = window.assemble(max_tokens=100)
    usage = window.usage()
    print("🧩 Priority-Based Context Assembly")
    print(f"   Blocks:   {usage['num_blocks']}")
    print(f"   Included: {usage['num_included']}")
    print(f"   Excluded: {usage['num_excluded']}")
    print()

    # 4. Compression
    from src.compressor import ContextCompressor, CompressionStrategy
    compressor = ContextCompressor()
    result = compressor.compress(long_text, target_tokens=30, strategy=CompressionStrategy.EXTRACTIVE)
    print("🗜️  Context Compression")
    print(f"   Original:   {result.original_tokens} tokens")
    print(f"   Compressed: {result.compressed_tokens} tokens")
    print(f"   Savings:    {result.savings_pct:.1f}%")
    print()

    # 5. Models
    print(f"🤖 Supported Models: {len(ModelRegistry.list_models())}")
    for m in ModelRegistry.list_models()[:5]:
        info = ModelRegistry.get(m)
        print(f"   {m:<20} {info.context_window:>10,} tokens")
    print("   ...")
    print()

    print("Run 'python main.py --help' for full CLI usage.")


def _read_input(source: str) -> str:
    """Read text from a file path or treat as inline text."""
    path = Path(source)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")
    return source


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ContextForge — LLM Context & Token Handling Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py count   "Hello world" --model gpt-4
  python main.py chunk   document.txt --strategy sentence --max-tokens 500
  python main.py assemble blocks.json --max-tokens 4096
  python main.py compress document.txt --target 500 --strategy extractive
  python main.py models
  python main.py --api --port 8000
  python main.py --demo
        """,
    )

    # Top-level flags
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--api", action="store_true", help="Start API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="API host")
    parser.add_argument("--port", type=int, default=8000, help="API port")

    sub = parser.add_subparsers(dest="command")

    # count
    p_count = sub.add_parser("count", help="Count tokens")
    p_count.add_argument("text", help="Text or file path")
    p_count.add_argument("--model", default="gpt-4o", help="Model name")

    # chunk
    p_chunk = sub.add_parser("chunk", help="Chunk a document")
    p_chunk.add_argument("input", help="Text or file path")
    p_chunk.add_argument("--strategy", default="paragraph",
                         choices=["fixed", "sentence", "paragraph", "semantic", "code"])
    p_chunk.add_argument("--max-tokens", type=int, default=500)
    p_chunk.add_argument("--overlap", type=int, default=50)
    p_chunk.add_argument("--model", default="gpt-4o")
    p_chunk.add_argument("--format", default="text", choices=["text", "json"])

    # assemble
    p_asm = sub.add_parser("assemble", help="Assemble context from JSON blocks")
    p_asm.add_argument("input", help="JSON file with blocks")
    p_asm.add_argument("--max-tokens", type=int, default=4096)
    p_asm.add_argument("--model", default="gpt-4o")
    p_asm.add_argument("--format", default="text", choices=["text", "json"])

    # compress
    p_comp = sub.add_parser("compress", help="Compress text")
    p_comp.add_argument("input", help="Text or file path")
    p_comp.add_argument("--target", type=int, required=True, help="Target tokens")
    p_comp.add_argument("--strategy", default="extractive",
                        choices=["extractive", "truncate", "middle_out", "map_reduce"])
    p_comp.add_argument("--model", default="gpt-4o")
    p_comp.add_argument("--format", default="text", choices=["text", "json"])

    # models
    sub.add_parser("models", help="List supported models")

    args = parser.parse_args()

    if args.demo:
        cmd_demo(args)
    elif args.api:
        cmd_api(args)
    elif args.command == "count":
        cmd_count(args)
    elif args.command == "chunk":
        cmd_chunk(args)
    elif args.command == "assemble":
        cmd_assemble(args)
    elif args.command == "compress":
        cmd_compress(args)
    elif args.command == "models":
        cmd_models(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
