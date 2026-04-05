"""
ContextForge Typer CLI
"""

import json
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from contextforge.tokenizer import TokenCounter, ModelRegistry
from contextforge.chunker import DocumentChunker, ChunkStrategy
from contextforge.context import ContextWindow, Priority
from contextforge.compressor import ContextCompressor, CompressionStrategy
from contextforge.cost import CostCalculator

app = typer.Typer(
    name="contextforge",
    help="Production LLMOps infrastructure for context management",
    no_args_is_help=True,
)
console = Console()


def _read_input(source: str) -> str:
    path = Path(source)
    if path.exists() and path.is_file():
        return path.read_text(encoding="utf-8")
    return source


@app.command()
def count(
    text: str = typer.Argument(..., help="Text or file path to count tokens for"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model name"),
) -> None:
    """Count tokens in text or a file."""
    content = _read_input(text)
    counter = TokenCounter(model)
    tokens = counter.count(content, model)
    info = counter.get_model_info(model)
    cost = counter.estimate_cost(content, model, "input")

    console.print(f"Model:          [bold blue]{model}[/bold blue]")
    console.print(f"Tokens:         [bold green]{tokens:,}[/bold green]")
    console.print(f"Context window: {info.context_window:,}")
    fits_str = "[bold green]✅ Yes[/bold green]" if tokens <= info.context_window else "[bold red]❌ No[/bold red]"
    console.print(f"Fits:           {fits_str}")
    console.print(f"Est. cost:      ${cost:.6f}")


@app.command()
def chunk(
    input_path: str = typer.Argument(..., help="Text or file path to chunk", metavar="INPUT"),
    strategy: str = typer.Option("paragraph", "--strategy", help="Chunking strategy (fixed, sentence, paragraph, semantic, code)"),
    max_tokens: int = typer.Option(500, "--max-tokens", help="Maximum tokens per chunk"),
    overlap: int = typer.Option(50, "--overlap", help="Overlap in tokens between chunks"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model to use for token counting"),
    format_out: str = typer.Option("text", "--format", help="Output format (text, json)"),
) -> None:
    """Chunk text into smaller pieces."""
    content = _read_input(input_path)
    strategy_map = {s.value: s for s in ChunkStrategy}
    strat = strategy_map.get(strategy.lower())
    if strat is None:
        console.print(f"[bold red]Error:[/bold red] Invalid strategy '{strategy}'. Choose from: {list(strategy_map.keys())}")
        raise typer.Exit(1)

    chunker = DocumentChunker(model)
    chunks = chunker.chunk(
        content,
        strategy=strat,
        max_tokens=max_tokens,
        overlap_tokens=overlap,
        model=model,
    )

    if format_out == "json":
        data = [{"index": c.index, "tokens": c.token_count, "text": c.text} for c in chunks]
        console.print(json.dumps(data, indent=2))
    else:
        for c in chunks:
            console.print(f"\n[bold yellow]{'='*60}[/bold yellow]")
            console.print(f"[bold green]Chunk {c.index}[/bold green] ({c.token_count} tokens)")
            console.print(f"[bold yellow]{'='*60}[/bold yellow]")
            snippet = c.text[:500] + ("..." if len(c.text) > 500 else "")
            console.print(snippet)

        console.print(f"\nTotal chunks: {len(chunks)}")
        console.print(f"Total tokens: {sum(c.token_count for c in chunks):,}")


@app.command()
def assemble(
    input_path: str = typer.Argument(..., help="JSON file with blocks"),
    max_tokens: int = typer.Option(4096, "--max-tokens", help="Maximum token budget"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model name"),
    format_out: str = typer.Option("text", "--format", help="Output format (text, json)"),
) -> None:
    """Assemble context from a JSON file of blocks."""
    path = Path(input_path)
    if not path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {input_path}")
        raise typer.Exit(1)

    data = json.loads(path.read_text(encoding="utf-8"))
    priority_map = {p.name: p for p in Priority}

    window = ContextWindow(model)
    for block in data:
        p = priority_map.get(block.get("priority", "MEDIUM").upper(), Priority.MEDIUM)
        window.add_block(
            content=block["content"],
            priority=p,
            label=block.get("label", ""),
        )

    assembled = window.assemble(max_tokens=max_tokens)
    usage = window.usage()

    if format_out == "json":
        console.print(json.dumps({"assembled": assembled, "usage": usage}, indent=2))
    else:
        console.print(assembled)
        console.print("\n[bold]--- Usage ---[/bold]")
        console.print(f"Included: {usage['included_tokens']:,} tokens ({usage['num_included']} blocks)")
        console.print(f"Excluded: {usage['excluded_tokens']:,} tokens ({usage['num_excluded']} blocks)")


@app.command()
def compress(
    input_path: str = typer.Argument(..., help="Text or file path", metavar="INPUT"),
    target: int = typer.Option(..., "--target", help="Target tokens"),
    strategy: str = typer.Option("extractive", "--strategy", help="Compression strategy (extractive, truncate, middle_out, map_reduce)"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model name"),
    format_out: str = typer.Option("text", "--format", help="Output format (text, json)"),
) -> None:
    """Compress text to fit a token budget."""
    content = _read_input(input_path)
    strategy_map = {s.value: s for s in CompressionStrategy}
    strat = strategy_map.get(strategy.lower())
    if strat is None:
        console.print(f"[bold red]Error:[/bold red] Invalid strategy '{strategy}'. Choose from: {list(strategy_map.keys())}")
        raise typer.Exit(1)

    compressor = ContextCompressor(model)
    result = compressor.compress(content, target, strat, model)

    if format_out == "json":
        console.print(json.dumps({
            "text": result.text,
            "original_tokens": result.original_tokens,
            "compressed_tokens": result.compressed_tokens,
            "ratio": round(result.ratio, 4),
            "savings_pct": round(result.savings_pct, 2),
        }, indent=2))
    else:
        console.print(result.text)
        console.print("\n[bold]--- Compression Stats ---[/bold]")
        console.print(f"Original:   {result.original_tokens:,} tokens")
        console.print(f"Compressed: {result.compressed_tokens:,} tokens")
        console.print(f"Savings:    {result.savings_pct:.1f}%")


@app.command()
def models() -> None:
    """List all registered models."""
    table = Table(title="Registered Models")
    table.add_column("Model", justify="left", style="cyan", no_wrap=True)
    table.add_column("Backend", style="magenta")
    table.add_column("Context Window", justify="right", style="green")
    table.add_column("Input $/1K", justify="right")
    table.add_column("Output $/1K", justify="right")

    for name in ModelRegistry.list_models():
        info = ModelRegistry.get(name)
        table.add_row(
            info.name,
            info.backend.value,
            f"{info.context_window:,}",
            f"${info.input_cost_per_1k:.5f}",
            f"${info.output_cost_per_1k:.5f}"
        )
    console.print(table)


@app.command()
def cost(
    input_path: str = typer.Argument(..., help="Text or file path", metavar="INPUT"),
    model: str = typer.Option("gpt-4o", "--model", "-m", help="Model name"),
) -> None:
    """Estimate cost for processing text using a specific model."""
    content = _read_input(input_path)
    calc = CostCalculator(model)
    result_input = calc.estimate_prompt(content, model)
    result_output = calc.estimate_completion(content, model)
    
    console.print(f"Model: [bold blue]{model}[/bold blue]")
    console.print(f"Tokens: {result_input.tokens:,}")
    console.print(f"Estimated Input Cost:  ${result_input.usd:.6f}")
    console.print(f"Estimated Output Cost: ${result_output.usd:.6f} (assuming same length output)")


@app.command()
def doctor() -> None:
    """Health check: display environment and model status."""
    console.print("[bold green]ContextForge Doctor[/bold green]")
    try:
        import tiktoken
        console.print("✅ tiktoken installed")
    except ImportError:
        console.print("❌ tiktoken is missing! OpenAI models will use estimate backend.")
    console.print(f"✅ Registered models: {len(ModelRegistry.list_models())}")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", help="API host"),
    port: int = typer.Option(8000, "--port", help="API port"),
) -> None:
    """Start the ContextForge API server."""
    import uvicorn
    console.print(f"Starting ContextForge API at http://{host}:{port}")
    console.print(f"Docs: http://{host}:{port}/docs")
    uvicorn.run("contextforge.api.app:app", host=host, port=port, reload=False)


@app.command()
def demo() -> None:
    """Run a quick interactive demo showcasing features."""
    console.print("=" * 60)
    console.print("  [bold]ContextForge[/bold] — Production LLMOps Infrastructure")
    console.print("=" * 60)
    console.print()

    # 1. Token counting
    counter = TokenCounter("gpt-4o")
    sample = "ContextForge helps you manage LLM context windows, count tokens accurately, chunk documents intelligently, and compress prompts to fit within model limits."
    tokens = counter.count(sample)
    cost_est = counter.estimate_cost(sample)
    console.print("📊 [bold]Token Counting[/bold]")
    console.print(f"   Text:   \"{sample[:60]}...\"")
    console.print(f"   Tokens: {tokens}")
    console.print(f"   Cost:   ${cost_est:.6f}")
    console.print()

    # 2. Chunking
    chunker = DocumentChunker()
    long_text = (sample + " ") * 20
    chunks = chunker.chunk(long_text, ChunkStrategy.SENTENCE, max_tokens=50)
    console.print("✂️  [bold]Smart Chunking[/bold]")
    console.print(f"   Input:  {counter.count(long_text)} tokens")
    console.print(f"   Chunks: {len(chunks)} (max 50 tokens each)")
    console.print()

    # 3. Context assembly
    window = ContextWindow()
    window.add_block("You are a helpful assistant.", Priority.CRITICAL, "system")
    window.add_block("The user's project is a Python web app.", Priority.HIGH, "project_info")
    window.add_block("Here is the full changelog...", Priority.LOW, "changelog")
    window.assemble(max_tokens=100)
    usage = window.usage()
    console.print("🧩 [bold]Priority-Based Context Assembly[/bold]")
    console.print(f"   Blocks:   {usage['num_blocks']}")
    console.print(f"   Included: {usage['num_included']}")
    console.print(f"   Excluded: {usage['num_excluded']}")
    console.print()

    # 4. Compression
    compressor = ContextCompressor()
    result = compressor.compress(long_text, target_tokens=30, strategy=CompressionStrategy.EXTRACTIVE)
    console.print("🗜️  [bold]Context Compression[/bold]")
    console.print(f"   Original:   {result.original_tokens} tokens")
    console.print(f"   Compressed: {result.compressed_tokens} tokens")
    console.print(f"   Savings:    {result.savings_pct:.1f}%")
    console.print()


if __name__ == "__main__":
    app()
