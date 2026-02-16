"""
ContextForge — FastAPI Application

REST API for token counting, document chunking, context assembly,
and context compression.
"""

import os
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ContextForge API",
    description=(
        "LLM context & token handling toolkit. "
        "Count tokens, chunk documents, assemble context windows, "
        "and compress prompts."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


# ── Request / Response Models ─────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str


class TokenCountRequest(BaseModel):
    text: str
    model: str = "gpt-4o"


class TokenCountResponse(BaseModel):
    tokens: int
    model: str
    context_window: int
    fits: bool
    estimated_cost_input: float


class TokenValidateRequest(BaseModel):
    text: str
    max_tokens: int = 4096
    model: str = "gpt-4o"
    reserve_output: int = 0


class TokenValidateResponse(BaseModel):
    fits: bool
    token_count: int
    max_tokens: int
    remaining: int


class ChunkRequest(BaseModel):
    text: str
    strategy: str = "paragraph"
    max_tokens: int = 500
    overlap_tokens: int = 50
    model: str = "gpt-4o"


class ChunkResponse(BaseModel):
    chunks: List[Dict[str, Any]]
    num_chunks: int
    total_tokens: int


class ContextBlockInput(BaseModel):
    content: str
    priority: str = "MEDIUM"
    label: str = ""


class ContextAssembleRequest(BaseModel):
    blocks: List[ContextBlockInput]
    max_tokens: int = 4096
    model: str = "gpt-4o"


class ContextAssembleResponse(BaseModel):
    assembled: str
    usage: Dict[str, Any]


class CompressRequest(BaseModel):
    text: str
    target_tokens: int
    strategy: str = "extractive"
    model: str = "gpt-4o"


class CompressResponse(BaseModel):
    text: str
    original_tokens: int
    compressed_tokens: int
    ratio: float
    savings_pct: float


# ── Endpoints ─────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with API info."""
    return HealthResponse(status="running", version="1.0.0")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check."""
    return HealthResponse(status="healthy", version="1.0.0")


@app.post("/tokens/count", response_model=TokenCountResponse)
async def count_tokens(req: TokenCountRequest):
    """Count tokens for the given text and model."""
    try:
        from src.tokenizer import TokenCounter

        counter = TokenCounter(req.model)
        tokens = counter.count(req.text, req.model)
        info = counter.get_model_info(req.model)
        cost = counter.estimate_cost(req.text, req.model, "input")

        return TokenCountResponse(
            tokens=tokens,
            model=req.model,
            context_window=info.context_window,
            fits=tokens <= info.context_window,
            estimated_cost_input=round(cost, 6),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tokens/validate", response_model=TokenValidateResponse)
async def validate_tokens(req: TokenValidateRequest):
    """Check whether text fits within a token limit."""
    try:
        from src.tokenizer import TokenCounter

        counter = TokenCounter(req.model)
        token_count = counter.count(req.text, req.model)
        effective_limit = req.max_tokens - req.reserve_output

        return TokenValidateResponse(
            fits=token_count <= effective_limit,
            token_count=token_count,
            max_tokens=req.max_tokens,
            remaining=effective_limit - token_count,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chunk", response_model=ChunkResponse)
async def chunk_text(req: ChunkRequest):
    """Chunk text using the specified strategy."""
    try:
        from src.chunker import DocumentChunker, ChunkStrategy

        strategy_map = {s.value: s for s in ChunkStrategy}
        strategy = strategy_map.get(req.strategy.lower())
        if strategy is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy. Choose from: {list(strategy_map.keys())}",
            )

        chunker = DocumentChunker(req.model)
        chunks = chunker.chunk(
            req.text,
            strategy=strategy,
            max_tokens=req.max_tokens,
            overlap_tokens=req.overlap_tokens,
            model=req.model,
        )

        chunk_dicts = [
            {
                "index": c.index,
                "text": c.text,
                "token_count": c.token_count,
                "char_count": c.char_count,
            }
            for c in chunks
        ]
        total_tokens = sum(c.token_count for c in chunks)

        return ChunkResponse(
            chunks=chunk_dicts,
            num_chunks=len(chunks),
            total_tokens=total_tokens,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/context/assemble", response_model=ContextAssembleResponse)
async def assemble_context(req: ContextAssembleRequest):
    """Assemble context blocks by priority."""
    try:
        from src.context import ContextWindow, Priority

        priority_map = {p.name: p for p in Priority}
        window = ContextWindow(req.model)

        for block in req.blocks:
            p = priority_map.get(block.priority.upper(), Priority.MEDIUM)
            window.add_block(block.content, priority=p, label=block.label)

        assembled = window.assemble(max_tokens=req.max_tokens)

        return ContextAssembleResponse(
            assembled=assembled,
            usage=window.usage(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compress", response_model=CompressResponse)
async def compress_text(req: CompressRequest):
    """Compress text to a target token count."""
    try:
        from src.compressor import ContextCompressor, CompressionStrategy

        strategy_map = {s.value: s for s in CompressionStrategy}
        strategy = strategy_map.get(req.strategy.lower())
        if strategy is None:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy. Choose from: {list(strategy_map.keys())}",
            )

        compressor = ContextCompressor(req.model)
        result = compressor.compress(
            req.text,
            target_tokens=req.target_tokens,
            strategy=strategy,
            model=req.model,
        )

        return CompressResponse(
            text=result.text,
            original_tokens=result.original_tokens,
            compressed_tokens=result.compressed_tokens,
            ratio=round(result.ratio, 4),
            savings_pct=round(result.savings_pct, 2),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
