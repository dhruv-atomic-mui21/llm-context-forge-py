from fastapi import APIRouter, HTTPException
from layoutlm_forge.api.schemas import ChunkRequest, ChunkResponse
from layoutlm_forge.chunker import DocumentChunker, ChunkStrategy

router = APIRouter()

@router.post("/", response_model=ChunkResponse)
async def chunk_text(req: ChunkRequest):
    """Chunk text using the specified strategy."""
    try:
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
