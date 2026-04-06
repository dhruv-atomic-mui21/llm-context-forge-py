from fastapi import APIRouter, HTTPException
from llm_context_forge.api.schemas import CompressRequest, CompressResponse
from llm_context_forge.compressor import ContextCompressor, CompressionStrategy

router = APIRouter()

@router.post("/", response_model=CompressResponse)
async def compress_text(req: CompressRequest):
    """Compress text to a target token count."""
    try:
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
