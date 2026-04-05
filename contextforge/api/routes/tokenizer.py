from fastapi import APIRouter, HTTPException
from contextforge.api.schemas import TokenCountRequest, TokenCountResponse, TokenValidateRequest, TokenValidateResponse
from contextforge.tokenizer import TokenCounter

router = APIRouter()

@router.post("/count", response_model=TokenCountResponse)
async def count_tokens(req: TokenCountRequest):
    """Count tokens for the given text and model."""
    try:
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

@router.post("/validate", response_model=TokenValidateResponse)
async def validate_tokens(req: TokenValidateRequest):
    """Check whether text fits within a token limit."""
    try:
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
