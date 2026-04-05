from fastapi import APIRouter, HTTPException
from layoutlm_forge.api.schemas import CostEstimateRequest, CostEstimateResponse
from layoutlm_forge.cost import CostCalculator

router = APIRouter()

@router.post("/estimate", response_model=CostEstimateResponse)
async def estimate_cost(req: CostEstimateRequest):
    """Estimate cost for processing text using a specific model."""
    try:
        calc = CostCalculator(req.model)
        result_input = calc.estimate_prompt(req.text, req.model)
        
        return CostEstimateResponse(
            input_tokens=result_input.tokens,
            input_cost_usd=result_input.usd,
            model=req.model,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
