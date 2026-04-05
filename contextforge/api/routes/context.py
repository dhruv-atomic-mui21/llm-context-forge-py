from fastapi import APIRouter, HTTPException
from contextforge.api.schemas import ContextAssembleRequest, ContextAssembleResponse
from contextforge.context import ContextWindow, Priority

router = APIRouter()

@router.post("/assemble", response_model=ContextAssembleResponse)
async def assemble_context(req: ContextAssembleRequest):
    """Assemble context blocks by priority."""
    try:
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
