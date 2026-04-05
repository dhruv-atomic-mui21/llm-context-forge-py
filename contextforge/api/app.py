"""
ContextForge — FastAPI Application

Production LLMOps Infrastructure HTTP API.
"""

from typing import List, Dict, Any
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from contextforge.models import ModelRegistry
from contextforge.api.routes import tokenizer, chunker, context, compression, cost
from contextforge.api.schemas import HealthResponse

# ── App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="ContextForge API",
    description="LLMOps Infrastructure HTTP API for token counting, chunking, and context management",
    version="1.0.0",
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────

app.include_router(tokenizer.router, prefix="/api/v1/tokens", tags=["tokens"])
app.include_router(chunker.router, prefix="/api/v1/chunks", tags=["chunks"])
app.include_router(context.router, prefix="/api/v1/context", tags=["context"])
app.include_router(compression.router, prefix="/api/v1/compress", tags=["compression"])
app.include_router(cost.router, prefix="/api/v1/cost", tags=["cost"])

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health():
    """Comprehensive health check"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        providers=len(ModelRegistry.list_models())
    )

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="ContextForge API",
        version="1.0.0",
        description="Production LLMOps Infrastructure HTTP API.",
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
