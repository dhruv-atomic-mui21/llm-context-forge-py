"""
Pydantic Schemas for the API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

# ── Health ────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    providers: int

# ── Tokenizer ─────────────────────────────────────────────────────────

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

# ── Chunker ───────────────────────────────────────────────────────────

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

# ── Context ───────────────────────────────────────────────────────────

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

# ── Compression ───────────────────────────────────────────────────────

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

# ── Cost ──────────────────────────────────────────────────────────────

class CostEstimateRequest(BaseModel):
    text: str
    model: str = "gpt-4o"
    
class CostEstimateResponse(BaseModel):
    input_tokens: int
    input_cost_usd: float
    model: str
