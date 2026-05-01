import asyncio
import hashlib
import time
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded  # type: ignore
from slowapi.util import get_remote_address

# 🔥 Import pipeline
from pipeline import run_pipeline_async

# =========================
# ⚙️ Setup & Metrics
# =========================

UPTIME_START = time.time()

_metrics = {
    "total": 0,
    "hits": 0,
    "latencies": [],
    "hallucinated": 0
}

_cache = {}
MAX_CACHE_SIZE = 500

limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="RUC-Detect API",
    version="1.0.0",
    description="Production API for real-time hallucination detection."
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# =========================
# 📦 Schemas
# =========================

class DetectRequest(BaseModel):
    prompt: str = Field(..., max_length=2000)
    response: str = Field(..., max_length=5000)
    model_id: Optional[str] = Field("llama-3-8b")
    sampled_responses: Optional[List[str]] = None


class Span(BaseModel):
    start: int
    end: int
    text: str
    confidence: float


class DetectResponse(BaseModel):
    hallucination_score: float
    is_hallucinated: bool
    explanation: str
    hallucination_pattern: Optional[str]
    flagged_spans: List[Span]
    component_scores: Dict[str, float]
    cache_hit: bool

# =========================
# 🟢 Health Check & Metrics
# =========================

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.get("/metrics")
async def get_metrics():
    hits = _metrics["hits"]
    total = _metrics["total"]
    cache_hit_rate = (hits / total) if total > 0 else 0.0

    lats = _metrics["latencies"]
    avg_latency = (sum(lats) / len(lats)) if lats else 0.0

    hallucinated = _metrics["hallucinated"]
    hallucination_rate = (hallucinated / total) if total > 0 else 0.0

    return {
        "total_requests": int(total),
        "cache_hits": int(hits),
        "cache_hit_rate": float(cache_hit_rate),
        "avg_latency_ms": float(avg_latency),
        "hallucination_rate": float(hallucination_rate),
        "uptime_seconds": float(time.time() - UPTIME_START)
    }

# =========================
# 🚀 Detection Endpoint
# =========================

@app.post("/detect", response_model=DetectResponse)
@limiter.limit("50/minute")
async def detect_hallucination(request: Request, payload: DetectRequest):
    _metrics["total"] += 1
    start_t = time.time()

    cache_hash = hashlib.sha256(f"{payload.prompt}|{payload.response}".encode()).hexdigest()

    if cache_hash in _cache:
        _metrics["hits"] += 1
        result = _cache[cache_hash]
        latency = (time.time() - start_t) * 1000
        _metrics["latencies"].append(latency)

        if result["label"]:
            _metrics["hallucinated"] += 1

        return DetectResponse(
            hallucination_score=result["score"],
            is_hallucinated=result["label"],
            explanation=result["explanation"],
            hallucination_pattern=result["pattern"],
            flagged_spans=[
                Span(
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    confidence=s["confidence"]
                )
                for s in result["spans"]
            ],
            component_scores=result["components"],
            cache_hit=True
        )

    try:
        result = await asyncio.wait_for(
            run_pipeline_async(payload.prompt, payload.response, payload.sampled_responses),
            timeout=30.0
        )
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Pipeline execution timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if len(_cache) >= MAX_CACHE_SIZE:
        _cache.pop(next(iter(_cache)))
    _cache[cache_hash] = result

    latency = (time.time() - start_t) * 1000
    _metrics["latencies"].append(latency)

    if result["label"]:
        _metrics["hallucinated"] += 1

    return DetectResponse(
        hallucination_score=result["score"],
        is_hallucinated=result["label"],
        explanation=result["explanation"],
        hallucination_pattern=result["pattern"],
        flagged_spans=[
            Span(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                confidence=s["confidence"]
            )
            for s in result["spans"]
        ],
        component_scores=result["components"],
        cache_hit=False
    )

# =========================
# ▶️ Run Server
# =========================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
