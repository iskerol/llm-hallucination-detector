from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field

class DetectionRequest(BaseModel):
    query: str
    response: str
    sampled_responses: Optional[List[str]] = None
    top_k: int = 10
    index_type: Literal["FlatIP", "IVFFlat", "HNSW"] = "IVFFlat"
    return_spans: bool = True
    return_passages: bool = True

class SignalScores(BaseModel):
    retrieval_similarity: float
    nli_entailment: float
    semantic_entropy: float

class SpanItem(BaseModel):
    span: str
    start: int
    end: int
    is_hallucinated: bool
    confidence: float
    supporting_text: Optional[str] = None

class DetectionResponse(BaseModel):
    query: str
    response: str
    is_hallucinated: bool
    confidence: float
    hallucination_score: float
    signals: SignalScores
    spans: List[SpanItem]
    highlighted_html: str
    supporting_passages: List[Dict[str, Any]]
    latency_ms: float
    index_type_used: str

class BatchDetectionRequest(BaseModel):
    items: List[DetectionRequest]
    max_workers: int = Field(default=4, ge=1, le=16)

class BatchDetectionResponse(BaseModel):
    results: List[DetectionResponse]
    total_latency_ms: float

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    index_stats: Dict[str, Any]
    uptime_seconds: float

class IndexBuildRequest(BaseModel):
    documents: List[Dict[str, Any]]
    index_type: str = "IVFFlat"
    rebuild: bool = False

class MetricsResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    hallucination_rate: float
    requests_per_minute: float
