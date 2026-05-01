import os
import time
import json
import logging
import asyncio
import hashlib
from functools import lru_cache
from uuid import uuid4
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from concurrent.futures import ThreadPoolExecutor

from config import config
from api.models import (
    DetectionRequest, DetectionResponse, 
    BatchDetectionRequest, BatchDetectionResponse,
    HealthResponse, IndexBuildRequest, MetricsResponse
)
from api.middleware import setup_middlewares

from knowledge_base.embedder import SentenceEmbedder
from knowledge_base.faiss_index import FAISSIndexManager
from knowledge_base.builder import KnowledgeBaseBuilder
from detection.retriever import FAISSRetriever
from detection.scorer import RetrievalSimilarityScorer, NLIEntailmentScorer, SemanticEntropyScorer
from detection.span_detector import SpanLevelDetector
from detection.ensemble import HallucinationDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM Hallucination Detector Core API", version="0.1.0")
setup_middlewares(app)

# Application Global Artifact Maps
app_state = {
    "start_time": time.time(),
    "is_loaded": False,
    "builder": None,
    "detector": None,
    "retriever": None,
    "metrics": {
        "total": 0,
        "total_latency_ms": 0.0,
        "hallucination_count": 0
    }
}

@app.on_event("startup")
async def startup_event():
    """Application bounds, configures environment paths, initializes NLP submodules into hot GPU memory blocks."""
    logger.info("Initializing Application Core Modules...")
    try:
        embedder = SentenceEmbedder(model_name=config.EMBED_MODEL)
        builder = KnowledgeBaseBuilder(embedder=embedder)
        
        # Load KB conditionally avoiding failure if directory structure hasn't synced
        if os.path.exists(config.faiss_index_path) and os.path.exists(os.path.join(config.faiss_index_path, "faiss_index.bin")):
            chunks, index_manager, index = builder.load(config.faiss_index_path)
            retriever = FAISSRetriever(chunks, index, embedder, top_k=config.TOP_K_DEFAULT)
        else:
            chunks, index, retriever = [], None, None
            logger.warning(f"No fully realized KB FAISS index available at {config.faiss_index_path}. Build manually required.")

        app_state["builder"] = builder
        
        sim_scorer = RetrievalSimilarityScorer(embedder)
        nli_scorer = NLIEntailmentScorer(model_name=config.NLI_MODEL)
        entropy_scorer = SemanticEntropyScorer(embedder)
        span_detector = SpanLevelDetector(sim_scorer, nli_scorer)
        
        detector = HallucinationDetector(
            retriever, sim_scorer, nli_scorer, entropy_scorer, span_detector
        )
        
        app_state["detector"] = detector
        app_state["retriever"] = retriever
        
        # We consider models strictly 'hotloaded' if Index is ready to bind onto FAISS. 
        if index is not None:
            app_state["is_loaded"] = True
            logger.info("Detector ensemble initialized actively on core hooks.")
            
    except Exception as e:
        logger.error(f"FATAL: Application modules failed dependency checks during bounds loads: {e}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Trap traces converting to JSON bounds. Suppresses traces unless DEBUG mode active implicitly."""
    logger.exception(exc)
    dev_mode = os.getenv("ENV", "dev") == "dev"
    detail = str(exc) if dev_mode else "Internal Server Error."
    return JSONResponse(status_code=500, content={"detail": detail})

def _hash_request(req_dict: dict) -> str:
    s = json.dumps(req_dict, sort_keys=True)
    return hashlib.md5(s.encode()).hexdigest()

@lru_cache(maxsize=1000)
def _cached_detect(req_hash: str, req_json: str) -> dict:
    """Core wrapped operation mapping requests natively through cache hits utilizing serialized MD5 hashkeys."""
    req_dict = json.loads(req_json)
    detector = app_state["detector"]
    
    res = detector.detect(
        req_dict.get("query", ""),
        req_dict.get("response", ""),
        req_dict.get("sampled_responses", [])
    )
    
    if not req_dict.get("return_spans", True):
        res["spans"] = []
        res["highlighted_html"] = ""
    if not req_dict.get("return_passages", True):
        res["supporting_passages"] = []
        
    return res


@app.get("/health", response_model=HealthResponse)
async def health():
    """Probes status metadata, module binds, index topologies, and application uptime."""
    stats = {}
    if app_state["is_loaded"] and app_state["retriever"]:
        mgr = FAISSIndexManager()
        idx = app_state["retriever"].index
        stats = mgr.get_index_stats(idx)
        
    return {
        "status": "ok",
        "version": app.version,
        "model_loaded": app_state["is_loaded"],
        "index_stats": stats,
        "uptime_seconds": time.time() - app_state["start_time"]
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    """Processes textual payload bounds searching for hallucinatory properties."""
    if not app_state["is_loaded"]:
        raise HTTPException(status_code=503, detail="Offline Error. Offline Models/FAISS index uninitialized.")
        
    if not request.response.strip():
        raise HTTPException(status_code=422, detail="Validation Error: Response element parameter bounds empty.")
        
    req_dict = request.model_dump()
    req_json = json.dumps(req_dict)
    req_hash = _hash_request(req_dict)
    
    start_t = time.time()
    res = await asyncio.to_thread(_cached_detect, req_hash, req_json)
    latency = (time.time() - start_t) * 1000
    res["latency_ms"] = latency
    
    app_state["metrics"]["total"] += 1
    app_state["metrics"]["total_latency_ms"] += latency
    if res["is_hallucinated"]:
        app_state["metrics"]["hallucination_count"] += 1
        
    return res

@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(request: BatchDetectionRequest):
    """High throughput inference API parsing sequentially nested bounds via ThreadPools targeting GPU streams."""
    if not app_state["is_loaded"]:
        raise HTTPException(status_code=503, detail="Offline Error. Offline Models/FAISS index uninitialized.")
        
    start_t = time.time()
    
    def process_item(item):
        req_dict = item.model_dump()
        req_json = json.dumps(req_dict)
        req_hash = _hash_request(req_dict)
        return _cached_detect(req_hash, req_json)
        
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=request.max_workers) as pool:
        futures = [loop.run_in_executor(pool, process_item, item) for item in request.items]
        results = await asyncio.gather(*futures)
        
    latency = (time.time() - start_t) * 1000
    
    app_state["metrics"]["total"] += len(results)
    app_state["metrics"]["total_latency_ms"] += latency
    app_state["metrics"]["hallucination_count"] += sum(1 for r in results if r.get("is_hallucinated", False))
    
    return {
        "results": results,
        "total_latency_ms": latency
    }


def _build_index_task(documents: list, index_type: str, rebuild: bool):
    try:
        logger.info(f"Background thread job: Build Index over {len(documents)} objects bounds launched.")
        builder = app_state["builder"]
        
        if rebuild or not os.path.exists(config.faiss_index_path):
            builder.build_from_scratch(documents, index_type=index_type, save_dir=config.faiss_index_path)
        else:
            builder.add_documents(documents)
            
        logger.info("Background index build terminated clean bounds. Re-mapping index to current application pool...")
        
        chunks, index_manager, index = builder.load(config.faiss_index_path)
        app_state["retriever"] = FAISSRetriever(chunks, index, builder.embedder, top_k=config.TOP_K_DEFAULT)
        app_state["detector"].retriever = app_state["retriever"]
        app_state["is_loaded"] = True
        logger.info("Index hot-swapped into inference loops.")
        
    except Exception as e:
        logger.error(f"Task Failed. Fault in background index build: {e}")

@app.post("/index/build")
async def build_index(req: IndexBuildRequest, background_tasks: BackgroundTasks):
    """Triggers dataset re-index tasks implicitly behind REST constraints into asyncio threads."""
    job_id = str(uuid4())
    if not req.documents:
        raise HTTPException(status_code=422, detail="Empty boundaries passed. Documents array missing blocks.")
        
    background_tasks.add_task(_build_index_task, req.documents, req.index_type, req.rebuild)
    return {"status": "building", "job_id": job_id}


@app.get("/index/stats")
async def index_stats():
    """Probe FAISS specific memory heuristics remotely."""
    if not app_state["is_loaded"]:
        raise HTTPException(status_code=503, detail="Index offline.")
    mgr = FAISSIndexManager()
    idx = app_state["retriever"].index
    return mgr.get_index_stats(idx)


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Scrapes aggregated global variables for Prometheus bindings bounding general usage endpoints."""
    total = app_state["metrics"]["total"]
    uptime = time.time() - app_state["start_time"]
    
    avg_lat = app_state["metrics"]["total_latency_ms"] / total if total > 0 else 0.0
    hal_rate = app_state["metrics"]["hallucination_count"] / total if total > 0 else 0.0
    rpm = (total / uptime) * 60 if uptime > 0 else 0.0
    
    return {
        "total_requests": total,
        "avg_latency_ms": avg_lat,
        "hallucination_rate": hal_rate,
        "requests_per_minute": rpm
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
