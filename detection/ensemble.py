import time
import logging
import mlflow
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HallucinationDetector:
    """The unified multi-signal scoring ensemble interface handling all detection routines."""
    
    def __init__(self, retriever, sim_scorer, nli_scorer, entropy_scorer, span_detector):
        self.retriever = retriever
        self.sim_scorer = sim_scorer
        self.nli_scorer = nli_scorer
        self.entropy_scorer = entropy_scorer
        self.span_detector = span_detector
        
        self.w_sim = 0.40
        self.w_nli = 0.35
        self.w_entropy = 0.25

    def detect(self, query: str, response: str, sampled_responses: Optional[List[str]] = None) -> Dict:
        """Process unified Hallucination scores mapping onto MLFlow telemetry systems."""
        start_time = time.time()
        
        # Safely abort empty
        if not response or not str(response).strip():
            return {
                "query": query, "response": response, "is_hallucinated": False,
                "confidence": 0.0, "hallucination_score": 0.0,
                "signals": {"retrieval_similarity": 0.0, "nli_entailment": 0.0, "semantic_entropy": 0.0},
                "spans": [], "highlighted_html": "", "supporting_passages": [],
                "latency_ms": 0.0, "index_type_used": "unknown"
            }
            
        passages = self.retriever.retrieve(query)
        context = self.retriever.get_context_window(passages)
        
        sim_res = self.sim_scorer.score(response, passages)
        sim_score = sim_res["similarity_score"]
        
        nli_res = self.nli_scorer.score(response, context)
        contradiction = nli_res["contradiction_prob"]
        entailment = nli_res["entailment_prob"]
        
        entropy = 0.0
        if sampled_responses and len(sampled_responses) > 1:
            ent_res = self.entropy_scorer.score(sampled_responses)
            entropy = ent_res["semantic_entropy"]
            
        # Normalization constraints
        # Max entropy for 5 distinct samples is ~1.6. Bound roughly to limit overflow
        norm_entropy = min(entropy / 2.0, 1.0)
        
        # High score means highly hallucinated. High similarity (sim_score) lowers it.
        hallucination_score = (self.w_sim * (1.0 - sim_score)) + \
                              (self.w_nli * contradiction) + \
                              (self.w_entropy * norm_entropy)
                              
        is_hallucinated = bool(hallucination_score > 0.5)
        
        spans = self.span_detector.detect_hallucinated_spans(response, passages)
        highlighted = self.span_detector.to_html(response, spans)
        
        latency_ms = float((time.time() - start_time) * 1000)
        
        index_type = "unknown"
        if hasattr(self.retriever.index, "__class__"):
            index_type = self.retriever.index.__class__.__name__
            
        result = {
            "query": query,
            "response": response,
            "is_hallucinated": is_hallucinated,
            "confidence": float(hallucination_score),
            "hallucination_score": float(hallucination_score),
            "signals": {
                "retrieval_similarity": sim_score,
                "nli_entailment": entailment,
                "semantic_entropy": entropy
            },
            "spans": spans,
            "highlighted_html": highlighted,
            "supporting_passages": passages,
            "latency_ms": latency_ms,
            "index_type_used": index_type
        }
        
        # Push artifacts telemetry to internal experiment server
        if mlflow.active_run():
            mlflow.log_metrics({
                "hallucination_score": hallucination_score,
                "latency_ms": latency_ms
            })
            mlflow.log_params({
                "is_hallucinated": is_hallucinated,
                "n_passages": len(passages)
            })
            
        return result

    def detect_batch(self, items: List[Dict], show_progress: bool = True) -> List[Dict]:
        """Aggregate block routines over collections sequentially."""
        results = []
        for i, item in enumerate(items):
            if show_progress and i % 10 == 0:
                logger.info(f"Processing batch block {i}/{len(items)}")
            query = item.get("query", "")
            response = item.get("response", "")
            samples = item.get("sampled_responses", [])
            results.append(self.detect(query, response, samples))
        return results
