import time
import logging
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HallucinationDetector:
    """
    Multi-signal hallucination detection system:
    Combines retrieval similarity, NLI contradiction, and entropy.
    """

    def __init__(self, retriever, sim_scorer, nli_scorer, entropy_scorer, span_detector):
        self.retriever = retriever
        self.sim_scorer = sim_scorer
        self.nli_scorer = nli_scorer
        self.entropy_scorer = entropy_scorer
        self.span_detector = span_detector

        # Weights for ensemble scoring
        self.w_sim = 0.40
        self.w_nli = 0.35
        self.w_entropy = 0.25

    def detect(self, query: str, response: str, sampled_responses: Optional[List[str]] = None) -> Dict:
        """Run hallucination detection on a single query-response pair."""

        start_time = time.time()

        # Handle empty response safely
        if not response or not str(response).strip():
            return {
                "query": query,
                "response": response,
                "is_hallucinated": False,
                "confidence": 0.0,
                "hallucination_score": 0.0,
                "signals": {
                    "retrieval_similarity": 0.0,
                    "nli_entailment": 0.0,
                    "semantic_entropy": 0.0
                },
                "spans": [],
                "highlighted_html": "",
                "supporting_passages": [],
                "latency_ms": 0.0,
                "index_type_used": "unknown"
            }

        # -----------------------------
        # Retrieval
        # -----------------------------
        passages = self.retriever.retrieve(query)
        context = self.retriever.get_context_window(passages)

        # -----------------------------
        # Similarity scoring
        # -----------------------------
        sim_res = self.sim_scorer.score(response, passages)
        sim_score = sim_res.get("similarity_score", 0.0)

        # -----------------------------
        # NLI scoring
        # -----------------------------
        nli_res = self.nli_scorer.score(response, context)
        contradiction = nli_res.get("contradiction_prob", 0.0)
        entailment = nli_res.get("entailment_prob", 0.0)

        # -----------------------------
        # Entropy scoring
        # -----------------------------
        entropy = 0.0
        if sampled_responses and len(sampled_responses) > 1:
            ent_res = self.entropy_scorer.score(sampled_responses)
            entropy = ent_res.get("semantic_entropy", 0.0)

        # Normalize entropy
        norm_entropy = min(entropy / 2.0, 1.0)

        # -----------------------------
        # Final ensemble score
        # -----------------------------
        hallucination_score = (
            (self.w_sim * (1.0 - sim_score)) +
            (self.w_nli * contradiction) +
            (self.w_entropy * norm_entropy)
        )

        is_hallucinated = hallucination_score > 0.5

        # -----------------------------
        # Span detection
        # -----------------------------
        spans = self.span_detector.detect_hallucinated_spans(response, passages)
        highlighted = self.span_detector.to_html(response, spans)

        latency_ms = (time.time() - start_time) * 1000

        index_type = getattr(self.retriever.index, "__class__", type("Unknown", (), {})).__name__

        # -----------------------------
        # Final result
        # -----------------------------
        result = {
            "query": query,
            "response": response,
            "is_hallucinated": bool(is_hallucinated),
            "confidence": float(hallucination_score),
            "hallucination_score": float(hallucination_score),
            "signals": {
                "retrieval_similarity": float(sim_score),
                "nli_entailment": float(entailment),
                "semantic_entropy": float(entropy)
            },
            "spans": spans,
            "highlighted_html": highlighted,
            "supporting_passages": passages,
            "latency_ms": float(latency_ms),
            "index_type_used": index_type
        }

        return result

    def detect_batch(self, items: List[Dict], show_progress: bool = True) -> List[Dict]:
        """Run detection over multiple items."""

        results = []

        for i, item in enumerate(items):
            if show_progress and i % 10 == 0:
                logger.info(f"Processing {i}/{len(items)}")

            query = item.get("query", "")
            response = item.get("response", "")
            samples = item.get("sampled_responses", [])

            results.append(self.detect(query, response, samples))

        return results