import logging
import nltk
from typing import List, Dict

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SpanLevelDetector:
    """Detects and isolates hallucinated textual spans directly inside an unstructured block."""
    def __init__(self, sim_scorer, nli_scorer):
        self.sim_scorer = sim_scorer
        self.nli_scorer = nli_scorer

    def detect_hallucinated_spans(self, response: str, passages: List[Dict]) -> List[Dict]:
        """Iterates sequentially highlighting sentence-level defects."""
        if not response:
            return []
            
        sentences = nltk.sent_tokenize(response)
        spans = []
        start_idx = 0
        
        for sent in sentences:
            start = response.find(sent, start_idx)
            end = start + len(sent)
            start_idx = end
            
            # Step 1: Handling edgecases natively without supporting passages
            if not passages:
                spans.append({
                    "span": sent,
                    "start": start,
                    "end": end,
                    "is_hallucinated": True,
                    "confidence": 1.0,
                    "supporting_text": None
                })
                continue
            
            # Step 2: Compute Retrieval Score Matrix 
            sim_res = self.sim_scorer.score(sent, passages)
            sim = sim_res.get("similarity_score", 0.0)
            top_passage = sim_res.get("top_passage", "")
            
            # Step 3: Compute isolated NLI Score
            nli_res = self.nli_scorer.score(sent, top_passage)
            contradiction_prob = nli_res.get("contradiction_prob", 0.0)
            
            # Step 4: Ensemble Mathematical Weightings
            # Low similarity boosts hallucination factor heavily, contradiction pushes it over edge.
            combined_score = 0.5 * (1 - sim) + 0.5 * contradiction_prob
            
            # Step 5: Decision bounds
            is_hallucinated = bool(combined_score > 0.5)
            
            spans.append({
                "span": sent,
                "start": start,
                "end": end,
                "is_hallucinated": is_hallucinated,
                "confidence": float(combined_score),
                "supporting_text": top_passage if top_passage else None
            })
            
        return spans

    def to_html(self, response: str, spans: List[Dict]) -> str:
        """Inject structured payload markers cleanly into strings dynamically tracing coordinate boundaries."""
        if not response:
            return ""
            
        # Reverse parsing maintains safe bounds modifications
        sorted_spans = sorted(spans, key=lambda x: x["start"], reverse=True)
        html = response
        
        for span in sorted_spans:
            if span["is_hallucinated"]:
                s = span["start"]
                e = span["end"]
                html = html[:s] + f'<mark class="hallucinated">{html[s:e]}</mark>' + html[e:]
                
        return html
