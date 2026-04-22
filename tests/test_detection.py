import os
import pytest
import numpy as np

from detection.retriever import FAISSRetriever
from detection.scorer import RetrievalSimilarityScorer, NLIEntailmentScorer, SemanticEntropyScorer
from detection.span_detector import SpanLevelDetector
from detection.ensemble import HallucinationDetector

class MockEmbedder:
    def encode(self, texts, **kwargs):
        if not texts:
            return np.array([])
        # Fixed deterministic seeding allows predictable norms 
        np.random.seed(42)
        return np.random.rand(len(texts), 16).astype(np.float32)
        
    def encode_single(self, text):
        return self.encode([text])[0]
        
    def normalize(self, vecs):
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms

class MockIndex:
    def search(self, query, k):
        # Fake responses guaranteeing alignment
        distances = np.array([[0.95] * k])
        indices = np.array([[0] * k])
        return distances, indices

@pytest.fixture
def retriever():
    chunks = [{"text": "First background textual entity describing conditions.", "id": "1"}]
    return FAISSRetriever(chunks, MockIndex(), MockEmbedder(), top_k=1)

@pytest.fixture
def sim_scorer():
    return RetrievalSimilarityScorer(MockEmbedder())

@pytest.fixture
def nli_scorer():
    # Substitute heavy sequence classifer directly without destroying API interfaces
    class MockNLI:
        def score(self, r, c):
            return {"entailment_prob": 0.8, "contradiction_prob": 0.15, "neutral_prob": 0.05, "nli_label": "entailment"}
    return MockNLI()

@pytest.fixture
def entropy_scorer():
    return SemanticEntropyScorer(MockEmbedder())

@pytest.fixture
def span_detector(sim_scorer, nli_scorer):
    return SpanLevelDetector(sim_scorer, nli_scorer)

def test_retriever(retriever):
    res = retriever.retrieve("What is something?", k=1)
    assert len(res) == 1
    assert "score" in res[0]
    
    ctx = retriever.get_context_window(res, max_tokens=100)
    assert ctx == "First background textual entity describing conditions."
    
    # Edge case
    empty_res = retriever.retrieve("")
    assert len(empty_res) == 0

def test_sim_scorer(sim_scorer):
    # Standard 
    res = sim_scorer.score("Dogs walk.", [{"text": "Dogs usually walk."}])
    assert "similarity_score" in res
    assert "coverage" in res
    
    # Edge case 
    empty_res = sim_scorer.score("", [])
    assert empty_res["similarity_score"] == 0.0

def test_span_detector(span_detector):
    spans = span_detector.detect_hallucinated_spans("Sentence one. Sentence two.", [{"text": "Sentence one is true."}])
    assert len(spans) == 2
    assert "is_hallucinated" in spans[0]
    
    html = span_detector.to_html("Sentence one. Sentence two.", spans)
    assert type(html) is str
    
    # Empty 
    spans_empty = span_detector.detect_hallucinated_spans("", [])
    assert len(spans_empty) == 0

def test_ensemble(retriever, sim_scorer, nli_scorer, entropy_scorer, span_detector):
    detector = HallucinationDetector(retriever, sim_scorer, nli_scorer, entropy_scorer, span_detector)
    res = detector.detect("query?", "Response text block.", ["Response text block."])
    
    assert "hallucination_score" in res
    assert isinstance(res["is_hallucinated"], bool)
    assert res["latency_ms"] >= 0.0
    
    # Empty detection mapping handles appropriately without catastrophic traceback
    res_empty = detector.detect("query?", "")
    assert res_empty["is_hallucinated"] is False
    assert res_empty["hallucination_score"] == 0.0
    
    # Batch operations
    batch_res = detector.detect_batch([{"query": "1", "response": "a"}, {"query": "2", "response": "b"}])
    assert len(batch_res) == 2
