import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer

class RandomBaseline:
    """Provides internal negative minimum bounds establishing random float matrices logic validations."""
    @property
    def name(self) -> str:
        return "RandomBaseline"
        
    def predict(self, query: str, response: str, **kwargs) -> float:
        return float(np.random.rand())


class LexicalSimilarityBaseline:
    """Provides Bag-Of-Words heuristics to cross reference transformer pipeline regressions."""
    @property
    def name(self) -> str:
        return "LexicalSimilarityBaseline"
        
    def predict(self, response: str, passages: list[str], **kwargs) -> float:
        if not response or not passages:
            return 1.0 
            
        vectorizer = TfidfVectorizer().fit([response] + passages)
        resp_vec = vectorizer.transform([response])
        pass_vecs = vectorizer.transform(passages)
        
        sims = cosine_similarity(resp_vec, pass_vecs)[0]
        max_sim = np.max(sims) if sims.size > 0 else 0.0
        
        # Hallucination prob scales inversely 
        return 1.0 - float(max_sim)


class SelfCheckGPTLite:
    """Third Baseline scaling sample drift thresholds tracking textual overlaps mapping inconsistency values internally."""
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        
    @property
    def name(self) -> str:
        return "SelfCheckGPT-lite"
        
    def predict(self, sampled_responses: list[str], **kwargs) -> float:
        if not sampled_responses or len(sampled_responses) <= 1:
            return 0.0
            
        pairwise_scores = []
        for i in range(len(sampled_responses)):
            for j in range(len(sampled_responses)):
                if i != j:
                    score = self.scorer.score(sampled_responses[i], sampled_responses[j])
                    pairwise_scores.append(score['rougeL'].fmeasure)
                    
        mean_rouge = np.mean(pairwise_scores) if pairwise_scores else 0.0
        # High structural inconsistency maps highly probable hallucination bounds natively.
        return 1.0 - float(mean_rouge)

if __name__ == "__main__":
    bl1 = RandomBaseline()
    bl2 = LexicalSimilarityBaseline()
    bl3 = SelfCheckGPTLite()
    
    print(bl1.name, "->", bl1.predict("q", "r"))
    print(bl2.name, "->", bl2.predict("resp", ["passage 1", "exact identical resp bounds mapping"]))
    print(bl3.name, "->", bl3.predict(["Response 1 exactly", "Response 2 completely different drift mapping."]))
