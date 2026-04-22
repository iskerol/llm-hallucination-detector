import logging
from typing import List, Dict, Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSRetriever:
    """Retriever utilizing FAISS Index and Embedders."""
    def __init__(self, chunks: List[Dict], index, embedder, top_k: int = 10):
        self.chunks = chunks
        self.index = index
        self.embedder = embedder
        self.top_k = top_k
        
    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        """Retrieve top k passages for a single query."""
        if not query:
            return []
            
        k = k or self.top_k
        try:
            q_emb = self.embedder.encode_single(query).reshape(1, -1)
            q_norm = self.embedder.normalize(q_emb).astype(np.float32)
            distances, indices = self.index.search(q_norm, k)
        except Exception as e:
            logger.error(f"Failed to search index: {e}")
            return []
            
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(distances[0][i])
            chunk["rank"] = i + 1
            results.append(chunk)
            
        return results

    def retrieve_batch(self, queries: List[str], k: int) -> List[List[Dict]]:
        """Retrieve top k passages for a batch of queries."""
        if not queries:
            return []
            
        try:
            q_embs = self.embedder.encode(queries, show_progress=False)
            q_norms = self.embedder.normalize(q_embs).astype(np.float32)
            distances, indices = self.index.search(q_norms, k)
        except Exception as e:
            logger.error(f"Failed to search index in batch: {e}")
            return [[] for _ in queries]
            
        batch_results = []
        for q_idx in range(len(queries)):
            res = []
            for i, idx in enumerate(indices[q_idx]):
                if idx == -1 or idx >= len(self.chunks):
                    continue
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(distances[q_idx][i])
                chunk["rank"] = i + 1
                res.append(chunk)
            batch_results.append(res)
            
        return batch_results

    def get_context_window(self, results: List[Dict], max_tokens: int = 1024) -> str:
        """Merge top retrieval distinct textual pieces into a unified string payload safely bounded by max_tokens."""
        if not results:
            return ""
            
        context = []
        current_len = 0
        
        for res in results:
            text = res.get("text", "")
            if not text:
                continue
                
            # Naive token approximation (1 Word ~= ~1.3 Tokens)
            tokens = int(len(text.split()) * 1.3)
            if current_len + tokens > max_tokens:
                break
                
            context.append(text)
            current_len += tokens
            
        return " ".join(context)
