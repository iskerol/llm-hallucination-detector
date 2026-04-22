import time
import logging
import faiss
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FAISSIndexManager:
    """Manager to create, save, load, and benchmark different FAISS index types."""
    
    def __init__(self):
        pass

    def build(self, vectors: np.ndarray, index_type: str) -> faiss.Index:
        """
        Build a FAISS index of the specified type.
        Supports: 'IndexFlatIP', 'IVFFlat', 'HNSWFlat'.
        Automatically normalizes vectors for Cosine Similarity.
        """
        if vectors.size == 0:
            raise ValueError("Cannot build index on empty vector array.")
            
        dim = vectors.shape[1]
        
        # L2 Normalization enables Inner Product to perform Cosine Similarity
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1
        vectors = (vectors / norms).astype(np.float32)
        
        start_time = time.time()
        logger.info(f"Building {index_type} index on {vectors.shape[0]} {dim}-d vectors...")
        
        if index_type == "IndexFlatIP":
            index = faiss.IndexFlatIP(dim)
            index.add(vectors)
            
        elif index_type == "IVFFlat":
            nlist = min(100, max(1, vectors.shape[0] // 10))
            if vectors.shape[0] < 100:
                logger.warning(f"Vector count {vectors.shape[0]} is small for IVFFlat. Adjusting nlist={nlist}.")
            
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
            
            logger.info("Training IVFFlat index...")
            if not index.is_trained:
                index.train(vectors)
            index.add(vectors)
            
        elif index_type == "HNSWFlat":
            M = 32
            index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efConstruction = 200
            index.add(vectors)
            
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
            
        elapsed = time.time() - start_time
        logger.info(f"Successfully built {index_type} in {elapsed:.4f}s")
        return index

    def save(self, index: faiss.Index, path: str):
        """Save an index to disk."""
        faiss.write_index(index, path)
        logger.info(f"Saved FAISS index to {path}")

    def load(self, path: str) -> faiss.Index:
        """Load an index from disk."""
        if not path:
            raise FileNotFoundError("Empty path provided for loading index.")
        logger.info(f"Loading FAISS index from {path}...")
        index = faiss.read_index(path)
        return index

    def search(self, index: faiss.Index, query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search the FAISS index to return top k (distances, indices)."""
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
            
        norms = np.linalg.norm(query_vector, axis=1, keepdims=True)
        norms[norms == 0] = 1
        query_vector = (query_vector / norms).astype(np.float32)
        
        distances, indices = index.search(query_vector, k)
        return distances, indices

    def get_index_stats(self, index: faiss.Index) -> dict:
        """Fetch statistics about the index."""
        try:
            dim = index.d
            mem_mb = (index.ntotal * dim * 4) / (1024 * 1024)
        except Exception:
            mem_mb = 0.0
            
        return {
            "ntotal": index.ntotal,
            "metric": "METRIC_INNER_PRODUCT" if getattr(index, "metric_type", -1) == faiss.METRIC_INNER_PRODUCT else "Other",
            "index_type": index.__class__.__name__,
            "memory_mb": mem_mb
        }

    def benchmark(self, vectors: np.ndarray, query_vectors: np.ndarray, k: int = 10) -> dict:
        """Benchmark index types measuring build time, query time, memory, and recall@k."""
        results = {}
        index_types = ["IndexFlatIP", "IVFFlat", "HNSWFlat"]
        
        # Establish exact ground truth
        exact_index = self.build(vectors, "IndexFlatIP")
        _, exact_indices = self.search(exact_index, query_vectors, k)
        
        for itype in index_types:
            try:
                start_t = time.time()
                index = self.build(vectors, itype)
                build_time = time.time() - start_t
                
                # Queries benchmark
                start_t = time.time()
                _, test_indices = self.search(index, query_vectors, k)
                query_time_ms = (time.time() - start_t) * 1000 / query_vectors.shape[0]
                
                # Recall@K calculation
                recalls = []
                for i in range(query_vectors.shape[0]):
                    correct = set(exact_indices[i])
                    retrieved = set(test_indices[i])
                    correct.discard(-1) # Clean blanks
                    if not correct:
                        continue
                    recall = len(correct.intersection(retrieved)) / len(correct)
                    recalls.append(recall)
                
                recall_at_k = np.mean(recalls) if recalls else 0.0
                stats = self.get_index_stats(index)
                
                results[itype] = {
                    "build_time_s": build_time,
                    "query_time_ms_per_item": query_time_ms,
                    "recall_at_k": recall_at_k,
                    "memory_mb": stats["memory_mb"]
                }
            except Exception as e:
                logger.error(f"Error benchmarking {itype}: {e}")
                results[itype] = {"error": str(e)}
                
        return results

if __name__ == "__main__":
    np.random.seed(42)
    dim = 768
    num_vectors = 10000
    num_queries = 1000
    
    print(f"Generating random {dim}-d vectors for {num_vectors} items and {num_queries} queries...")
    vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    query_vectors = np.random.rand(num_queries, dim).astype(np.float32)
    
    manager = FAISSIndexManager()
    stats = manager.benchmark(vectors, query_vectors, k=10)
    
    print("\n" + "=" * 90)
    print(f"{'index_type':<15} | {'build_time (s)':<15} | {'query_time (ms)':<15} | {'memory (MB)':<15} | {'recall@10':<15}")
    print("=" * 90)
    for itype in ["IndexFlatIP", "IVFFlat", "HNSWFlat"]:
        if itype in stats and "error" not in stats[itype]:
            s = stats[itype]
            print(f"{itype:<15} | {s['build_time_s']:<15.4f} | {s['query_time_ms_per_item']:<15.4f} | {s['memory_mb']:<15.2f} | {s['recall_at_k']:<15.4f}")
        else:
            print(f"{itype:<15} | ERROR: {stats.get(itype, {}).get('error', 'Unknown')}")
    print("=" * 90)
