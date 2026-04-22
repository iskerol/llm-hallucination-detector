import os
import hashlib
import time
import logging
import numpy as np
from typing import List
import torch
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceEmbedder:
    """
    Sentence embedder using sentence-transformers, with persistent caching and GPU support.
    """
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2", cache_dir: str = "./.cache/embeddings"):
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading SentenceTransformer model '{self.model_name}' on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info("Model loaded successfully.")
        
    def _get_cache_path(self, text: str) -> str:
        """Get the cache file path for a given text based on MD5 hash."""
        hash_str = hashlib.md5(text.encode('utf-8')).hexdigest()
        return os.path.join(self.cache_dir, f"{hash_str}.npy")

    def normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize an array of vectors to unit length using L2 norm."""
        if vectors.ndim == 1:
            norm = np.linalg.norm(vectors)
            if norm == 0:
                return vectors
            return vectors / norm
        else:
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            norms[norms == 0] = 1
            return vectors / norms

    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = True) -> np.ndarray:
        """
        Encode a list of texts into embeddings, utilizing an MD5-driven local cache.
        """
        if not texts:
            logger.warning("Empty list of texts provided to encode().")
            return np.array([])
            
        start_time = time.time()
        
        dim = self.model.get_sentence_embedding_dimension()
        result_embeddings = np.zeros((len(texts), dim), dtype=np.float32)
        
        to_encode = []
        to_encode_idx = []
        
        for i, text in enumerate(texts):
            if not text:
                continue
            cache_path = self._get_cache_path(text)
            if os.path.exists(cache_path):
                try:
                    result_embeddings[i] = np.load(cache_path)
                except Exception as e:
                    logger.warning(f"Corrupted cache file {cache_path}: {e}")
                    to_encode.append(text)
                    to_encode_idx.append(i)
            else:
                to_encode.append(text)
                to_encode_idx.append(i)
                
        if to_encode:
            logger.info(f"Computing embeddings for {len(to_encode)} missing entries...")
            computed = self.model.encode(to_encode, batch_size=batch_size, show_progress_bar=show_progress)
            
            for i, idx in enumerate(to_encode_idx):
                result_embeddings[idx] = computed[i]
                cache_path = self._get_cache_path(to_encode[i])
                np.save(cache_path, computed[i].astype(np.float32))
                
        elapsed = time.time() - start_time
        throughput = len(texts) / elapsed if elapsed > 0 else 0
        logger.info(f"Encoded {len(texts)} texts in {elapsed:.2f}s (Throughput: {throughput:.2f} items/sec)")
        
        return result_embeddings

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text into a 1D embedding array."""
        if not text:
            raise ValueError("Empty text provided")
        res = self.encode([text], show_progress=False)
        return res[0]
