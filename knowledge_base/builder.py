import os
import json
import logging
import numpy as np
import faiss

from knowledge_base.embedder import SentenceEmbedder
from knowledge_base.faiss_index import FAISSIndexManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeBaseBuilder:
    """End-to-end Knowledge Base orchestration (Loading, Chunking, Building, and Updating)."""
    
    def __init__(self, embedder: SentenceEmbedder = None):
        self.embedder = embedder if embedder else SentenceEmbedder()
        self.index_manager = FAISSIndexManager()
        self.documents: list[dict] = []
        self.chunks: list[dict] = []
        self.index: faiss.Index = None
        self.save_dir: str = ""

    def load_wikipedia_passages(self, path_or_hf_dataset: str, max_docs: int = 100000) -> list[dict]:
        """Load text passages from HuggingFace OR a local JSONL file."""
        logger.info(f"Loading up to {max_docs} docs from {path_or_hf_dataset}")
        docs = []
        
        try:
            from datasets import load_dataset
            dataset = load_dataset(path_or_hf_dataset, split="train", streaming=True)
            for i, item in enumerate(dataset):
                if i >= max_docs:
                    break
                docs.append({
                    "id": str(item.get("id", i)),
                    "title": str(item.get("title", "")),
                    "text": str(item.get("text", ""))
                })
        except Exception as e:
            logger.info(f"Could not load via HuggingFace datasets API ({e}). Falling back to local file load.")
            try:
                with open(path_or_hf_dataset, "r", encoding="utf-8") as f:
                    for i, line in enumerate(f):
                        if i >= max_docs:
                            break
                        item = json.loads(line)
                        docs.append({
                            "id": str(item.get("id", i)),
                            "title": str(item.get("title", "")),
                            "text": str(item.get("text", ""))
                        })
            except Exception as file_e:
                logger.error(f"Failed to load local file: {file_e}")
        
        self.documents = docs
        logger.info(f"Successfully loaded {len(self.documents)} documents.")
        return self.documents

    def chunk_documents(self, docs: list[dict], chunk_size: int = 512, overlap: int = 64) -> list[dict]:
        """Segment lengthy texts into smaller contiguous units with given overlap parameter."""
        logger.info(f"Chunking {len(docs)} documents (chunk={chunk_size}, overlap={overlap})...")
        chunks = []
        chunk_id = 0
        
        for doc in docs:
            words = doc["text"].split()
            if not words:
                continue
                
            start = 0
            while start < len(words):
                end = min(start + chunk_size, len(words))
                chunk_text = " ".join(words[start:end])
                
                chunks.append({
                    "id": str(chunk_id),
                    "doc_id": doc["id"],
                    "text": chunk_text,
                    "start": start,
                    "end": end
                })
                chunk_id += 1
                
                if end == len(words):
                    break
                start += (chunk_size - overlap)
                
        logger.info(f"Produced {len(chunks)} overlapping chunks.")
        return chunks

    def build_from_scratch(self, docs: list[dict], index_type: str = "IVFFlat", save_dir: str = "./kb/") -> None:
        """Embed and format all content into artifacts stored onto disk."""
        if not docs:
            raise ValueError("No documents provided to build the knowledge base.")
            
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        
        self.chunks = self.chunk_documents(docs)
        texts = [c["text"] for c in self.chunks]
        
        logger.info("Encoding chunks into representations...")
        embeddings = self.embedder.encode(texts)
        if embeddings.size == 0:
            raise RuntimeError("Embedder returned an empty array.")
            
        self.index = self.index_manager.build(embeddings, index_type)
        
        # Save structured outputs
        with open(os.path.join(save_dir, "chunks.jsonl"), "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c) + "\n")
                
        np.save(os.path.join(save_dir, "embeddings.npy"), embeddings)
        self.index_manager.save(self.index, os.path.join(save_dir, "faiss_index.bin"))
        
        metadata = {
            "num_docs": len(docs),
            "num_chunks": len(self.chunks),
            "index_type": index_type,
            "dimension": embeddings.shape[1]
        }
        with open(os.path.join(save_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Knowledge base successfully staged and built at {save_dir}")

    def load(self, save_dir: str) -> tuple[list[dict], FAISSIndexManager, faiss.Index]:
        """Bootstrap Knowledge Base states from precomputations on disk."""
        if not os.path.exists(save_dir):
            raise FileNotFoundError(f"State directory not found at {save_dir}")
            
        self.save_dir = save_dir
        self.chunks = []
        
        chunks_path = os.path.join(save_dir, "chunks.jsonl")
        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.chunks.append(json.loads(line))
        else:
            logger.warning(f"chunks.jsonl not found at {save_dir}")

        index_path = os.path.join(save_dir, "faiss_index.bin")
        if os.path.exists(index_path):
            self.index = self.index_manager.load(index_path)
        else:
            raise FileNotFoundError(f"Missing core requirements: faiss_index.bin not found at {save_dir}")
            
        logger.info(f"Restored Knowledge base with {len(self.chunks)} items.")
        return self.chunks, self.index_manager, self.index

    def add_documents(self, new_docs: list[dict]) -> None:
        """Extend the existing knowledge base dynamically."""
        if self.index is None:
            raise ValueError("State missing! Cannot update an uninitialized index.")
        if not new_docs:
            return
            
        new_chunks = self.chunk_documents(new_docs)
        if not new_chunks:
            return
            
        embeddings = self.embedder.encode([c["text"] for c in new_chunks])
        
        # FAISS normalization is inherently performed in `build`, must be forced here!
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized_embeddings = (embeddings / norms).astype(np.float32)
        
        self.index.add(normalized_embeddings)
        self.chunks.extend(new_chunks)
        
        # Mirror mutations logically to memory/disk configurations 
        if self.save_dir:
            with open(os.path.join(self.save_dir, "chunks.jsonl"), "a", encoding="utf-8") as f:
                for chunk in new_chunks:
                    f.write(json.dumps(chunk) + "\n")
                    
            emb_path = os.path.join(self.save_dir, "embeddings.npy")
            if os.path.exists(emb_path):
                old = np.load(emb_path)
                combined = np.vstack([old, embeddings])
                np.save(emb_path, combined)
            else:
                np.save(emb_path, embeddings)
                
            self.index_manager.save(self.index, os.path.join(self.save_dir, "faiss_index.bin"))
            
            meta_path = os.path.join(self.save_dir, "metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["num_chunks"] = len(self.chunks)
                meta["num_docs"] = meta.get("num_docs", 0) + len(new_docs)
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, indent=4)
                    
        logger.info(f"Systematic extensions applied. Size grown by {len(new_chunks)} blocks.")
