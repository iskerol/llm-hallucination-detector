import os
import shutil
import numpy as np
import pytest
import faiss

from knowledge_base.embedder import SentenceEmbedder
from knowledge_base.faiss_index import FAISSIndexManager
from knowledge_base.builder import KnowledgeBaseBuilder

@pytest.fixture(scope="module")
def embedder():
    # Use a faster, lightweight model to prevent hanging during automated unit tests
    return SentenceEmbedder(model_name="sentence-transformers/all-MiniLM-L6-v2")

@pytest.fixture
def sample_texts():
    return ["This is a test document.", "Another sample text.", "A completely different context object with different concepts."]

def test_embedder(embedder, sample_texts):
    embeds = embedder.encode(sample_texts, show_progress=False)
    assert isinstance(embeds, np.ndarray)
    assert embeds.shape[0] == 3
    assert embeds.shape[1] > 0
    
    single = embedder.encode_single(sample_texts[0])
    assert single.shape == (embeds.shape[1],)
    
    # Test Normalization
    normed = embedder.normalize(embeds)
    norms = np.linalg.norm(normed, axis=1)
    assert np.allclose(norms, 1.0)

def test_faiss_index_manager():
    manager = FAISSIndexManager()
    dim = 384
    vectors = np.random.rand(100, dim).astype(np.float32)
    query = np.random.rand(1, dim).astype(np.float32)
    
    for idx_type in ["IndexFlatIP", "IVFFlat", "HNSWFlat"]:
        idx = manager.build(vectors, idx_type)
        assert isinstance(idx, faiss.Index)
        assert idx.ntotal == 100
        
        dist, indices = manager.search(idx, query, k=5)
        assert dist.shape == (1, 5)
        assert indices.shape == (1, 5)

def test_knowledge_base_builder(embedder):
    builder = KnowledgeBaseBuilder(embedder=embedder)
    docs = [
        {"id": "1", "title": "Doc1", "text": "This is the text for document one. It has multiple words."},
        {"id": "2", "title": "Doc2", "text": "Short document two."}
    ]
    
    chunks = builder.chunk_documents(docs, chunk_size=4, overlap=1)
    assert len(chunks) > 0
    assert chunks[0]["doc_id"] == "1"
    
    save_dir = "./test_kb_artifacts"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
        
    builder.build_from_scratch(docs, index_type="IndexFlatIP", save_dir=save_dir)
    assert os.path.exists(os.path.join(save_dir, "chunks.jsonl"))
    assert os.path.exists(os.path.join(save_dir, "embeddings.npy"))
    assert os.path.exists(os.path.join(save_dir, "faiss_index.bin"))
    
    # Reload workflow
    builder2 = KnowledgeBaseBuilder(embedder=embedder)
    loaded_chunks, mngr, idx = builder2.load(save_dir)
    assert len(loaded_chunks) == len(chunks)
    assert idx.ntotal == len(loaded_chunks)
    
    # Update workflow
    new_docs = [{"id": "3", "title": "Doc3", "text": "Dynamically added document later based upon real time event hooks."}]
    builder2.add_documents(new_docs)
    assert idx.ntotal > len(loaded_chunks)
    
    shutil.rmtree(save_dir)
