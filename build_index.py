import argparse
import os
import pickle

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


def chunk_text(text, size=200, stride=100):
    words = text.split()
    chunks = []

    if not words:
        return chunks

    if len(words) <= size:
        return [" ".join(words)]

    for i in range(0, len(words), stride):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
        if i + size >= len(words):
            break

    return chunks

def main():
    parser = argparse.ArgumentParser(description="Build FAISS Index for RUC-Detect")
    parser.add_argument("--sample", type=int, default=50000, help="Number of documents to sample")
    args = parser.parse_args()

    print("🚀 Starting FAISS index build...")
    os.makedirs("models", exist_ok=True)

    print(f"📥 Loading dataset (wikipedia 20220301.en), format: train[:{args.sample}]...")
    dataset = load_dataset("wikipedia", "20220301.en", split=f"train[:{args.sample}]")

    documents = []
    metadata = []

    print("✂️ Chunking documents...")
    for doc in dataset:
        title = doc.get("title", "Unknown")
        text = doc.get("text", "")
        chunks = chunk_text(text, size=200, stride=100)

        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadata.append({
                "title": title,
                "chunk_idx": idx
            })

    print(f"✅ Created {len(documents)} overlapping chunks from {len(dataset)} articles")

    print("🧠 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("⚡ Generating embeddings with batch_size=256...")
    embeddings = model.encode(
        documents,
        batch_size=256,
        show_progress_bar=True
    )
    embeddings = np.array(embeddings).astype("float32")

    print(f"✅ Embeddings shape: {embeddings.shape}")

    print("📦 Building FAISS IndexIVFFlat...")
    dimension = embeddings.shape[1]

    # Constructing quantizer and IVFFlat
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, 100)

    print("🎓 Training FAISS index...")
    index.train(embeddings)
    index.add(embeddings)

    print(f"✅ FAISS index built and populated with {index.ntotal} vectors")

    index_path = "models/faiss.index"
    faiss.write_index(index, index_path)
    print(f"💾 Saved FAISS index to {index_path}")

    docs_path = "models/docs.pkl"
    with open(docs_path, "wb") as f:
        pickle.dump(documents, f)
    print(f"💾 Saved {len(documents)} document chunks to {docs_path}")

    meta_path = "models/meta.pkl"
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"💾 Saved chunk metadata to {meta_path}")

    print("🎉 SUCCESS: Index building completed!")

if __name__ == "__main__":
    main()
