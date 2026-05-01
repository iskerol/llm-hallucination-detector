import os
import pickle

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# -----------------------------
# Helper: Chunk text
# -----------------------------
def chunk_text(text, size=100, stride=50):
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


# -----------------------------
# Main
# -----------------------------
def main():
    print("🚀 Starting FAISS index build...")

    os.makedirs("models", exist_ok=True)

    # ✅ LOCAL DATASET (No internet needed)
    dataset = [
        {
            "title": "Gravity",
            "text": "Gravity was discovered by Isaac Newton. It is a force that attracts objects toward each other."
        },
        {
            "title": "Python Programming",
            "text": "Python is a programming language created by Guido van Rossum. It is widely used in artificial intelligence and data science."
        },
        {
            "title": "Earth",
            "text": "Earth is the third planet from the Sun. It supports life and has water and atmosphere."
        },
        {
            "title": "Artificial Intelligence",
            "text": "Artificial Intelligence is the simulation of human intelligence in machines that are programmed to think and learn."
        },
        {
            "title": "Machine Learning",
            "text": "Machine learning is a subset of AI that allows systems to learn and improve from experience without being explicitly programmed."
        }
    ]

    documents = []
    metadata = []

    print("✂️ Chunking documents...")

    for doc in dataset:
        title = doc["title"]
        text = doc["text"]

        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadata.append({
                "title": title,
                "chunk_idx": idx
            })

    print(f"✅ Created {len(documents)} chunks")

    # -----------------------------
    # Embedding
    # -----------------------------
    print("🧠 Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("⚡ Generating embeddings...")
    embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True
    )

    embeddings = np.array(embeddings).astype("float32")
    print(f"✅ Embedding shape: {embeddings.shape}")

    # -----------------------------
    # FAISS Index
    # -----------------------------
    print("📦 Building FAISS index...")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    print(f"✅ Index contains {index.ntotal} vectors")

    # -----------------------------
    # Save files
    # -----------------------------
    faiss.write_index(index, "models/faiss.index")

    with open("models/docs.pkl", "wb") as f:
        pickle.dump(documents, f)

    with open("models/meta.pkl", "wb") as f:
        pickle.dump(metadata, f)

    print("💾 Saved all files in /models")
    print("🎉 SUCCESS: Index built successfully!")


if __name__ == "__main__":
    main()