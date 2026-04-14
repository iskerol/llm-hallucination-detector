import pickle

import faiss
from sentence_transformers import SentenceTransformer

print("🔍 Loading index...")

# Load index
index = faiss.read_index("models/faiss.index")

# Load documents
with open("models/docs.pkl", "rb") as f:
    docs = pickle.load(f)

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Test query
query = "Who is the Prime Minister of India?"

print(f"\nQuery: {query}\n")

# Encode query
q_emb = model.encode([query]).astype("float32")

# Search
distances, indices = index.search(q_emb, k=3)

print("📄 Top Retrieved Documents:\n")

for i in indices[0]:
    print(docs[i][:300])
    print("-" * 60)
