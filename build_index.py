from sentence_transformers import SentenceTransformer
import faiss, numpy as np, pickle
from datasets import load_dataset
import os

os.makedirs("models", exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

dataset = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:1000]")
docs = [x["text"] for x in dataset if x.get("text")]

emb = model.encode(docs, show_progress_bar=True)
emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

index = faiss.IndexFlatL2(emb.shape[1])
index.add(np.array(emb))

faiss.write_index(index, "models/faiss.index")

with open("models/docs.pkl", "wb") as f:
    pickle.dump(docs, f)

print("✅ Index built successfully")