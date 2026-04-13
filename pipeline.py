from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import pickle

from utils.nli import get_nli_score
from utils.selfcheck import selfcheck_nli
from utils.taxonomy import classify_pattern

model = SentenceTransformer("all-MiniLM-L6-v2")

index = faiss.read_index("models/faiss.index")

with open("models/docs.pkl", "rb") as f:
    documents = pickle.load(f)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def run_pipeline(prompt, response, sampled_responses=None):

    response_embedding = model.encode([response])
    response_embedding = response_embedding / np.linalg.norm(
        response_embedding, axis=1, keepdims=True
    )

    D, I = index.search(np.array(response_embedding), k=5)

    similarities = []
    top_docs = I[0]

    for idx in top_docs:
        doc_emb = index.reconstruct(int(idx))
        sim = cosine_similarity(response_embedding[0], doc_emb)
        similarities.append(sim)

    avg_sim = sum(similarities) / len(similarities)
    uncertainty = float(np.var(similarities))

    # ================= NLI (FactCC idea) =================
    nli_scores = []

    for sent in response.split("."):
        sent = sent.strip()
        if not sent:
            continue

        for idx in top_docs:
            label, score = get_nli_score(documents[idx], sent)

            if label == "ENTAILMENT":
                nli_scores.append(score)
            elif label == "CONTRADICTION":
                nli_scores.append(0)

    nli_score = sum(nli_scores)/len(nli_scores) if nli_scores else 0

    # ================= SelfCheckGPT =================
    selfcheck_score = 0.0

    if sampled_responses and len(sampled_responses) >= 3:
        _, sc_scores = selfcheck_nli(response, sampled_responses)
        selfcheck_score = float(np.mean(sc_scores))

    # ================= Final Score =================
    retrieval_score = (1 - avg_sim) + 0.2 * uncertainty

    final_score = min(
        1.0,
        0.4 * retrieval_score +
        0.3 * (1 - nli_score) +
        0.3 * selfcheck_score
    )

    label = final_score > 0.6

    # ================= Taxonomy =================
    pattern, explanation = classify_pattern(avg_sim, nli_score, final_score)

    # ================= Span Detection =================
    flagged_spans = []

    sentences = response.split(".")
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        emb = model.encode([sent])
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        D, I = index.search(np.array(emb), k=3)

        sims = []
        for idx in I[0]:
            doc_emb = index.reconstruct(int(idx))
            sims.append(cosine_similarity(emb[0], doc_emb))

        avg = sum(sims) / len(sims)

        if avg < 0.5:
            start = response.find(sent)
            flagged_spans.append({
                "start": start,
                "end": start + len(sent),
                "text": sent,
                "confidence": float(1 - avg)
            })

    return {
        "score": float(final_score),
        "label": bool(label),
        "explanation": explanation,
        "pattern": pattern,
        "spans": flagged_spans,
        "components": {
            "retrieval_similarity": float(avg_sim),
            "nli_score": float(nli_score),
            "selfcheck_score": float(selfcheck_score)
        }
    }