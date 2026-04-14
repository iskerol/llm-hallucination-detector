from functools import lru_cache

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_name = "cross-encoder/nli-deberta-v3-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model.eval()

LABEL_MAP = {
    0: "CONTRADICTION",
    1: "NEUTRAL",
    2: "ENTAILMENT"
}

def _chunk_premise(premise, max_tokens=400):
    tokens = tokenizer(premise, add_special_tokens=False)['input_ids']
    if len(tokens) <= max_tokens:
        return [premise]

    windows = []
    for i in range(0, len(tokens), max_tokens):
        chunk_tokens = tokens[i:i+max_tokens]
        windows.append(tokenizer.decode(chunk_tokens))
    return windows

@lru_cache(maxsize=1000)
def _cached_nli(key_hash, premise, hypothesis):
    windows = _chunk_premise(premise, max_tokens=400)

    best_window_label = "NEUTRAL"
    best_window_score = 0.0
    max_entail_score = -1.0

    for w in windows:
        inputs = tokenizer(w, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0]

        score, label_idx = torch.max(probs, dim=-1)
        entail_prob = probs[2].item()

        if entail_prob > max_entail_score:
            max_entail_score = entail_prob
            best_window_label = LABEL_MAP.get(label_idx.item(), "NEUTRAL")
            best_window_score = score.item()

    return best_window_label, best_window_score

def get_nli_score(premise, hypothesis):
    h = hash(premise[:100] + hypothesis)
    return _cached_nli(h, premise, hypothesis)

def batch_nli_scores(pairs, batch_size=16):
    if not pairs:
        return []

    flattened_inputs = []
    pair_to_window_indices = []

    current_idx = 0
    for p, h in pairs:
        windows = _chunk_premise(p, max_tokens=400)
        start_idx = current_idx
        for w in windows:
            flattened_inputs.append((w, h))
            current_idx += 1
        pair_to_window_indices.append((start_idx, current_idx))

    window_results = []
    for i in range(0, len(flattened_inputs), batch_size):
        batch = flattened_inputs[i:i+batch_size]
        premises = [b[0] for b in batch]
        hypotheses = [b[1] for b in batch]

        inputs = tokenizer(premises, hypotheses, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

        scores, label_idxs = torch.max(probs, dim=-1)

        for j in range(len(batch)):
            label_idx = label_idxs[j].item()
            entail_prob = probs[j][2].item()
            window_results.append({
                "label": LABEL_MAP.get(label_idx, "NEUTRAL"),
                "score": scores[j].item(),
                "entail_prob": entail_prob
            })

    final_scores = []
    for start_idx, end_idx in pair_to_window_indices:
        w_res = window_results[start_idx:end_idx]
        best = max(w_res, key=lambda x: x["entail_prob"])
        final_scores.append((best["label"], best["score"]))

    return final_scores
