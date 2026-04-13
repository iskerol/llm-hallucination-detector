import numpy as np
from transformers import pipeline

nli_model = pipeline(
    "text-classification",
    model="cross-encoder/nli-deberta-v3-large"
)

def selfcheck_nli(main_response, sampled_responses):
    sentences = [s.strip() for s in main_response.split(".") if s.strip()]
    scores = []

    for sent in sentences:
        contradiction_scores = []

        for sample in sampled_responses:
            result = nli_model(f"{sample} </s> {sent}")[0]

            if result["label"] == "CONTRADICTION":
                contradiction_scores.append(result["score"])
            else:
                contradiction_scores.append(0.0)

        scores.append(float(np.mean(contradiction_scores)))

    return sentences, scores
