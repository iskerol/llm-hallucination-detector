from transformers import pipeline

nli_model = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli"
)

def get_nli_score(premise, hypothesis):
    result = nli_model(f"{premise} </s> {hypothesis}")[0]
    return result["label"], result["score"]