from transformers import pipeline
print("Loading zero-shot-classification...")
try:
    p2 = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
    print(p2(sequences="A man is playing soccer.", candidate_labels=["A man is playing a sport.", "A man is running.", "A man is eating."]))
    print(p2("A man is playing soccer.", candidate_labels=["entailment", "contradiction", "neutral"]))
except Exception as e:
    print(e)
