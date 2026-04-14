from transformers import pipeline
print("Loading text-classification...")
try:
    p1 = pipeline("text-classification", model="facebook/bart-large-mnli")
    print(p1([{"text": "A man is playing soccer.", "text_pair": "A man is playing a sport."}]))
except Exception as e:
    print(e)
