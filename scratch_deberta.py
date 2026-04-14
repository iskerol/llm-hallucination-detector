import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "cross-encoder/nli-deberta-v3-small"
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print("Config id2label:", model.config.id2label)
