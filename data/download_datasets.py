import os
import json
import logging
from datasets import load_dataset
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_and_save(name, iterator, limit=None):
    os.makedirs("data", exist_ok=True)
    out_path = f"data/{name}_eval.jsonl"
    
    count = 0
    positives = 0
    total_len = 0
    
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, item in enumerate(iterator):
            if limit and idx >= limit:
                break
            f.write(json.dumps(item) + "\n")
            count += 1
            positives += item["label"]
            response_len = len(item.get("response", "").split())
            total_len += response_len
            
    pos_rate = positives / count if count > 0 else 0
    avg_len = total_len / count if count > 0 else 0
    
    logger.info(f"Dataset block: {name}")
    logger.info(f"  Size: {count}")
    logger.info(f"  Positive Rate (Hallucination 1 Bounds): {pos_rate:.2%}")
    logger.info(f"  Avg Response Length (Words): {avg_len:.1f}\n")

def main():
    np.random.seed(42)
    
    logger.info("Downloading and processing HaluEval via streaming boundaries...")
    ds_halu = load_dataset("PatronusAI/HaluEval", split="qa", streaming=True)
    def it_halu():
        for i, row in enumerate(ds_halu):
            lbl = 1 if row.get("hallucination", "no").lower() == "yes" else 0
            ans = row.get("hallucinated_answer") if lbl == 1 else row.get("right_answer", "")
            if not ans: ans = row.get("answer", "")
            yield {"id": f"halueval_{i}", "query": row["question"], "response": ans, "label": lbl}
    process_and_save("halueval", it_halu(), 2000)

    logger.info("Downloading and processing TriviaQA subsets...")
    ds_trivia = load_dataset("trivia_qa", "rc", split="train", streaming=True)
    def it_trivia():
        for i, row in enumerate(ds_trivia):
            q = row["question"]
            ans = row["answer"]["value"]
            lbl = 0
            if np.random.rand() < 0.2:
                ans = "An incorrect completely hallucinated generated response mapping completely fabricated entities."
                lbl = 1
            yield {"id": f"trivia_{i}", "query": q, "response": ans, "label": lbl}
    process_and_save("triviaqa", it_trivia(), 2000)

    logger.info("Downloading and processing TruthfulQA multiple choice nodes...")
    ds_truth = load_dataset("truthful_qa", "multiple_choice", split="validation", streaming=True)
    def it_truth():
        item_uid = 0
        for row in ds_truth:
            q = row["question"]
            targets = row["mc1_targets"]
            choices = targets["choices"]
            labels = targets["labels"]
            for c, l in zip(choices, labels):
                # Using 0=False (Hallucinated) / 1=True (Grounded) mapping inversely to 1=Hallucination
                label = 1 if l == 0 else 0
                yield {"id": f"truth_{item_uid}", "query": q, "response": c, "label": label}
                item_uid += 1
    process_and_save("truthfulqa", it_truth(), 2000)

    logger.info("Downloading and processing Natural Questions pipeline array bounds...")
    ds_nq = load_dataset("nq_open", split="train", streaming=True)
    def it_nq():
        for i, row in enumerate(ds_nq):
            q = row["question"]
            ans = row["answer"][0] if row.get("answer") else ""
            lbl = 0
            if np.random.rand() < 0.2:
                ans = "A completely arbitrary hallucinated fabrication failing grounding logic constraints."
                lbl = 1
            yield {"id": f"nq_{i}", "query": q, "response": ans, "label": lbl}
    process_and_save("nq_open", it_nq(), 2000)

if __name__ == "__main__":
    logger.info("Preparing artifacts via local downloading bounds pipelines...")
    main()
