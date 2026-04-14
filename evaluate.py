# -*- coding: utf-8 -*-
import argparse
import io
import json
import os
import random
import sys
from datetime import datetime

import numpy as np

# Ensure stdout encodes to utf-8 (Moved to __main__)
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description="Evaluate RUC-Detect on HaluEval")
    parser.add_argument("--quick", action="store_true", help="Run only 50 samples for fast iteration")
    args = parser.parse_args()

    print("[INFO] Loading dataset pminervini/HaluEval (qa_samples)...")
    dataset = load_dataset("pminervini/HaluEval", "qa_samples", split="data")

    # The 'hallucination' field uses 'yes' or 'no'
    hallucinated = [item for item in dataset if item.get('hallucination', '').lower() == 'yes']
    non_hallucinated = [item for item in dataset if item.get('hallucination', '').lower() == 'no']

    # Stratified logic
    sample_size = 50 if args.quick else 200
    half_size = sample_size // 2

    random.seed(42)
    sample_h = random.sample(hallucinated, min(half_size, len(hallucinated)))
    sample_nh = random.sample(non_hallucinated, min(half_size, len(non_hallucinated)))

    combined_samples = sample_h + sample_nh
    random.shuffle(combined_samples)

    print(f"[INFO] Evaluating {len(combined_samples)} samples ({len(sample_h)} hallucinated, {len(sample_nh)} non-hallucinated).")

    y_true_binary = []  # 1 for hallucinated
    y_pred_binary = []  # 1 for hallucinated (at default threshold from pipeline)
    y_scores = []       # continuous score

    pattern_counts = {"extrinsic": 0, "intrinsic": 0, "semantic_drift": 0, "None": 0}
    scores_h = []
    scores_nh = []

    results_data = []

    for idx, sample in enumerate(combined_samples):
        prompt = sample["question"]
        response = sample["answer"]
        label_str = sample.get("hallucination", "no").lower()
        true_label = True if label_str == "yes" else False

        print(f"[{idx+1}/{len(combined_samples)}] True: {true_label} | Prompt: {prompt[:80]}...")

        try:
            result = run_pipeline(prompt, response, sampled_responses=None)

            p_score = result["score"]
            predicted_label = result["label"]
            pattern = result["pattern"]

            y_true_binary.append(1 if true_label else 0)
            y_pred_binary.append(1 if predicted_label else 0)
            y_scores.append(p_score)

            p_key = pattern if pattern in pattern_counts else "None"
            pattern_counts[p_key] += 1

            if true_label:
                scores_h.append(p_score)
            else:
                scores_nh.append(p_score)

            results_data.append({
                "question": prompt,
                "answer": response,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "score": p_score,
                "pattern": pattern,
                "explanation": result["explanation"]
            })

        except Exception as e:
            print(f"[ERROR] Failed sample {idx+1}: {e}")
            continue

    if not y_scores:
        print("[ERROR] No successful evaluations completed.")
        return

    # Basic Metrics
    acc = accuracy_score(y_true_binary, y_pred_binary)
    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true_binary, y_scores)
    except Exception:
        roc_auc = float('nan')

    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    avg_score_h = float(np.mean(scores_h)) if scores_h else 0.0
    avg_score_nh = float(np.mean(scores_nh)) if scores_nh else 0.0

    print("\n========================")
    print("EVALUATION RESULTS")
    print("========================")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}\n")

    print("Confusion Matrix:")
    print(f"TN: {cm[0][0]:<5} | FP: {cm[0][1]:<5}")
    print(f"FN: {cm[1][0]:<5} | TP: {cm[1][1]:<5}\n")

    print("Average Scores:")
    print(f"Hallucinated:     {avg_score_h:.4f}")
    print(f"Non-Hallucinated: {avg_score_nh:.4f}\n")

    print("Pattern Breakdown:")
    for pat, count in pattern_counts.items():
        print(f"  {pat}: {count}")

    # Threshold Sweep
    print("\n--- Threshold Sweep ---")
    thresholds = [0.40, 0.45, 0.50, 0.52, 0.55, 0.60]
    best_thresh = 0.52
    best_f1 = -1.0

    for t in thresholds:
        preds_t = [1 if s > t else 0 for s in y_scores]
        f1_t = f1_score(y_true_binary, preds_t, zero_division=0)
        print(f"Threshold {t:.2f} -> F1: {f1_t:.4f}")
        if f1_t > best_f1:
            best_f1 = f1_t
            best_thresh = t

    print(f"\n=> OPTIMAL THRESHOLD: {best_thresh:.2f} (F1: {best_f1:.4f})")

    # Save to JSON
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"results/halueval_eval_{timestamp}.json"

    output_payload = {
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "best_threshold": best_thresh,
            "best_f1": best_f1,
            "avg_score_hallucinated": avg_score_h,
            "avg_score_non_hallucinated": avg_score_nh,
            "patterns": pattern_counts
        },
        "samples": results_data
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=4)

    print(f"\n💾 Results saved to {out_file}")

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    main()
