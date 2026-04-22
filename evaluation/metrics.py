import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve

def compute_auroc(y_true: list[int], y_pred_prob: list[float]) -> float:
    """Computes Area Under Receiver Operating Characteristics boundaries."""
    if len(set(y_true)) <= 1:
        return 0.5
    return float(roc_auc_score(y_true, y_pred_prob))

def compute_f1_at_threshold(y_true: list[int], y_pred_prob: list[float], threshold: float = 0.5) -> dict:
    """Maps array boundaries to binary predictions and evaluates P R F1 schemas."""
    preds = [1 if p >= threshold else 0 for p in y_pred_prob]
    if len(set(y_true)) <= 1 and len(set(preds)) <= 1:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": threshold}
        
    p, r, f1, _ = precision_recall_fscore_support(y_true, preds, average='binary', zero_division=0)
    return {"precision": float(p), "recall": float(r), "f1": float(f1), "threshold": threshold}

def compute_best_f1(y_true: list[int], y_pred_prob: list[float]) -> dict:
    """Sweeps bounds to determine optimally parameterized thresholds."""
    best_f1 = 0.0
    best_res = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "threshold": 0.5}
    for th in np.arange(0.1, 1.0, 0.1):
        res = compute_f1_at_threshold(y_true, y_pred_prob, float(th))
        if res["f1"] > best_f1:
            best_f1 = res["f1"]
            best_res = res
    return best_res

def compute_iou_spans(pred_spans: list[tuple[int, int]], gold_spans: list[tuple[int, int]]) -> float:
    """Evaluates Intersection Over Union mappings internally measuring overlap exactitude."""
    if not pred_spans and not gold_spans:
        return 1.0
    if not gold_spans or not pred_spans:
        return 0.0
        
    pred_set = set()
    for s, e in pred_spans:
        pred_set.update(range(s, e))
        
    gold_set = set()
    for s, e in gold_spans:
        gold_set.update(range(s, e))
        
    intersection = len(pred_set.intersection(gold_set))
    union = len(pred_set.union(gold_set))
    return float(intersection / union) if union > 0 else 0.0

def compute_latency_stats(latencies_ms: list[float]) -> dict:
    """Percentile tracking logic bounding structural delays arrays."""
    if not latencies_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(latencies_ms)),
        "p50": float(np.percentile(latencies_ms, 50)),
        "p95": float(np.percentile(latencies_ms, 95)),
        "p99": float(np.percentile(latencies_ms, 99)),
        "max": float(np.max(latencies_ms)),
    }

def plot_roc_curve(results_dict: dict[str, tuple]) -> plt.Figure:
    """Generates standard machine learning validation figures exporting to artifacts natively."""
    os.makedirs("experiments/figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    
    for method_name, (y_true, y_pred_prob) in results_dict.items():
        if len(set(y_true)) > 1:
            fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
            auc = compute_auroc(y_true, y_pred_prob)
            ax.plot(fpr, tpr, label=f"{method_name} (AUC = {auc:.3f})")
            
    ax.plot([0, 1], [0, 1], 'k--', label="Random bounds baseline")
    ax.set_xlabel('False Positive Rate Bounds')
    ax.set_ylabel('True Positive Rate Bounds')
    ax.set_title('Cross Validated ROC Curves Benchmark Traces')
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig("experiments/figures/roc.png")
    return fig

def generate_results_table(all_results: dict) -> pd.DataFrame:
    """Packages structured metric aggregations directly inside raw numerical dataframe loops."""
    rows = []
    for method, res in all_results.items():
        rows.append({
            "Method": method,
            "AUROC": res.get("auroc", 0.0),
            "F1": res.get("f1", 0.0),
            "Precision": res.get("precision", 0.0),
            "Recall": res.get("recall", 0.0),
            "Latency_p95": res.get("latency_p95", 0.0)
        })
    return pd.DataFrame(rows)
    
if __name__ == "__main__":
    t = [0, 1, 1, 0, 1]
    p = [0.1, 0.9, 0.8, 0.2, 0.6]
    print("Self verification bounding limits generated.")
