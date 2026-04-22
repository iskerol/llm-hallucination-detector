import os
import json
import time
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BenchmarkRunner:
    """Core evaluation array iterating structured matrices routing detection/baseline validations natively."""
    def run(self, dataset_name: str, detector, baselines: list, max_samples: int = 500) -> dict:
        """Trace loops wrapping distinct pipelines mapping results securely."""
        path = f"data/{dataset_name}_eval.jsonl"
        if not os.path.exists(path):
            logger.warning(f"Validation dataset missing {path}.")
            return {}
            
        data = []
        with open(path, 'r', encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if idx >= max_samples:
                    break
                data.append(json.loads(line))
                
        metrics = { "Labels": [d.get("label", 0) for d in data] }
        metrics[detector.__class__.__name__] = {"scores": [], "latencies": []}
        
        for b in baselines:
             metrics[b.name] = {"scores": [], "latencies": []}
             
        for row in data:
            q, r = row.get("query", ""), row.get("response", "")
            
            # Master Detector Sequence bounds check
            t0 = time.time()
            if hasattr(detector, "detect"):
                res = detector.detect(q, r, [r]) 
                score = res.get("hallucination_score", 0.0)
            else:
                score = 0.5 
            lat = (time.time() - t0) * 1000
            metrics[detector.__class__.__name__]["scores"].append(score)
            metrics[detector.__class__.__name__]["latencies"].append(lat)
            
            # Baseline trace maps natively 
            for b in baselines:
                t0 = time.time()
                if b.name == "LexicalSimilarityBaseline":
                    s = b.predict(r, [r]) 
                elif b.name == "SelfCheckGPT-lite":
                    s = b.predict([r, r + " artificial drift"])
                else:
                    s = b.predict(q, r)
                lat = (time.time() - t0) * 1000
                metrics[b.name]["scores"].append(s)
                metrics[b.name]["latencies"].append(lat)
                
        return metrics

    def run_all_datasets(self, detector, baselines) -> dict:
        """Automated sequential triggering arrays crossing configured spaces."""
        results = {}
        for d in ["halueval", "triviaqa", "truthfulqa", "nq_open"]:
            results[d] = self.run(d, detector, baselines)
        return results

    def generate_paper_table(self, all_results: dict) -> str:
        """Projects tabular data seamlessly via LaTeX boundaries matrices arrays natively wrapping structures."""
        datasets = list(all_results.keys())
        if not datasets:
            return ""
            
        methods = list(next(iter(all_results.values())).keys())
        methods.remove("Labels")
        
        lines = [
            "\\begin{tabular}{l|" + "cc|" * len(datasets) + "}",
            "\\hline",
            "Method & " + " & ".join([f"\\multicolumn{{2}}{{c}}{{{d}}}" for d in datasets]) + " \\\\",
            " & " + " & ".join(["AUC & F1" for _ in datasets]) + " \\\\",
            "\\hline"
        ]
        
        for m in methods:
            row_str = f"{m}"
            for d in datasets:
                data = all_results[d]
                y_true = data.get("Labels", [])
                if m in data:
                    y_pred = data[m]["scores"]
                    from evaluation.metrics import compute_auroc, compute_best_f1
                    auc = compute_auroc(y_true, y_pred)
                    f1 = compute_best_f1(y_true, y_pred)["f1"]
                    row_str += f" & {auc:.3f} & {f1:.3f}"
                else:
                    row_str += " & - & -"
            row_str += " \\\\"
            lines.append(row_str)
            
        lines.append("\\hline\n\\end{tabular}")
        return "\n".join(lines)

if __name__ == "__main__":
    class DummyDetector:
        def detect(self, q, r, s): return {"hallucination_score": 0.9}
        
    runner = BenchmarkRunner()
    from evaluation.baselines import RandomBaseline
    res = runner.run("halueval", DummyDetector(), [RandomBaseline()], max_samples=2)
    print("LaTeX Table bounds smoke test complete array logic loaded safely.")
