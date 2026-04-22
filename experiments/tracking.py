import mlflow
import pandas as pd
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_experiment(name: str = "hallucination-detector") -> mlflow.ActiveRun:
    """Configures MLflow tracking environment natively into specified host loops."""
    try:
        from config import config
        target_uri = getattr(config, "mlflow_tracking_uri", None)
        if target_uri:
            mlflow.set_tracking_uri(target_uri)
    except ImportError:
        pass
        
    mlflow.set_experiment(name)
    logger.info(f"MLflow experiment boundaries strictly initialized: {name}")
    return None

def log_detection(result: dict, dataset: str, split: str):
    """Maps single textual detection evaluations traces into MLFlow telemetry arrays natively."""
    try:
        with mlflow.start_run(nested=True):
            mlflow.log_param("query", str(result.get("query", ""))[:200])
            mlflow.log_param("response", str(result.get("response", ""))[:200])
            mlflow.log_param("dataset", dataset)
            mlflow.log_param("split", split)
            
            mlflow.log_metric("hallucination_score", result.get("hallucination_score", 0.0))
            mlflow.log_metric("is_hallucinated", float(result.get("is_hallucinated", False)))
            mlflow.log_metric("latency_ms", result.get("latency_ms", 0.0))
            
            signals = result.get("signals", {})
            if signals:
                mlflow.log_metrics({
                    "signal_retrieval": signals.get("retrieval_similarity", 0.0),
                    "signal_nli": signals.get("nli_entailment", 0.0),
                    "signal_entropy": signals.get("semantic_entropy", 0.0)
                })
    except Exception as e:
        logger.error(f"Failed artifact MLFLow log matrix bounds: {e}")

def log_benchmark_run(results: dict, run_name: str):
    """Pushes macro-metric AUROC validations cleanly wrapped under isolated run traces."""
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag("index_type", "IVFFlat")
        mlflow.set_tag("embed_model", "all-mpnet-base-v2")
        mlflow.set_tag("k", 10)
        
        for dataset_name, metrics in results.items():
            if "Labels" not in metrics:
                continue
            for method, data in metrics.items():
                if method == "Labels":
                    continue
                try:
                    from sklearn.metrics import roc_auc_score
                    auc = roc_auc_score(metrics["Labels"], data["scores"])
                    mlflow.log_metric(f"{dataset_name}_{method}_AUROC", auc)
                except Exception:
                    pass

def compare_runs(experiment_name: str) -> pd.DataFrame:
    """Scrapes local experiments outputting DataFrame arrays sorted dynamically."""
    current_exp = mlflow.get_experiment_by_name(experiment_name)
    if not current_exp:
         return pd.DataFrame()
         
    df = mlflow.search_runs(experiment_ids=[current_exp.experiment_id])
    if df.empty:
        return df
        
    auroc_cols = [c for c in df.columns if "AUROC" in c]
    if auroc_cols:
         df["mean_auroc"] = df[auroc_cols].mean(axis=1)
         df = df.sort_values(by="mean_auroc", ascending=False)
    return df

def get_best_config() -> Dict[str, Any]:
    """Sweeps internal dataframe outputs returning serialized dictionary of best weights."""
    df = compare_runs("hallucination-detector")
    if df.empty:
        return {}
        
    halueval_cols = [c for c in df.columns if "halueval" in c.lower() and "AUROC" in c]
    if not halueval_cols:
        return df.iloc[0].to_dict()
        
    col = halueval_cols[0]
    best_series = df.sort_values(by=col, ascending=False).iloc[0]
    return best_series.to_dict()

if __name__ == "__main__":
    setup_experiment("smoke-test-telemetry")
    log_detection({"query": "Q", "response": "R", "hallucination_score": 0.5}, "synthetic", "test")
    logger.info("Telemetry module bounded safely locally.")
