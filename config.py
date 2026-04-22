import os
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Environment Variables
    hf_token: str = ""
    openai_api_key: str = ""
    knowledge_base_path: str = "./data/knowledge_base"
    faiss_index_path: str = "./data/index"
    mlflow_tracking_uri: str = "./mlruns"
    log_level: str = "INFO"
    max_workers: int = 4

    # Hard-coded Constants
    EMBED_MODEL: str = "sentence-transformers/all-mpnet-base-v2"
    NLI_MODEL: str = "cross-encoder/nli-deberta-v3-base"
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    TOP_K_DEFAULT: int = 10
    SIMILARITY_THRESHOLD: float = 0.75
    HALLUCINATION_THRESHOLD: float = 0.5
    INDEX_TYPES: List[str] = ["FlatIP", "IVFFlat", "HNSW"]
    N_SAMPLES: int = 5
    DATASETS: List[str] = ["halueval", "triviaqa", "truthfulqa", "nq_open"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

config = Settings()
