# LLM Hallucination Detector — Project Summary

## 1. Purpose & Goal

**RUC-Detect** is a research-grade system designed to automatically detect and classify hallucinations in Large Language Model (LLM) outputs. Given a user query and an LLM-generated response, the system determines:

- **Whether** the response contains hallucinations (binary label)
- **How confident** it is (continuous score 0–1)
- **What type** of hallucination it is (intrinsic, extrinsic, or semantic drift)
- **Which specific sentences** are hallucinated (span-level detection)

---

## 2. Architecture Overview

The system uses a **multi-signal ensemble** approach that fuses three independent verification techniques:

```text
+-------------------+      +------------------+      +-------------------+
|   User Query      +------>   LLM Generator  +------>   Initial Answer  |
+-------------------+      +--------+---------+      +--------+----------+
                                    |                          |
                                    v                          v
                           +--------+---------+      +--------+----------+
                           | Semantic Entropy |      | Knowledge Base    |
                           |  (SelfCheckGPT)  |      | (FAISS Index)     |
                           +--------+---------+      +--------+----------+
                                    |                          |
                                    v                          v
+-------------------+      +--------+---------+      +--------+----------+
| Final Verdict     <------+  Ensemble Scorer <------+  Retriever & NLI  |
| & Span Detection  |      |                  |      |                   |
+-------------------+      +------------------+      +-------------------+
```

---

## 3. Detection Pipeline — Stage by Stage

### Stage 1: Knowledge Base Construction ([build_index.py](file:///c:/Users/Priyanka/Desktop/LLM_HAL_DETECTOR/build_index.py))
- Loads Wikipedia articles (default: 50K) via HuggingFace `datasets`
- Chunks documents with a sliding window (200 words, stride 100)
- Embeds chunks using `all-MiniLM-L6-v2` (SentenceTransformer)
- Builds and persists a **FAISS IVFFlat** index for fast similarity search

### Stage 2: Evidence Retrieval ([detection/retriever.py](file:///c:/Users/Priyanka/Desktop/LLM_HAL_DETECTOR/detection/retriever.py))
- Encodes the LLM response into an embedding
- Queries the FAISS index for top-K most similar knowledge base chunks
- Returns relevant passages as grounding evidence

### Stage 3: Retrieval Similarity Scoring ([detection/scorer.py](file:///c:/Users/Priyanka/Desktop/LLM_HAL_DETECTOR/detection/scorer.py))
- Computes **cosine similarity** between the response embedding and retrieved document embeddings
- Calculates average similarity and variance (uncertainty)
- Low similarity → higher hallucination risk

### Stage 4: NLI-Based Factual Verification ([utils/nli.py](file:///c:/Users/Priyanka/Desktop/LLM_HAL_DETECTOR/utils/nli.py))
- Uses **DeBERTa-v3** (`cross-encoder/nli-deberta-v3-small`) for Natural Language Inference
- Each claim (sentence) in the response is checked against each retrieved passage
- Produces ENTAILMENT / CONTRADICTION / NEUTRAL labels with confidence scores
- Supports **chunked premises** (sliding window over long documents) and **batched inference**
- Cached via `lru_cache` for performance

### Stage 5: SelfCheckGPT — Consistency Scoring ([utils/selfcheck.py](file:///c:/Users/Priyanka/Desktop/LLM_HAL_DETECTOR/utils/selfcheck.py))
- Implements the **SelfCheckGPT-NLI** method
- Compares the original response against 3+ independently sampled responses from the same LLM
- If the LLM contradicts itself across samples → likely hallucination
- Produces sentence-level contradiction probability scores

### Stage 6: Ensemble Scoring & Taxonomy ([detection/ensemble.py](file:///c:/Users/Priyanka/Desktop/LLM_HAL_DETECTOR/detection/ensemble.py), [utils/taxonomy.py](file:///c:/Users/Priyanka/Desktop/LLM_HAL_DETECTOR/utils/taxonomy.py))

**Final score formula:**
```
hallucination_score = 0.40 × (1 - retrieval_similarity)
                    + 0.35 × contradiction_probability
                    + 0.25 × normalized_semantic_entropy
```

**Hallucination taxonomy** classifies detections into:

| Type | Condition | Description |
|---|---|---|
| **Intrinsic** | High doc similarity + low NLI | Response contradicts retrieved evidence |
| **Extrinsic** | Low doc similarity + low NLI | Response contains unverifiable claims |
| **Semantic Drift** | Other cases | Response shifts context beyond source material |

### Stage 7: Span Detection ([detection/span_detector.py](file:///c:/Users/Priyanka/Desktop/LLM_HAL_DETECTOR/detection/span_detector.py))
- Extracts individual sentences using spaCy (`en_core_web_sm`)
- Independently retrieves evidence and scores each sentence
- Flags sentences with similarity < 0.5 as hallucinated spans
- Returns character-level start/end positions with confidence scores

---

## 4. Project Structure

```
LLM_HAL_DETECTOR/
├── config.py                 # Centralized settings (Pydantic)
├── pipeline.py               # Standalone end-to-end detection pipeline
├── build_index.py            # FAISS index builder from Wikipedia
├── evaluate.py               # Benchmark evaluation on HaluEval dataset
├── app.py                    # Gradio web UI
├── api.py                    # Legacy API entry point
│
├── api/                      # FastAPI REST API (production)
│   ├── main.py               #   Endpoints: /detect, /detect/batch, /health, /metrics
│   ├── models.py             #   Pydantic request/response schemas
│   └── middleware.py          #   CORS, rate limiting, logging middleware
│
├── detection/                # Core detection modules
│   ├── retriever.py          #   FAISS-based passage retrieval
│   ├── scorer.py             #   Similarity, NLI, and entropy scorers
│   ├── span_detector.py      #   Sentence-level hallucination detection
│   └── ensemble.py           #   Multi-signal fusion detector
│
├── knowledge_base/           # Knowledge base construction
│   ├── builder.py            #   End-to-end KB build/load pipeline
│   ├── embedder.py           #   Sentence embedding wrapper
│   └── faiss_index.py        #   FAISS index management (stats, types)
│
├── utils/                    # Utility modules
│   ├── nli.py                #   DeBERTa NLI inference (batched, cached)
│   ├── selfcheck.py          #   SelfCheckGPT-NLI implementation
│   └── taxonomy.py           #   Hallucination type classifier
│
├── evaluation/               # Evaluation framework
│   ├── benchmarks.py         #   Benchmark dataset loaders
│   ├── baselines.py          #   Baseline comparisons
│   └── metrics.py            #   Precision, recall, F1, ROC-AUC
│
├── tests/                    # Test suite
│   ├── integration_test.py   #   End-to-end integration tests
│   ├── test_detection.py     #   Detection module unit tests
│   └── test_knowledge_base.py#   KB module unit tests
│
├── scripts/
│   └── demo.py               # CLI demo script
│
├── models/                   # Persisted FAISS index & document store
├── data/                     # Datasets and preprocessing
├── experiments/              # Ablation studies
├── paper/                    # Research paper drafts
│
├── Dockerfile                # Container image definition
├── docker-compose.yml        # Multi-service orchestration
├── Makefile                  # Build/run shortcuts
├── requirements.txt          # Python dependencies
├── model_card.md             # HuggingFace-style model card
└── README.md                 # Project documentation
```

---

## 5. Technology Stack

| Category | Technology |
|---|---|
| **Language** | Python |
| **Embeddings** | SentenceTransformers (`all-MiniLM-L6-v2`, `all-mpnet-base-v2`) |
| **NLI Model** | DeBERTa-v3 (`cross-encoder/nli-deberta-v3-small`) |
| **Vector Search** | FAISS (IVFFlat, FlatIP, HNSW) |
| **NLP** | spaCy (`en_core_web_sm`) |
| **Deep Learning** | PyTorch, HuggingFace Transformers |
| **REST API** | FastAPI + Uvicorn |
| **Web UI** | Gradio |
| **Experiment Tracking** | MLflow |
| **Configuration** | Pydantic Settings + dotenv |
| **Evaluation Datasets** | HaluEval, TruthfulQA, TriviaQA, NQ-Open |
| **Deployment** | Docker, Docker Compose |
| **Testing** | pytest |

---

## 6. API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | System health, model status, index stats, uptime |
| `POST` | `/detect` | Single query/response hallucination detection |
| `POST` | `/detect/batch` | Batch detection with thread-pool parallelism |
| `POST` | `/index/build` | Trigger background FAISS index rebuild |
| `GET` | `/index/stats` | FAISS index memory and vector statistics |
| `GET` | `/metrics` | Aggregated request counts, latency, hallucination rate |

> All endpoints include **request caching** (LRU via MD5 hash), **CORS middleware**, and **global exception handling**.

---

## 7. Evaluation Framework

The [evaluate.py](file:///c:/Users/Priyanka/Desktop/LLM_HAL_DETECTOR/evaluate.py) script runs stratified evaluation on the **HaluEval** benchmark:

- Samples balanced hallucinated/non-hallucinated examples (50 quick / 200 full)
- Computes: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
- Performs **threshold sweep** (0.40–0.60) to find optimal decision boundary
- Outputs pattern distribution breakdown and per-sample results to JSON

**Reported Evaluation Results** (from model card):
| Benchmark | Score |
|---|---|
| TruthfulQA (MC2) | 79.5% |
| FactScore | 84.4 F1 |
| HaluEval (AUC) | 81.2% |
| Inference Latency | ~850ms (RTX 3090) |

---

## 8. Current Development Status

The project is in the **late-stage implementation and refinement** phase:

- ✅ Core detection pipeline fully implemented and functional
- ✅ FAISS knowledge base builder operational
- ✅ NLI scoring integrated with batched inference
- ✅ SelfCheckGPT consistency verification implemented
- ✅ Span-level hallucination detection working
- ✅ Hallucination taxonomy classifier (intrinsic/extrinsic/semantic drift)
- ✅ FastAPI REST API with batch support, caching, and metrics
- ✅ Gradio web UI for interactive testing
- ✅ Evaluation framework with HaluEval benchmark
- ✅ Docker deployment configuration
- ✅ MLflow experiment tracking integration
- ✅ Test suite (unit + integration)
