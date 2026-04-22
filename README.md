<!--
llm-hallucination-detector/
├── config.py
├── data/
│   ├── download_datasets.py
│   └── preprocessing.py
├── knowledge_base/
│   ├── builder.py
│   ├── faiss_index.py
│   └── embedder.py
├── detection/
│   ├── retriever.py
│   ├── scorer.py
│   ├── span_detector.py
│   └── ensemble.py
├── api/
│   ├── main.py
│   ├── models.py
│   └── middleware.py
├── ui/
│   └── app.py
├── evaluation/
│   ├── benchmarks.py
│   ├── baselines.py
│   └── metrics.py
├── experiments/
│   └── ablation.py
└── tests/
-->

# LLM Hallucination Detector

## Project Overview

LLM Hallucination Detector is a research-grade system designed to systematically identify and categorize hallucinations in Large Language Model outputs. By leveraging advanced retrieval-augmented techniques, NLI verification, and semantic entropy scoring, the pipeline precisely extracts specific textual claims and compares them against a verified trusted knowledge base, producing robust hallucination detection metrics.

## Architecture

```text
+-------------------+      +------------------+      +-------------------+
|                   |      |                  |      |                   |
|   User Query      +------>   LLM Generator  +------>   Initial Answer  |
|                   |      |                  |      |                   |
+-------------------+      +--------+---------+      +--------+----------+
                                    |                         |
                                    v                         v
                           +--------+---------+      +--------+----------+
                           |                  |      |                   |
                           | Semantic Entropy |      | Knowledge Base    |
                           |  Generation (5x) |      | (FAISS Index)     |
                           |                  |      |                   |
                           +--------+---------+      +--------+----------+
                                    |                         |
                                    v                         v
+-------------------+      +--------+---------+      +--------+----------+
|                   |      |                  |      |                   |
| Final Verdict     <------+  Ensemble Scorer <------+  Retriever & NLI  |
| & Span Detection  |      |                  |      |                   |
+-------------------+      +------------------+      +-------------------+
```

## Quick Start

1. Clone the repository and navigate into it.
2. Install the necessary dependencies and setup the package:
   ```bash
   make install
   ```
3. Set your environment variables in `.env` (copied from `.env.example`).
4. Build the knowledge base index:
   ```bash
   make build-index
   ```
5. Run the Rest API server:
   ```bash
   make run-api
   ```
6. Open the Gradio UI:
   ```bash
   make run-ui
   ```

## Citation

If you use this system in your research, please cite:

```bibtex
@software{llm_hallucination_detector_2024,
  title = {LLM Hallucination Detector},
  author = {Your Name},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/llm-hallucination-detector}}
}
```
