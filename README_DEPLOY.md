# Deployment & CI/CD Guide

## Local Development Setup
It's recommended to work within a conda environment.
```bash
conda create -n ruc-detect python=3.10
conda activate ruc-detect
pip install -r requirements.txt
```

## Docker Deployment
We use a multi-stage `Dockerfile` keeping deployments lean.
```bash
# Build the image locally
docker build -t ruc-detect-api .

# Or run the full stack (API + Gradio)
docker-compose up --build -d
```
The API bounds to `http://localhost:8000` and Gradio UI at `http://localhost:7860`.

## HuggingFace Spaces Deployment
This repository is configured natively for HuggingFace Spaces via Docker/Gradio.
1. Create a `New Space` on Hugging Face.
2. Select `Gradio` as the SDK.
3. Push these contents. Requirements are handled dynamically over CPU containers securely.

## API Endpoint Documentation

### Check Health Status
```bash
curl -X GET http://localhost:8000/health
```

### Detect Hallucinations
```bash
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Who is the CEO of Tesla?",
    "response": "The CEO of Tesla is Elon Musk.",
    "model_id": "llama-3-8b"
  }'
```

## Environment Variables
- `SPACE_API_URL`: Configures Gradio destination to the FastAPI backend (defaults to `http://localhost:8000`).
