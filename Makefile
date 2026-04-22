.PHONY: install test lint run-api run-ui build-index evaluate paper-figures

install:
	pip install -r requirements.txt
	pip install -e .

test:
	pytest tests/

lint:
	black .
	ruff check .

run-api:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	gradio ui/app.py

build-index:
	python knowledge_base/builder.py

evaluate:
	python evaluation/benchmarks.py

paper-figures:
	python experiments/ablation.py
