.PHONY: install index eval eval-quick api ui lint docker-up docker-down

install:
	bash setup.sh

index:
	python build_index.py --sample 50000

eval:
	python evaluate.py

eval-quick:
	python evaluate.py --quick

api:
	uvicorn api:app --host 0.0.0.0 --port 8000 --reload

ui:
	python app.py

lint:
	ruff check .

docker-up:
	docker-compose up --build -d

docker-down:
	docker-compose down
