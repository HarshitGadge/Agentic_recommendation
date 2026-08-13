.PHONY: help install install-baseline dev test lint run ingest bench docker-build docker-up docker-down clean

PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin
CORPUS ?= ./data/docs

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

$(VENV):
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip

install: $(VENV) ## Install runtime + dev dependencies
	$(BIN)/pip install -e ".[dev]"

install-baseline: $(VENV) ## Also install torch/sentence-transformers (benchmark baseline only)
	$(BIN)/pip install -e ".[dev,baseline]"

test: ## Run the test suite
	$(BIN)/pytest

lint: ## Lint with ruff
	$(BIN)/ruff check app bench scripts tests

run: ## Start the API on :8000 with reload
	$(BIN)/uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

ingest: ## Ingest $(CORPUS) into the vector store
	$(BIN)/python -m scripts.ingest $(CORPUS)

bench: ## Run the latency benchmark against $(CORPUS)
	$(BIN)/python -m bench.benchmark --corpus $(CORPUS)

docker-build: ## Build the container image
	docker compose build

docker-up: ## Start the service in Docker
	docker compose up -d
	@echo "http://localhost:8000/docs"

docker-down: ## Stop the service
	docker compose down

clean: ## Remove venv, caches, and the local vector store
	rm -rf $(VENV) .pytest_cache .ruff_cache **/__pycache__ *.egg-info data/chroma
