.PHONY: install lint format test validate data train eval export-onnx promote serve \
	docker-serve docker-serve-onnx docker-serve-all benchmark benchmark-comparar clean

# ── Setup ──────────────────────────────────────────────────────────
install:
	poetry install

validate:
	poetry run python scripts/validate_env.py

# ── Qualidade de código ────────────────────────────────────────────
lint:
	poetry run ruff check src/ tests/ scripts/
	poetry run ruff format --check src/ tests/ scripts/

format:
	poetry run ruff check --fix src/ tests/ scripts/
	poetry run ruff format src/ tests/ scripts/

# ── Testes ────────────────────────────────────────────────────────
test:
	poetry run pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

# ── Dados ─────────────────────────────────────────────────────────
data:
	poetry run python -m src.data.pipeline

# ── Modelo ────────────────────────────────────────────────────────
train:
	poetry run python -m src.models.train

eval:
	poetry run python -m src.models.evaluate

export-onnx:
	poetry run python -m src.models.export_onnx

promote:
	poetry run python -m src.models.registry

# ── Serving ───────────────────────────────────────────────────────
serve:
	poetry run uvicorn src.api.main:app --reload --port 8000

docker-serve:
	docker compose up --build api-sklearn

docker-serve-onnx:
	docker compose up --build api-onnx

docker-serve-all:
	docker compose up --build api-sklearn api-onnx

benchmark:
	poetry run python scripts/benchmark_latency.py

benchmark-comparar:
	poetry run python scripts/benchmark_latency.py --comparar

# ── Limpeza ───────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
