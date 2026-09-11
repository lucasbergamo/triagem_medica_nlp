.PHONY: install lint format test validate data train clean

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

# ── Limpeza ───────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
