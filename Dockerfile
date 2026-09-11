# Stage 1 — builder: instala dependências em ambiente isolado
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir poetry==2.4.1

COPY pyproject.toml poetry.lock* ./
RUN poetry config virtualenvs.create false \
    && poetry install --only main --no-root --no-interaction --no-ansi

# Stage 2 — runtime: imagem final enxuta sem ferramentas de build
FROM python:3.11-slim AS runtime

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# Cria diretórios de dados em tempo de build — resolve o problema de dirs faltantes
RUN mkdir -p data/bronze data/silver data/gold models metrics

COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Stage 3 — lint: só o ruff, isolado do builder — não precisa de sklearn/onnx/etc
# para checar sintaxe e estilo. Mais rápido e não depende do builder ter sucesso.
FROM python:3.11-slim AS lint

WORKDIR /app

RUN pip install --no-cache-dir "ruff>=0.7"

COPY pyproject.toml ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Stage 4 — ci: runtime + pytest, para rodar os testes reais em container
FROM python:3.11-slim AS ci

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

RUN pip install --no-cache-dir "pytest>=8.3" "pytest-cov>=5.0"

RUN mkdir -p data/bronze data/silver data/gold models metrics

COPY pyproject.toml ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY scripts/ ./scripts/
COPY configs/ ./configs/

ENV PYTHONPATH=/app
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Stage serve: runtime + modelo treinado embutido — a imagem que sobe sozinha em produção,
# sem depender de volume externo (ECS Fargate não tem disco persistente por padrão). Exige
# `make train && make eval && make promote` antes do build, para `models/current/` existir.
FROM runtime AS serve

COPY models/current/pipeline.joblib ./models/current/pipeline.joblib

EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
