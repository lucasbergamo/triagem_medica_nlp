.PHONY: install lint format test validate data train eval export-onnx promote serve \
	docker-serve docker-serve-onnx docker-serve-all docker-pipeline docker-lint docker-test \
	docker-load-test \
	benchmark benchmark-comparar monitoring-up monitoring-down load-test airflow-up \
	airflow-down airflow-reset dag-test clean

# Uid/gid do host: o Compose dá precedência a variável de ambiente do shell sobre o valor de
# --env-file ou de default no próprio compose.yml, então isso vale para qualquer pessoa que
# rode `make`, sem depender de editar nada à mão.
# - AIRFLOW_UID: os artefatos que a DAG escreve em data/, models/ e metrics/ (montados por
#   volume) saem com o dono certo, não com o uid 50000 do container.
# - CURRENT_UID/CURRENT_GID: os serviços `lint`/`ci` do compose (perfil `ci`) rodam com esse
#   uid/gid — sem isso, os arquivos que o `ci` escreve em data/silver, data/gold, models/ e
#   metrics/ saem donos de root, e o Airflow (rodando com o uid do host) não consegue
#   sobrescrevê-los depois.
export AIRFLOW_UID := $(shell id -u)
export CURRENT_UID := $(shell id -u)
export CURRENT_GID := $(shell id -g)

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

# Alvos em container, espelhando install/lint/test/data/train/eval/export-onnx/promote —
# para quem quer executar e avaliar o projeto sem Python nem Poetry no host (Caminho 1 do
# README). Usam os serviços `lint`/`ci` do perfil `ci`, que já existem no compose.
docker-pipeline:
	docker compose --profile ci run --rm --build ci sh -c \
		"python -m src.data.pipeline && python -m src.models.train && \
		 python -m src.models.evaluate && python -m src.models.export_onnx && \
		 python -m src.models.registry"

docker-lint:
	docker compose --profile ci run --rm --build lint

docker-test:
	docker compose --profile ci run --rm --build ci

# Mesmo gerador de carga do `load-test`, mas de dentro da rede do compose, para quem seguiu o
# caminho Docker e não tem Poetry no host. O endereço precisa ser explícito: na rede interna as
# duas APIs escutam na 8000, e o mapeamento para a 8001 só existe do lado de fora.
docker-load-test:
	docker compose --profile ci run --rm --build ci python scripts/load_test.py \
	  --base-url http://api-sklearn:8000 --backend sklearn --rps 10 --duracao 60
	docker compose --profile ci run --rm ci python scripts/load_test.py \
	  --base-url http://api-onnx:8000 --backend onnx --rps 10 --duracao 60

benchmark:
	poetry run python scripts/benchmark_latency.py

benchmark-comparar:
	poetry run python scripts/benchmark_latency.py --comparar

# ── Observabilidade ───────────────────────────────────────────────
monitoring-up:
	docker compose up -d --build

monitoring-down:
	docker compose down

load-test:
	poetry run python scripts/load_test.py --rps 10 --duracao 60 --backend sklearn
	poetry run python scripts/load_test.py --rps 10 --duracao 60 --backend onnx

# ── Orquestração (Airflow) ────────────────────────────────────────
# --env-file aponta pro airflow/.env: é a fonte das variáveis que o docker compose interpola
# no próprio compose (POSTGRES_*, AIRFLOW_UID, AIRFLOW_ADMIN_*) — sem ele, o compose não acha
# essas variáveis (o .env da raiz é outro arquivo, para a app; env_file: no compose só injeta
# variáveis dentro do container, não alimenta a interpolação do próprio arquivo).
#
# Serviços nomeados explicitamente, sem --profile: api-sklearn/api-onnx/prometheus/grafana não
# têm `profiles:` no compose (sobem em qualquer `up` por padrão), então `--profile airflow`
# sozinho não isola a stack de orquestração da de serving — as duas subiriam juntas. Listar os
# serviços do perfil airflow por nome é o que garante que só eles sobem (risco 3 do projeto:
# RAM limitada no ambiente de desenvolvimento local).
AIRFLOW_SERVICES = airflow-postgres airflow-init airflow-webserver airflow-scheduler airflow-dag-processor

airflow-up:
	# airflow/logs/ está fora do git (estado local) e o compose monta esse diretório num
	# volume — se ele não existe, o Docker o cria como root na primeira subida, e o container
	# (rodando com o uid do host) não consegue escrever nele. mkdir -p aqui garante que a
	# pasta já existe com o dono certo antes do primeiro `up`.
	mkdir -p airflow/logs
	python3 scripts/gen_airflow_env.py
	docker compose --env-file airflow/.env up -d --build $(AIRFLOW_SERVICES)

airflow-down:
	docker compose --env-file airflow/.env down $(AIRFLOW_SERVICES)

# Recria o metadata DB do Airflow do zero e sobe a stack de novo. Serve para quando o
# `airflow-init` falha com `password authentication failed for user "airflow"`: o Postgres só
# lê POSTGRES_PASSWORD ao inicializar um volume vazio, então um volume deixado por uma versão
# anterior do projeto (que gerava essa senha aleatoriamente) segue esperando a senha de então,
# e nenhuma mudança no airflow/.env alcança ele. `docker compose down -v` também resolveria,
# mas levaria junto os volumes do Prometheus e do Grafana, apagando o histórico de métricas —
# por isso o volume é removido pelo nome (prefixo `triagem`, de `name:` no compose raiz).
airflow-reset:
	docker compose --env-file airflow/.env down $(AIRFLOW_SERVICES)
	docker volume rm -f triagem_airflow_postgres_data
	$(MAKE) airflow-up

dag-test:
	docker compose --env-file airflow/.env build airflow-scheduler
	# Sem --no-deps: o entrypoint da imagem oficial do Airflow espera o Postgres antes de
	# executar qualquer comando, mesmo um alheio ao airflow — precisa do banco de pé, então
	# deixamos o compose subir `airflow-postgres`/`airflow-init` como dependência.
	# --entrypoint python3.12 -m pytest, não o script `pytest`: o entrypoint padrão da imagem
	# trata todo argumento como subcomando do `airflow` (precisa ser substituído), e o script
	# `pytest` tem shebang fixo em `/usr/python/bin/python3.12` (o interpretador base da
	# imagem) — sob o UID não-root que o compose usa para rodar as tasks, esse interpretador
	# não enxerga o site-packages de usuário onde o requirements.txt foi instalado, e o import
	# de `_pytest` falha. `python3.12` resolvido via PATH (o do venv do Airflow) não tem esse
	# problema.
	docker compose --env-file airflow/.env run --rm \
		--entrypoint python3.12 airflow-scheduler \
		-m pytest /opt/airflow/project/tests/test_dag.py -v

# ── Limpeza ───────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
