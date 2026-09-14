ARG AIRFLOW_IMAGE=apache/airflow:3.1.5-python3.12

FROM ${AIRFLOW_IMAGE} AS builder

ENV PATH="/home/airflow/.local/bin:${PATH}" \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt
COPY constraints.txt /constraints.txt
RUN pip install --no-cache-dir -r /requirements.txt --constraint /constraints.txt

# scikit-learn é a única dependência de requirements.txt cuja versão muda o resultado do
# treino, e o constraints.txt da Apache a fixa em 1.8.0, uma major abaixo da do projeto
# (poetry.lock: 1.9.1) — sem isso, a DAG treina com um scikit-learn diferente do `make train`
# local e produz um macro-F1 diferente do documentado. --no-deps evita que essa instalação
# arraste numpy/scipy para fora das versões que o constraints.txt já fixou para o Airflow
# (scikit-learn 1.9.1 só exige numpy>=1.24.1 e scipy>=1.10.0, satisfeitos pelas versões que
# o passo acima já instalou; joblib>=1.4.0 e threadpoolctl>=3.5.0 também já satisfeitos pelo
# que o constraints.txt instala). narwhals entra na mesma linha porque o scikit-learn 1.9
# passou a importá-lo incondicionalmente em utils/validation.py — é dependência real, não
# opcional, e --no-deps pula qualquer dependência que não venha listada explicitamente aqui.
RUN pip install --no-cache-dir --no-deps scikit-learn==1.9.1 narwhals==2.26.0

FROM ${AIRFLOW_IMAGE} AS runtime

ENV PATH="/home/airflow/.local/bin:${PATH}" \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

USER airflow
COPY --from=builder --chown=airflow:0 /home/airflow/.local /home/airflow/.local

# O provider do Google vem na imagem base e causa timeout no import de DAGs
# (incompatibilidade google-api-core + Python 3.12). Não é usado aqui.
RUN pip uninstall -y apache-airflow-providers-google 2>/dev/null || true

COPY --chown=airflow:0 plugins /opt/airflow/plugins
