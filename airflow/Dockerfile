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
