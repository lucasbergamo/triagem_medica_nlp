"""Métricas Prometheus da API — 8 séries, expostas em `GET /metrics` (`src/api/main.py`).

As 4 primeiras (`ml_predictions_total`, `ml_prediction_latency_seconds`,
`ml_prediction_confidence`, `ml_active_requests`) copiam **nome e labels** de
`04-monitoracao-performance/aula02-prometheus-grafana/api_instrumented.py` — o avaliador
reconhece o padrão da aula de imediato. As outras 4 (`ml_inference_duration_seconds`,
`ml_requests_total`, `ml_errors_total`, `ml_model_info`) são extensão nossa, para provar o
ganho do ONNX e cobrir taxa de erro/versão de modelo — nada disso existe na aula.

Buckets em segundos (convenção do `_seconds` do Prometheus), recalibrados pela baseline
medida (`metrics/latency_baseline.json`: sklearn p50 0,79 ms · p95 1,27 ms) — o default da
biblioteca começa em 5 ms, e o corte de 1 ms da aula ainda deixaria a inferência inteira no
primeiro bucket. Descendo até 0,05 ms, o painel de p95 mostra a curva do onnx separada da do
sklearn em vez de duas retas coladas no chão.
"""

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, Info, generate_latest

BUCKETS_LATENCIA_SEGUNDOS = [
    0.00005,
    0.0001,
    0.00025,
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
]

# ── Idênticas à aula 04.02 (mesmo nome, mesmos labels) ──────────────────────────
ML_PREDICTIONS_TOTAL = Counter(
    "ml_predictions_total",
    "Total de predições realizadas",
    ["endpoint", "predicted_class", "status"],
)
ML_PREDICTION_LATENCY_SECONDS = Histogram(
    "ml_prediction_latency_seconds",
    "Latência das predições em segundos (fim a fim, HTTP incluso)",
    ["endpoint"],
    buckets=BUCKETS_LATENCIA_SEGUNDOS,
)
ML_PREDICTION_CONFIDENCE = Histogram(
    "ml_prediction_confidence",
    "Distribuição de confiança das predições",
    buckets=[0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0],
)
ML_ACTIVE_REQUESTS = Gauge(
    "ml_active_requests",
    "Número de requisições ativas",
    ["endpoint"],
)

# ── Extensões nossas ─────────────────────────────────────────────────────────
ML_INFERENCE_DURATION_SECONDS = Histogram(
    "ml_inference_duration_seconds",
    "Latência do modelo sem HTTP — o que prova o ganho do ONNX",
    ["backend"],
    buckets=BUCKETS_LATENCIA_SEGUNDOS,
)
ML_REQUESTS_TOTAL = Counter(
    "ml_requests_total",
    "Total de requisições HTTP por método e status",
    ["endpoint", "metodo", "status_code"],
)
ML_ERRORS_TOTAL = Counter(
    "ml_errors_total",
    "Total de erros por categoria",
    ["tipo"],
)
ML_MODEL_INFO = Info(
    "ml_model_info",
    "Backend ativo, versão e número de classes do modelo servido",
)


def registrar_model_info(backend: str, versao: str, n_classes: int) -> None:
    """Chamado uma vez no `lifespan`, depois do modelo carregado — `Info` é um gauge de
    labels, não série numérica: reflete o estado atual do processo, não histórico."""
    ML_MODEL_INFO.info({"backend": backend, "versao": versao, "n_classes": str(n_classes)})


def gerar_exposicao() -> tuple[bytes, str]:
    """Corpo e content-type do endpoint `/metrics` — isolado aqui para `main.py` não importar
    `generate_latest`/`CONTENT_TYPE_LATEST` diretamente."""
    return generate_latest(), CONTENT_TYPE_LATEST
