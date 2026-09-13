"""Middleware ASGI de métricas de requisição — mesma abordagem do `@app.middleware("http")`
da aula 04.02 (`api_instrumented.py`). Cobre as métricas de **requisição** (`ml_active_requests`,
`ml_prediction_latency_seconds`, `ml_requests_total`, `ml_errors_total`); as métricas de
**modelo** (`ml_predictions_total`, `ml_prediction_confidence`, `ml_inference_duration_seconds`)
são instrumentadas no ponto da inferência, em `src/api/main.py` — são coisas diferentes: uma
requisição de `/predict/batch` é uma única requisição HTTP, mas gera várias predições.
"""

import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from src.api.metrics import (
    ML_ACTIVE_REQUESTS,
    ML_ERRORS_TOTAL,
    ML_PREDICTION_LATENCY_SECONDS,
    ML_REQUESTS_TOTAL,
)

Handler = Callable[[Request], Awaitable[Response]]


async def track_request_metrics(request: Request, call_next: Handler) -> Response:
    """`/metrics` fica de fora — expor a própria coleta como requisição instrumentada infla o
    histograma com o ruído do scrape do Prometheus a cada 5s, sem informação nova."""
    if request.url.path == "/metrics":
        return await call_next(request)

    endpoint = request.url.path
    ML_ACTIVE_REQUESTS.labels(endpoint=endpoint).inc()
    inicio = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        ML_ERRORS_TOTAL.labels(tipo="excecao_nao_tratada").inc()
        raise
    finally:
        ML_ACTIVE_REQUESTS.labels(endpoint=endpoint).dec()

    duracao = time.perf_counter() - inicio
    ML_PREDICTION_LATENCY_SECONDS.labels(endpoint=endpoint).observe(duracao)
    ML_REQUESTS_TOTAL.labels(
        endpoint=endpoint, metodo=request.method, status_code=str(response.status_code)
    ).inc()
    if response.status_code >= 400:
        categoria = "cliente" if response.status_code < 500 else "servidor"
        ML_ERRORS_TOTAL.labels(tipo=categoria).inc()

    return response
