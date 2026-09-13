"""API HTTP de triagem de laudos por urgência.

Todos os endpoints abaixo são `def`, nunca `async def`. A inferência (TF-IDF + regressão
logística) é CPU-bound e síncrona: dentro de uma coroutine ela bloqueia o event loop, e todas
as requisições concorrentes passam a ser atendidas em fila — o que invalidaria qualquer
medição de latência sob carga (`scripts/load_test.py`). Com `def`, o Starlette despacha a
chamada para o threadpool automaticamente, liberando o event loop para outras requisições.
`/health`, `/ready` e `/model/info` não têm trabalho pesado, mas ficam `def` por uniformidade
com os dois endpoints que importam.
"""

from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from src.api.metrics import (
    ML_INFERENCE_DURATION_SECONDS,
    ML_PREDICTION_CONFIDENCE,
    ML_PREDICTIONS_TOTAL,
    gerar_exposicao,
    registrar_model_info,
)
from src.api.middleware import track_request_metrics
from src.api.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictBatchRequest,
    PredictBatchResponse,
    PredictRequest,
    PredictResponse,
    ReadyResponse,
)
from src.models.predictor import Predicao, Predictor, get_predictor
from src.utils.logger import get_logger

logger = get_logger(__name__)

_state: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o modelo uma única vez — jamais por requisição. Falha de carregamento não
    derruba o processo: fica registrada em `_state` para o `/ready` reportar 503 sem tirar o
    `/health` do ar — é o que permite o ECS reciclar a task em vez de achar o container morto."""
    try:
        predictor = get_predictor()
        _state["predictor"] = predictor
        logger.info("modelo_carregado", backend=predictor.backend)
        registrar_model_info(
            backend=predictor.backend,
            versao=predictor.modelo_versao,
            n_classes=len(predictor.classes),
        )
    except Exception as exc:  # qualquer falha aqui vira "não pronto", não derruba o processo
        _state["predictor"] = None
        logger.error("falha_ao_carregar_modelo", erro=str(exc))
    yield
    _state.clear()


app = FastAPI(
    title="Triagem Médica NLP",
    description="Triagem automática de laudos médicos por urgência — Tech Challenge Fase 03",
    version="0.1.0",
    lifespan=lifespan,
)
app.middleware("http")(track_request_metrics)


def _traduzir_erro_validacao(erro: dict) -> str:
    """Traduz um erro do Pydantic para uma frase em português — a API nunca devolve as
    mensagens padrão do Pydantic, que saem em inglês."""
    tipo = erro["type"]
    campo = ".".join(str(parte) for parte in erro["loc"][1:]) or "corpo da requisição"

    if tipo == "string_too_short":
        return f"'{campo}' deve ter no mínimo {erro['ctx']['min_length']} caracteres."
    if tipo == "string_too_long":
        return f"'{campo}' deve ter no máximo {erro['ctx']['max_length']} caracteres."
    if tipo == "too_short":
        return f"'{campo}' precisa de pelo menos {erro['ctx']['min_length']} item(ns)."
    if tipo == "too_long":
        return f"'{campo}' aceita no máximo {erro['ctx']['max_length']} item(ns)."
    if tipo == "missing":
        return f"'{campo}' é obrigatório."
    return f"'{campo}' é inválido: {erro['msg']}"


@app.exception_handler(RequestValidationError)
def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    mensagem = " ".join(_traduzir_erro_validacao(erro) for erro in exc.errors())
    return JSONResponse(status_code=422, content={"detail": mensagem})


def _predictor_pronto() -> Predictor:
    predictor = _state.get("predictor")
    if predictor is None:
        detail = "Modelo não carregado — serviço não está pronto."
        raise HTTPException(status_code=503, detail=detail)
    return predictor  # type: ignore[return-value]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness: só confirma que o processo está de pé. Nunca toca o modelo."""
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    """Readiness: modelo carregado e respondendo — o healthcheck que ECS/ALB usam antes de
    mandar tráfego para a task."""
    _predictor_pronto()
    return ReadyResponse(status="ok")


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    predictor = _predictor_pronto()
    return ModelInfoResponse(
        backend=predictor.backend,
        modelo_versao=predictor.modelo_versao,
        # mesmo valor: `predictor.modelo_versao` já é o `data_treino` lido de
        # `models/current/model_meta.json` (gravado pelo treino, copiado pela promoção) —
        # ver `src/models/predictor.py::_versao_do_modelo`.
        data_treino=predictor.modelo_versao,
        classes=predictor.classes,
    )


def _registrar_predicoes(endpoint: str, predicoes: list[Predicao]) -> None:
    """Métricas de **modelo**, uma observação por predição do lote — diferente da métrica de
    **requisição** do middleware, que registra uma vez por chamada HTTP mesmo quando o lote
    tem 100 laudos."""
    for predicao in predicoes:
        ML_PREDICTIONS_TOTAL.labels(
            endpoint=endpoint, predicted_class=predicao.urgencia, status="sucesso"
        ).inc()
        ML_PREDICTION_CONFIDENCE.observe(predicao.confianca)
        ML_INFERENCE_DURATION_SECONDS.labels(backend=predicao.backend).observe(
            predicao.latencia_inferencia_ms / 1000
        )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    predictor = _predictor_pronto()
    try:
        (predicao,) = predictor.predict([request.texto])
    except Exception as exc:
        ML_PREDICTIONS_TOTAL.labels(endpoint="/predict", predicted_class="n/a", status="erro").inc()
        logger.error("falha_inferencia", erro=str(exc))
        raise HTTPException(status_code=503, detail="Falha ao gerar predição.") from exc
    _registrar_predicoes("/predict", [predicao])
    return PredictResponse(**asdict(predicao))


@app.post("/predict/batch", response_model=PredictBatchResponse)
def predict_batch(request: PredictBatchRequest) -> PredictBatchResponse:
    predictor = _predictor_pronto()
    textos = [laudo.texto for laudo in request.laudos]
    try:
        predicoes = predictor.predict(textos)
    except Exception as exc:
        ML_PREDICTIONS_TOTAL.labels(
            endpoint="/predict/batch", predicted_class="n/a", status="erro"
        ).inc()
        logger.error("falha_inferencia_batch", erro=str(exc), n_laudos=len(textos))
        raise HTTPException(status_code=503, detail="Falha ao gerar predições do lote.") from exc
    _registrar_predicoes("/predict/batch", predicoes)
    return PredictBatchResponse(resultados=[PredictResponse(**asdict(p)) for p in predicoes])


@app.get("/metrics")
def metrics() -> Response:
    """Exposição Prometheus (OpenMetrics) — scrape sem adaptador, local hoje e AMP depois."""
    corpo, content_type = gerar_exposicao()
    return Response(content=corpo, media_type=content_type)
