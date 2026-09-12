"""Contratos Pydantic da API — validação de entrada e forma das respostas. Limites e mensagens
em português vivem aqui; a tradução dos erros de validação do Pydantic para texto legível
fica em `src/api/main.py`.
"""

from pydantic import BaseModel, Field

TEXTO_MIN_LENGTH = 20
TEXTO_MAX_LENGTH = 20_000
BATCH_MAX_LAUDOS = 100


class PredictRequest(BaseModel):
    texto: str = Field(
        min_length=TEXTO_MIN_LENGTH,
        max_length=TEXTO_MAX_LENGTH,
        description="Texto do laudo médico a triar.",
    )


class PredictBatchRequest(BaseModel):
    laudos: list[PredictRequest] = Field(
        min_length=1,
        max_length=BATCH_MAX_LAUDOS,
        description=f"Lote de 1 a {BATCH_MAX_LAUDOS} laudos.",
    )


class PredictResponse(BaseModel):
    urgencia: str
    confianca: float
    probabilidades: dict[str, float]
    latencia_inferencia_ms: float
    backend: str
    modelo_versao: str


class PredictBatchResponse(BaseModel):
    resultados: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    status: str


class ModelInfoResponse(BaseModel):
    backend: str
    modelo_versao: str
    data_treino: str
    classes: list[str]
