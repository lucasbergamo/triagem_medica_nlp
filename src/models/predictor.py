"""Predictor — Strategy selecionável em runtime para o backend de inferência.

`SklearnPredictor` é o único backend implementado neste bloco (TfidfVectorizer +
LogisticRegression via scipy/sklearn puro). `OnnxPredictor`, cobrindo os backends `onnx` e
`onnx-int8`, entra no bloco de otimização de latência — `get_predictor()` já lê
`MODEL_BACKEND` das settings, então trocar de backend depois é só implementar a classe e
registrar na factory, sem tocar na API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from joblib import load

from src.utils.config import MODELS_CURRENT_DIR, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

NOME_ARTEFATO_SKLEARN = "pipeline.joblib"


@dataclass(frozen=True)
class Predicao:
    urgencia: str
    confianca: float
    probabilidades: dict[str, float]
    latencia_inferencia_ms: float
    backend: str
    modelo_versao: str


class Predictor(Protocol):
    backend: str
    modelo_versao: str
    classes: list[str]

    def predict(self, textos: list[str]) -> list[Predicao]: ...


class SklearnPredictor:
    """Baseline: vetorização e classificação em Python/scipy, dentro do processo da API.

    Carregado uma vez (no `lifespan` da API — nunca por requisição). `modelo_versao` é o
    horário de modificação do arquivo do artefato: `registry.promover()` usa `shutil.copy2`,
    que preserva o mtime original do `joblib.dump` feito em `train.py` — ou seja, o valor
    reflete o horário real do treino, não o da promoção.
    """

    backend = "sklearn"

    def __init__(self, artefato: Path | None = None) -> None:
        artefato = artefato or (MODELS_CURRENT_DIR / NOME_ARTEFATO_SKLEARN)
        if not artefato.exists():
            raise FileNotFoundError(
                f"Artefato do modelo não encontrado em {artefato} — rode "
                "`make train && make eval && make promote` antes de subir a API."
            )
        self._pipeline = load(artefato)
        self.classes: list[str] = list(self._pipeline.classes_)
        self.modelo_versao = datetime.fromtimestamp(artefato.stat().st_mtime, tz=UTC).isoformat()
        logger.info(
            "predictor_carregado",
            backend=self.backend,
            classes=self.classes,
            artefato=str(artefato),
        )

    def predict(self, textos: list[str]) -> list[Predicao]:
        """Prediz o lote inteiro numa única chamada ao pipeline (mais eficiente que texto a
        texto). `latencia_inferencia_ms` reportada por item é a média do lote — exata quando
        `len(textos) == 1` (caso do `/predict`), amostral quando é um batch de verdade."""
        inicio = time.perf_counter()
        probas = self._pipeline.predict_proba(textos)
        latencia_media_ms = (time.perf_counter() - inicio) * 1000 / len(textos)

        resultados = []
        for linha in probas:
            distribuicao = dict(zip(self.classes, (float(p) for p in linha), strict=True))
            urgencia = max(distribuicao, key=distribuicao.get)
            resultados.append(
                Predicao(
                    urgencia=urgencia,
                    confianca=distribuicao[urgencia],
                    probabilidades=distribuicao,
                    latencia_inferencia_ms=round(latencia_media_ms, 3),
                    backend=self.backend,
                    modelo_versao=self.modelo_versao,
                )
            )
        return resultados


def get_predictor(backend: str | None = None) -> Predictor:
    """Factory — lê `MODEL_BACKEND` das settings quando `backend` não é informado."""
    backend = backend or settings.model_backend
    if backend == "sklearn":
        return SklearnPredictor()
    raise NotImplementedError(
        f"Backend {backend!r} ainda não implementado — só 'sklearn' está disponível "
        "até o bloco de otimização ONNX."
    )
