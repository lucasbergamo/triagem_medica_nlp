"""Predictor — Strategy selecionável em runtime para o backend de inferência.

`SklearnPredictor` (TfidfVectorizer + LogisticRegression via scipy/sklearn puro) e
`OnnxPredictor`, cobrindo os backends `onnx` e `onnx-int8`. O TF-IDF continua em Python nos
dois backends ONNX — decisão forçada por uma limitação real do tokenizador do skl2onnx, não
por falta de esforço (ver `src/models/export_onnx.py` e `docs/latencia.md`). `get_predictor()`
lê `MODEL_BACKEND` das settings.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np
from joblib import load
from onnxruntime import InferenceSession

from src.utils.config import MODELS_CURRENT_DIR, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

NOME_ARTEFATO_SKLEARN = "pipeline.joblib"
NOME_ARTEFATO_ONNX = "pipeline.onnx"
NOME_ARTEFATO_ONNX_INT8 = "pipeline.int8.onnx"
NOME_METADADOS = "model_meta.json"


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


def _versao_do_modelo(diretorio: Path, artefato: Path) -> str:
    """`modelo_versao` vem de `model_meta.json` (`data_treino`, gravado pelo treino e
    enriquecido pela avaliação — ver `src/models/train.py` e `src/models/evaluate.py`), nunca
    do mtime do arquivo: metadado de sistema de arquivos não sobrevive a um download do S3
    nem a qualquer cópia que não preserve mtime. Sem o arquivo (artefato solto num teste, por
    exemplo), cai para o mtime — comportamento anterior a este metadado, mantido só como rede
    de segurança.
    """
    meta_path = diretorio / NOME_METADADOS
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta["data_treino"]
    logger.warning("model_meta_ausente", diretorio=str(diretorio))
    return datetime.fromtimestamp(artefato.stat().st_mtime, tz=UTC).isoformat()


class SklearnPredictor:
    """Baseline: vetorização e classificação em Python/scipy, dentro do processo da API.

    Carregado uma vez (no `lifespan` da API — nunca por requisição).
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
        self.modelo_versao = _versao_do_modelo(artefato.parent, artefato)
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


class OnnxPredictor:
    """Backends `onnx` e `onnx-int8`: TF-IDF em Python — o mesmo `pipeline.joblib` do backend
    `sklearn`, só a etapa de vetorização — e classificador em ONNX Runtime.

    O TF-IDF não entra no grafo ONNX. Não por falta de esforço: a exportação do pipeline
    inteiro esbarrou num tokenizador que diverge do sklearn em ngrams e casos de borda —
    medido, não suposto (ver `src/models/export_onnx.py` e `docs/latencia.md`). O ganho do
    ONNX aqui vem só da etapa de classificação, que é onde o backend realmente troca de
    motor de execução.
    """

    def __init__(
        self,
        backend: str = "onnx",
        artefato_pipeline: Path | None = None,
        artefato_onnx: Path | None = None,
    ) -> None:
        if backend not in ("onnx", "onnx-int8"):
            raise ValueError(f"Backend {backend!r} inválido para OnnxPredictor.")
        self.backend = backend

        artefato_pipeline = artefato_pipeline or (MODELS_CURRENT_DIR / NOME_ARTEFATO_SKLEARN)
        nome_onnx = NOME_ARTEFATO_ONNX if backend == "onnx" else NOME_ARTEFATO_ONNX_INT8
        artefato_onnx = artefato_onnx or (MODELS_CURRENT_DIR / nome_onnx)

        if not artefato_pipeline.exists():
            raise FileNotFoundError(
                f"Artefato do vetorizador não encontrado em {artefato_pipeline} — o backend "
                f"{backend!r} usa o TF-IDF do pipeline sklearn. Rode `make train && make "
                "export-onnx && make promote` antes de subir a API."
            )
        if not artefato_onnx.exists():
            raise FileNotFoundError(
                f"Artefato ONNX não encontrado em {artefato_onnx}. Rode `make export-onnx && "
                f"make promote` antes de subir a API com MODEL_BACKEND={backend}."
            )

        pipeline = load(artefato_pipeline)
        self._tfidf = pipeline.named_steps["tfidf"]
        self.classes: list[str] = list(pipeline.classes_)
        self._sessao = InferenceSession(str(artefato_onnx), providers=["CPUExecutionProvider"])
        self._entrada = self._sessao.get_inputs()[0].name
        self.modelo_versao = _versao_do_modelo(artefato_onnx.parent, artefato_onnx)

        logger.info(
            "predictor_carregado",
            backend=self.backend,
            classes=self.classes,
            artefato=str(artefato_onnx),
        )

    def predict(self, textos: list[str]) -> list[Predicao]:
        """TF-IDF em sklearn, classificação em ONNX Runtime — a latência reportada cobre as
        duas etapas, exatamente como o baseline sklearn cobre vetorização + classificação."""
        inicio = time.perf_counter()
        x = self._tfidf.transform(textos).toarray().astype(np.float32)
        (probas,) = self._sessao.run(None, {self._entrada: x})
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
    if backend in ("onnx", "onnx-int8"):
        return OnnxPredictor(backend=backend)
    raise NotImplementedError(
        f"Backend {backend!r} não suportado — use 'sklearn', 'onnx' ou 'onnx-int8'."
    )
