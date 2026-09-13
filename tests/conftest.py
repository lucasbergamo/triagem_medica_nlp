"""Fixtures compartilhadas entre `test_api.py` e `test_metrics.py` — o mesmo modelo sintético
serve os dois: contrato HTTP e instrumentação Prometheus são duas visões do mesmo endpoint,
não dois setups diferentes.
"""

from pathlib import Path

import joblib
import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.models.train import build_pipeline

# Vocabulário repetido de propósito: o TfidfVectorizer do pipeline real usa min_df=3, e um
# termo com menos de 3 ocorrências no corpus é descartado — o dataset sintético precisa
# respeitar o mesmo piso para o treino produzir um vocabulário não vazio.
_VOCAB_POR_CLASSE = {
    "urgente": ["dor precordial infarto emergencia", "cancer estadiamento urgente"],
    "atencao": ["dor cronica cefaleia moderada", "sintoma digestivo persistente"],
    "normal": ["consulta rotina check up", "exame preventivo anual"],
}


def _dataset_sintetico(n_por_classe: int = 5) -> tuple[list[str], list[str]]:
    textos, urgencias = [], []
    for classe, frases in _VOCAB_POR_CLASSE.items():
        for i in range(n_por_classe):
            textos.append(f"{frases[i % len(frases)]} caso {i}")
            urgencias.append(classe)
    return textos, urgencias


@pytest.fixture(scope="session")
def artefato_modelo_sintetico(tmp_path_factory) -> Path:
    """Treina um Pipeline mínimo (mesmo `build_pipeline` de src/models/train.py) com ~15
    frases sintéticas cobrindo as 3 classes, e salva como se fosse o artefato promovido —
    sessão inteira reusa o mesmo arquivo, treinado uma única vez."""
    textos, urgencias = _dataset_sintetico()
    pipeline = build_pipeline(seed=42)
    pipeline.fit(textos, urgencias)

    diretorio = tmp_path_factory.mktemp("models_current")
    path = diretorio / "pipeline.joblib"
    joblib.dump(pipeline, path)
    return path


@pytest.fixture()
def client(artefato_modelo_sintetico, monkeypatch):
    """TestClient com o predictor apontado para o modelo sintético — o monkeypatch precisa
    rodar antes do `with TestClient(...)`, porque é o `lifespan` que carrega o modelo."""
    monkeypatch.setattr("src.models.predictor.MODELS_CURRENT_DIR", artefato_modelo_sintetico.parent)
    with TestClient(main.app) as test_client:
        yield test_client
