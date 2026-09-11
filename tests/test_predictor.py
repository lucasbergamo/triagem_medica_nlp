"""Testes de src/models/predictor.py — SklearnPredictor e a factory get_predictor()."""

import joblib
import pandas as pd
import pytest

from src.models.predictor import Predicao, SklearnPredictor, get_predictor
from src.models.train import build_pipeline


def _pipeline_treinado(tmp_path):
    df = pd.DataFrame(
        {
            "texto": [
                "dor precordial infarto emergencia caso 1",
                "cancer estadiamento urgente caso 2",
                "dor cronica cefaleia moderada caso 3",
                "sintoma digestivo persistente caso 4",
                "consulta rotina check up caso 5",
                "exame preventivo anual caso 6",
            ],
            "urgencia": ["urgente", "urgente", "atencao", "atencao", "normal", "normal"],
        }
    )
    pipeline = build_pipeline(seed=42)
    pipeline.fit(df["texto"], df["urgencia"])
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    return path


def test_sklearn_predictor_levanta_erro_claro_quando_artefato_nao_existe(tmp_path):
    with pytest.raises(FileNotFoundError, match="make train"):
        SklearnPredictor(artefato=tmp_path / "inexistente.joblib")


def test_sklearn_predictor_carrega_classes_e_versao_do_artefato(tmp_path):
    path = _pipeline_treinado(tmp_path)

    predictor = SklearnPredictor(artefato=path)

    assert predictor.backend == "sklearn"
    assert set(predictor.classes) == {"urgente", "atencao", "normal"}
    assert predictor.modelo_versao  # ISO timestamp não vazio


def test_predict_devolve_uma_predicao_por_texto_com_probabilidades_somando_um(tmp_path):
    path = _pipeline_treinado(tmp_path)
    predictor = SklearnPredictor(artefato=path)

    resultados = predictor.predict(["dor precordial forte", "consulta de rotina"])

    assert len(resultados) == 2
    for predicao in resultados:
        assert isinstance(predicao, Predicao)
        assert predicao.urgencia in {"urgente", "atencao", "normal"}
        assert pytest.approx(sum(predicao.probabilidades.values()), abs=1e-6) == 1.0
        assert predicao.probabilidades[predicao.urgencia] == predicao.confianca
        assert predicao.latencia_inferencia_ms >= 0
        assert predicao.backend == "sklearn"


def test_predict_com_um_unico_texto_reporta_latencia_exata_nao_dividida(tmp_path):
    path = _pipeline_treinado(tmp_path)
    predictor = SklearnPredictor(artefato=path)

    (resultado,) = predictor.predict(["dor precordial forte"])

    assert resultado.latencia_inferencia_ms >= 0


def test_get_predictor_sklearn_le_o_backend_das_settings(tmp_path, monkeypatch):
    path = _pipeline_treinado(tmp_path)
    monkeypatch.setattr("src.models.predictor.MODELS_CURRENT_DIR", tmp_path)

    predictor = get_predictor("sklearn")

    assert predictor.backend == "sklearn"
    assert path.exists()


def test_get_predictor_backend_nao_implementado_levanta_not_implemented_error():
    with pytest.raises(NotImplementedError, match="onnx"):
        get_predictor("onnx")
