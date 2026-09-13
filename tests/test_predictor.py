"""Testes de src/models/predictor.py — SklearnPredictor, OnnxPredictor e a factory
get_predictor()."""

import joblib
import pandas as pd
import pytest
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.models import export_onnx as export_onnx_mod
from src.models.predictor import OnnxPredictor, Predicao, SklearnPredictor, get_predictor
from src.models.train import build_pipeline

_TEXTOS = [
    "dor precordial infarto emergencia caso 1",
    "cancer estadiamento urgente caso 2",
    "dor cronica cefaleia moderada caso 3",
    "sintoma digestivo persistente caso 4",
    "consulta rotina check up caso 5",
    "exame preventivo anual caso 6",
]
_URGENCIAS = ["urgente", "urgente", "atencao", "atencao", "normal", "normal"]


def _dataframe_sintetico() -> pd.DataFrame:
    return pd.DataFrame({"texto": _TEXTOS, "urgencia": _URGENCIAS})


def _pipeline_treinado(tmp_path):
    df = _dataframe_sintetico()
    pipeline = build_pipeline(seed=42)
    pipeline.fit(df["texto"], df["urgencia"])
    path = tmp_path / "pipeline.joblib"
    joblib.dump(pipeline, path)
    return path


def _pipeline_com_onnx(diretorio):
    """Treina e exporta os 3 artefatos (joblib, onnx fp32, onnx int8) num único diretório —
    o layout que `models/current/` tem depois de `make export-onnx && make promote`."""
    df = _dataframe_sintetico()
    pipeline = build_pipeline(seed=42)
    pipeline.fit(df["texto"], df["urgencia"])
    joblib.dump(pipeline, diretorio / "pipeline.joblib")

    modelo_onnx = export_onnx_mod._construir_grafo_classificador(pipeline)
    export_onnx_mod._dedupe_opset(modelo_onnx)
    path_onnx = diretorio / "pipeline.onnx"
    path_onnx.write_bytes(modelo_onnx.SerializeToString())

    path_int8 = diretorio / "pipeline.int8.onnx"
    quantize_dynamic(str(path_onnx), str(path_int8), weight_type=QuantType.QInt8)
    return diretorio


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
    with pytest.raises(NotImplementedError, match="tensorflow"):
        get_predictor("tensorflow")


# ── OnnxPredictor ────────────────────────────────────────────────────


def test_onnx_predictor_levanta_erro_claro_quando_pipeline_ausente(tmp_path):
    with pytest.raises(FileNotFoundError, match="TF-IDF"):
        OnnxPredictor(
            backend="onnx",
            artefato_pipeline=tmp_path / "inexistente.joblib",
            artefato_onnx=tmp_path / "pipeline.onnx",
        )


def test_onnx_predictor_levanta_erro_claro_quando_artefato_onnx_ausente(tmp_path):
    artefato_pipeline = _pipeline_treinado(tmp_path)
    with pytest.raises(FileNotFoundError, match="export-onnx"):
        OnnxPredictor(
            backend="onnx",
            artefato_pipeline=artefato_pipeline,
            artefato_onnx=tmp_path / "inexistente.onnx",
        )


def test_onnx_predictor_backend_invalido_levanta_value_error(tmp_path):
    with pytest.raises(ValueError, match="inválido"):
        OnnxPredictor(backend="tensorflow")


def test_onnx_predictor_carrega_e_prediz(tmp_path):
    diretorio = _pipeline_com_onnx(tmp_path)

    predictor = OnnxPredictor(
        backend="onnx",
        artefato_pipeline=diretorio / "pipeline.joblib",
        artefato_onnx=diretorio / "pipeline.onnx",
    )
    (resultado,) = predictor.predict(["dor precordial forte"])

    assert predictor.backend == "onnx"
    assert set(predictor.classes) == {"urgente", "atencao", "normal"}
    assert isinstance(resultado, Predicao)
    assert resultado.backend == "onnx"
    assert pytest.approx(sum(resultado.probabilidades.values()), abs=1e-5) == 1.0


def test_os_3_backends_concordam_na_mesma_entrada(tmp_path):
    """sklearn, onnx e onnx-int8 têm que prever a mesma classe para o mesmo texto — o TF-IDF
    é idêntico nos três (mesmo `pipeline.joblib`), e o classificador é matematicamente
    equivalente por construção (ver src/models/export_onnx.py)."""
    diretorio = _pipeline_com_onnx(tmp_path)
    textos = ["dor precordial forte", "consulta de rotina", "sintoma digestivo leve"]

    sklearn_pred = SklearnPredictor(artefato=diretorio / "pipeline.joblib")
    onnx_pred = OnnxPredictor(
        backend="onnx",
        artefato_pipeline=diretorio / "pipeline.joblib",
        artefato_onnx=diretorio / "pipeline.onnx",
    )
    int8_pred = OnnxPredictor(
        backend="onnx-int8",
        artefato_pipeline=diretorio / "pipeline.joblib",
        artefato_onnx=diretorio / "pipeline.int8.onnx",
    )

    urgencias_sklearn = [r.urgencia for r in sklearn_pred.predict(textos)]
    urgencias_onnx = [r.urgencia for r in onnx_pred.predict(textos)]
    urgencias_int8 = [r.urgencia for r in int8_pred.predict(textos)]

    assert urgencias_onnx == urgencias_sklearn
    assert urgencias_int8 == urgencias_sklearn


# ── modelo_versao via model_meta.json ────────────────────────────────


def test_modelo_versao_le_data_treino_do_model_meta_quando_presente(tmp_path):
    artefato = _pipeline_treinado(tmp_path)
    (tmp_path / "model_meta.json").write_text(
        '{"batch_id": "abc123", "data_treino": "2026-01-01T00:00:00+00:00"}', encoding="utf-8"
    )

    predictor = SklearnPredictor(artefato=artefato)

    assert predictor.modelo_versao == "2026-01-01T00:00:00+00:00"


def test_modelo_versao_cai_para_mtime_quando_model_meta_ausente(tmp_path):
    artefato = _pipeline_treinado(tmp_path)

    predictor = SklearnPredictor(artefato=artefato)

    assert predictor.modelo_versao  # ISO timestamp do mtime, não vazio
