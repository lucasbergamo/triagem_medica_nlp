"""Testes de src/models/export_onnx.py — construção do grafo Gemm+Softmax, validação de
equivalência (fp32) e de degradação (int8), dedupe de opset e o fluxo run() ponta a ponta.

Dados sintéticos, sem rede — mesmo padrão de tests/test_models.py. `TOLERANCIA_PROBA_FP32` e
`TOLERANCIA_DEGRADACAO_MACRO_F1_INT8` são exercitadas com casos que realmente cruzam a
tolerância (coeficientes escalados, quantização de verdade), não com asserts cosméticos.
"""

import copy

import joblib
import numpy as np
import onnx
import pandas as pd
import pytest
from onnx import helper
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.models import export_onnx as export_onnx_mod
from src.models.train import build_pipeline


def _dataset_sintetico(n_por_classe: int = 20) -> pd.DataFrame:
    vocab = {
        "urgente": ["dor precordial infarto emergencia", "cancer estadiamento urgente"],
        "atencao": ["dor cronica cefaleia moderada", "sintoma digestivo persistente"],
        "normal": ["consulta rotina check up", "exame preventivo anual"],
    }
    linhas = [
        {"texto": f"{frase} caso {i}", "urgencia": classe}
        for classe, frases in vocab.items()
        for i in range(n_por_classe)
        for frase in [frases[i % len(frases)]]
    ]
    return pd.DataFrame(linhas)


@pytest.fixture()
def pipeline_treinado():
    df = _dataset_sintetico()
    pipeline = build_pipeline(seed=42)
    pipeline.fit(df["texto"], df["urgencia"])
    return pipeline, df


def _matriz_tfidf(pipeline, textos: list[str]) -> np.ndarray:
    return pipeline.named_steps["tfidf"].transform(textos).toarray().astype(np.float32)


# ── _construir_grafo_classificador / _inferir_onnx ──────────────────────


def test_grafo_gemm_softmax_reproduz_predict_proba_do_sklearn(pipeline_treinado):
    pipeline, df = pipeline_treinado
    x = _matriz_tfidf(pipeline, df["texto"].tolist())

    modelo_onnx = export_onnx_mod._construir_grafo_classificador(pipeline)
    probas_onnx = export_onnx_mod._inferir_onnx(modelo_onnx.SerializeToString(), x)
    probas_sklearn = pipeline.named_steps["clf"].predict_proba(x)

    assert np.max(np.abs(probas_onnx - probas_sklearn)) < 1e-5


# ── _validar_fp32 ─────────────────────────────────────────────────────


def test_validar_fp32_passa_quando_o_grafo_e_equivalente(pipeline_treinado):
    pipeline, df = pipeline_treinado
    x = _matriz_tfidf(pipeline, df["texto"].tolist())
    modelo_onnx = export_onnx_mod._construir_grafo_classificador(pipeline)

    export_onnx_mod._validar_fp32(pipeline, x, modelo_onnx.SerializeToString())  # não levanta


def test_validar_fp32_falha_quando_diverge_alem_da_tolerancia(pipeline_treinado):
    """Constrói o grafo ONNX a partir de um classificador com coeficientes escalados — uma
    divergência real, não cosmética — e valida contra o pipeline original."""
    pipeline, df = pipeline_treinado
    x = _matriz_tfidf(pipeline, df["texto"].tolist())

    pipeline_divergente = copy.deepcopy(pipeline)
    pipeline_divergente.named_steps["clf"].coef_ = pipeline_divergente.named_steps["clf"].coef_ * 5
    modelo_onnx = export_onnx_mod._construir_grafo_classificador(pipeline_divergente)

    with pytest.raises(ValueError, match="diverge do sklearn"):
        export_onnx_mod._validar_fp32(pipeline, x, modelo_onnx.SerializeToString())


# ── _validar_int8 ─────────────────────────────────────────────────────


def test_validar_int8_passa_dentro_da_tolerancia_de_degradacao(tmp_path, pipeline_treinado):
    pipeline, df = pipeline_treinado
    x = _matriz_tfidf(pipeline, df["texto"].tolist())
    y = df["urgencia"].tolist()

    modelo_onnx = export_onnx_mod._construir_grafo_classificador(pipeline)
    export_onnx_mod._dedupe_opset(modelo_onnx)
    path_fp32 = tmp_path / "clf.onnx"
    path_int8 = tmp_path / "clf.int8.onnx"
    path_fp32.write_bytes(modelo_onnx.SerializeToString())
    quantize_dynamic(str(path_fp32), str(path_int8), weight_type=QuantType.QInt8)

    resultado = export_onnx_mod._validar_int8(pipeline, x, y, path_int8.read_bytes())

    assert resultado["macro_f1_sklearn"] >= resultado["macro_f1_int8"] - 1e-9
    assert "divergencias_int8" in resultado


def test_validar_int8_falha_quando_degradacao_passa_do_piso(pipeline_treinado, monkeypatch):
    """Não depende de conseguir quantizar mal de propósito — baixa o piso de tolerância a
    zero e usa o próprio fp32 (equivalente, mas não bit-a-bit) como se fosse o int8: caminho
    determinístico para exercitar a mensagem de erro."""
    pipeline, df = pipeline_treinado
    x = _matriz_tfidf(pipeline, df["texto"].tolist())
    y = df["urgencia"].tolist()
    monkeypatch.setattr(export_onnx_mod, "TOLERANCIA_DEGRADACAO_MACRO_F1_INT8", -1.0)

    modelo_onnx = export_onnx_mod._construir_grafo_classificador(pipeline)

    with pytest.raises(ValueError, match="degrada macro-F1"):
        export_onnx_mod._validar_int8(pipeline, x, y, modelo_onnx.SerializeToString())


# ── _dedupe_opset ─────────────────────────────────────────────────────


def test_dedupe_opset_remove_entradas_duplicadas_mantendo_a_maior_versao():
    modelo = helper.make_model(
        helper.make_graph([], "vazio", [], []),
        opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("", 22)],
    )

    export_onnx_mod._dedupe_opset(modelo)

    assert len(modelo.opset_import) == 1
    assert modelo.opset_import[0].version == 22


# ── run() ─────────────────────────────────────────────────────────────


def test_run_exporta_dois_artefatos_a_partir_do_staging(tmp_path, monkeypatch, pipeline_treinado):
    pipeline, df = pipeline_treinado
    staging = tmp_path / "staging"
    staging.mkdir()
    joblib.dump(pipeline, staging / "pipeline.joblib")

    monkeypatch.setattr(export_onnx_mod, "MODELS_STAGING_DIR", staging)
    monkeypatch.setattr(export_onnx_mod, "load_gold", lambda: (df.iloc[:0], df.iloc[:0], df, {}))

    resultado = export_onnx_mod.run()

    assert (staging / "pipeline.onnx").exists()
    assert (staging / "pipeline.int8.onnx").exists()
    assert resultado["n_amostras_validadas"] == len(df)
    assert isinstance(onnx.load(staging / "pipeline.onnx"), onnx.ModelProto)
