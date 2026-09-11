"""Testes de src/models — treino, gate de avaliação e promoção. Dados sintéticos, sem rede."""

import joblib
import pandas as pd
import pytest

from src.models import evaluate as evaluate_mod
from src.models import registry as registry_mod
from src.models import train as train_mod
from src.models.store import LocalModelStore
from src.models.train import build_baseline_pipeline, build_pipeline


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


# ── train.py ──────────────────────────────────────────────────────────


def test_build_pipeline_treina_e_prediz_apenas_classes_conhecidas():
    df = _dataset_sintetico()
    pipeline = build_pipeline(seed=42)
    pipeline.fit(df["texto"], df["urgencia"])
    preds = pipeline.predict(df["texto"])
    assert set(preds) <= {"urgente", "atencao", "normal"}


def test_build_baseline_pipeline_tambem_treina_e_prediz():
    df = _dataset_sintetico()
    pipeline = build_baseline_pipeline(seed=42)
    pipeline.fit(df["texto"], df["urgencia"])
    preds = pipeline.predict(df["texto"])
    assert set(preds) <= {"urgente", "atencao", "normal"}


def test_run_treino_salva_artefato_em_staging_e_devolve_metricas(tmp_path, monkeypatch):
    df = _dataset_sintetico()
    monkeypatch.setattr(train_mod, "MODELS_STAGING_DIR", tmp_path)
    monkeypatch.setattr(train_mod, "load_gold", lambda: (df, df.iloc[:0], df.iloc[:0], {}))

    resultado = train_mod.run(batch_id="teste123")

    assert (tmp_path / "pipeline.joblib").exists()
    assert resultado["batch_id"] == "teste123"
    assert resultado["n_train"] == len(df)
    assert resultado["tamanho_artefato_kb"] > 0
    assert "baseline" not in resultado


def test_run_treino_com_baseline_salva_os_dois_artefatos(tmp_path, monkeypatch):
    df = _dataset_sintetico()
    monkeypatch.setattr(train_mod, "MODELS_STAGING_DIR", tmp_path)
    monkeypatch.setattr(train_mod, "load_gold", lambda: (df, df.iloc[:0], df.iloc[:0], {}))

    resultado = train_mod.run(batch_id="teste123", baseline=True)

    assert (tmp_path / "pipeline.joblib").exists()
    assert (tmp_path / "pipeline_rf_baseline.joblib").exists()
    assert resultado["baseline"]["modelo"] == "random_forest_tfidf"


def test_run_treino_gera_batch_id_quando_nao_informado(tmp_path, monkeypatch):
    df = _dataset_sintetico()
    monkeypatch.setattr(train_mod, "MODELS_STAGING_DIR", tmp_path)
    monkeypatch.setattr(train_mod, "load_gold", lambda: (df, df.iloc[:0], df.iloc[:0], {}))

    resultado = train_mod.run()

    assert resultado["batch_id"]  # não vazio


def test_run_treino_limpa_staging_antes_de_treinar(tmp_path, monkeypatch):
    """Um artefato de um batch anterior (ex.: pipeline.onnx já exportado) não pode
    sobreviver a um novo run() — senão a promoção mistura modelos de batches diferentes."""
    df = _dataset_sintetico()
    monkeypatch.setattr(train_mod, "MODELS_STAGING_DIR", tmp_path)
    monkeypatch.setattr(train_mod, "load_gold", lambda: (df, df.iloc[:0], df.iloc[:0], {}))
    (tmp_path / "pipeline.onnx").write_bytes(b"artefato-de-batch-anterior")

    train_mod.run(batch_id="teste123")

    assert not (tmp_path / "pipeline.onnx").exists()
    assert (tmp_path / "pipeline.joblib").exists()


# ── evaluate.py ───────────────────────────────────────────────────────


def test_avaliar_calcula_macro_f1_acuracia_e_matriz_de_confusao():
    df = _dataset_sintetico()
    pipeline = build_pipeline(seed=42)
    pipeline.fit(df["texto"], df["urgencia"])

    metrics = evaluate_mod.avaliar(pipeline, df["texto"], df["urgencia"])

    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert 0.0 <= metrics["acuracia"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 3
    assert set(metrics["labels"]) == {"urgente", "atencao", "normal"}
    assert metrics["n_amostras"] == len(df)


def test_run_avalia_val_e_test_separadamente(tmp_path, monkeypatch):
    df = _dataset_sintetico()
    pipeline = build_pipeline(seed=42)
    pipeline.fit(df["texto"], df["urgencia"])

    staging = tmp_path / "staging"
    staging.mkdir()
    joblib.dump(pipeline, staging / "pipeline.joblib")

    monkeypatch.setattr(evaluate_mod, "MODELS_STAGING_DIR", staging)
    monkeypatch.setattr(evaluate_mod, "METRICS_DIR", tmp_path / "metrics")
    monkeypatch.setattr(
        evaluate_mod, "load_gold", lambda: (df.iloc[:0], df, df, {})
    )  # mesmo df em val e test só para o teste checar as duas chaves

    metrics = evaluate_mod.run()

    assert set(metrics) == {"val", "test"}
    assert 0.0 <= metrics["val"]["macro_f1"] <= 1.0
    assert 0.0 <= metrics["test"]["macro_f1"] <= 1.0
    assert (tmp_path / "metrics" / "eval_metrics.json").exists()


def test_validar_aprova_quando_macro_f1_do_val_atinge_o_piso():
    metrics = {"val": {"macro_f1": 0.60}, "test": {"macro_f1": 0.10}}
    assert evaluate_mod.validar(metrics, piso=0.50) == metrics


def test_validar_reprova_quando_macro_f1_do_val_fica_abaixo_do_piso():
    metrics = {"val": {"macro_f1": 0.40}, "test": {"macro_f1": 0.90}}
    with pytest.raises(ValueError, match="não aprovado"):
        evaluate_mod.validar(metrics, piso=0.50)


def test_validar_decide_pelo_val_e_ignora_o_test():
    """O test pode estar ótimo ou péssimo — quem decide a promoção é sempre o val."""
    metrics_val_ruim = {"val": {"macro_f1": 0.10}, "test": {"macro_f1": 0.95}}
    with pytest.raises(ValueError, match="não aprovado"):
        evaluate_mod.validar(metrics_val_ruim, piso=0.50)


def test_validar_usa_o_piso_das_settings_quando_nao_informado(monkeypatch):
    monkeypatch.setattr(evaluate_mod.settings, "min_macro_f1", 0.99)
    with pytest.raises(ValueError, match=r"0\.9900"):
        evaluate_mod.validar({"val": {"macro_f1": 0.60}, "test": {"macro_f1": 0.60}})


# ── registry.py ───────────────────────────────────────────────────────


def test_promover_copia_apenas_os_artefatos_que_existem_em_staging(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    current = tmp_path / "current"
    staging.mkdir()
    (staging / "pipeline.joblib").write_bytes(b"fake-model")
    monkeypatch.setattr(registry_mod, "MODELS_STAGING_DIR", staging)
    monkeypatch.setattr(registry_mod, "MODELS_CURRENT_DIR", current)

    promovidos = registry_mod.promover(LocalModelStore(current))

    assert promovidos == ["pipeline.joblib"]
    assert (current / "pipeline.joblib").exists()
    assert not (current / "pipeline.onnx").exists()


def test_promover_falha_quando_staging_nao_tem_nenhum_artefato(tmp_path, monkeypatch):
    staging = tmp_path / "staging"
    current = tmp_path / "current"
    staging.mkdir()
    monkeypatch.setattr(registry_mod, "MODELS_STAGING_DIR", staging)
    monkeypatch.setattr(registry_mod, "MODELS_CURRENT_DIR", current)

    with pytest.raises(FileNotFoundError):
        registry_mod.promover(LocalModelStore(current))
