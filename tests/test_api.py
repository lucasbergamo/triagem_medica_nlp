"""Testes da API HTTP — TestClient real, sem mocks: o lifespan carrega o modelo já promovido
em models/current/pipeline.joblib (existe localmente desde `make promote`, no bloco anterior).
"""

import inspect

from fastapi.testclient import TestClient

from src.api import main
from src.api.main import app

LAUDO_VALIDO = (
    "Paciente do sexo masculino, 58 anos, apresenta dor precordial em aperto com irradiação "
    "para o braço esquerdo, sudorese e dispneia associada, com início há trinta minutos."
)


def test_predict_caminho_feliz_devolve_contrato_completo():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"texto": LAUDO_VALIDO})

    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["urgencia"] in {"urgente", "atencao", "normal"}
    assert 0.0 <= corpo["confianca"] <= 1.0
    assert set(corpo["probabilidades"]) == {"urgente", "atencao", "normal"}
    assert corpo["latencia_inferencia_ms"] >= 0
    assert corpo["backend"] == "sklearn"
    assert corpo["modelo_versao"]


def test_predict_texto_curto_devolve_422_com_mensagem_em_portugues():
    with TestClient(app) as client:
        resp = client.post("/predict", json={"texto": "muito curto"})

    assert resp.status_code == 422
    assert "caracteres" in resp.json()["detail"].lower()


def test_predict_texto_ausente_devolve_422():
    with TestClient(app) as client:
        resp = client.post("/predict", json={})

    assert resp.status_code == 422
    assert "obrigatório" in resp.json()["detail"].lower()


def test_predict_batch_devolve_um_resultado_por_laudo():
    with TestClient(app) as client:
        resp = client.post(
            "/predict/batch",
            json={"laudos": [{"texto": LAUDO_VALIDO}, {"texto": LAUDO_VALIDO}]},
        )

    assert resp.status_code == 200
    resultados = resp.json()["resultados"]
    assert len(resultados) == 2
    assert all(r["backend"] == "sklearn" for r in resultados)


def test_predict_batch_acima_do_limite_devolve_422():
    laudos = [{"texto": LAUDO_VALIDO}] * 101
    with TestClient(app) as client:
        resp = client.post("/predict/batch", json={"laudos": laudos})

    assert resp.status_code == 422


def test_predict_batch_vazio_devolve_422():
    with TestClient(app) as client:
        resp = client.post("/predict/batch", json={"laudos": []})

    assert resp.status_code == 422


def test_health_responde_ok_e_nao_depende_do_modelo():
    with TestClient(app) as client:
        resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_confirma_modelo_carregado():
    with TestClient(app) as client:
        resp = client.get("/ready")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_info_devolve_backend_versao_e_classes():
    with TestClient(app) as client:
        resp = client.get("/model/info")

    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["backend"] == "sklearn"
    assert set(corpo["classes"]) == {"urgente", "atencao", "normal"}
    assert corpo["modelo_versao"] == corpo["data_treino"]


def test_endpoints_de_inferencia_sao_sincronos():
    """Inferência é CPU-bound — `async def` bloquearia o event loop, então /predict e
    /predict/batch têm que continuar `def` para o FastAPI despachar via threadpool."""
    assert not inspect.iscoroutinefunction(main.predict)
    assert not inspect.iscoroutinefunction(main.predict_batch)
