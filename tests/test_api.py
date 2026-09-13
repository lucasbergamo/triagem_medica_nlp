"""Testes da API HTTP — TestClient contra um modelo sintético treinado na hora, nunca contra
`models/current/pipeline.joblib`. Estes testes verificam contrato (status, forma do JSON,
validação), não qualidade de modelo — não podem depender de `make train` ter rodado antes,
porque o CI clona o repo sem nenhum artefato. As fixtures `client` e
`artefato_modelo_sintetico` moram em `tests/conftest.py` — `test_metrics.py` reusa as mesmas.
"""

import inspect

from src.api import main

LAUDO_VALIDO = (
    "Paciente do sexo masculino, 58 anos, apresenta dor precordial em aperto com irradiação "
    "para o braço esquerdo, sudorese e dispneia associada, com início há trinta minutos."
)


def test_predict_caminho_feliz_devolve_contrato_completo(client):
    resp = client.post("/predict", json={"texto": LAUDO_VALIDO})

    assert resp.status_code == 200
    corpo = resp.json()
    assert corpo["urgencia"] in {"urgente", "atencao", "normal"}
    assert 0.0 <= corpo["confianca"] <= 1.0
    assert set(corpo["probabilidades"]) == {"urgente", "atencao", "normal"}
    assert corpo["latencia_inferencia_ms"] >= 0
    assert corpo["backend"] == "sklearn"
    assert corpo["modelo_versao"]


def test_predict_texto_curto_devolve_422_com_mensagem_em_portugues(client):
    resp = client.post("/predict", json={"texto": "muito curto"})

    assert resp.status_code == 422
    assert "caracteres" in resp.json()["detail"].lower()


def test_predict_texto_ausente_devolve_422(client):
    resp = client.post("/predict", json={})

    assert resp.status_code == 422
    assert "obrigatório" in resp.json()["detail"].lower()


def test_predict_batch_devolve_um_resultado_por_laudo(client):
    resp = client.post(
        "/predict/batch",
        json={"laudos": [{"texto": LAUDO_VALIDO}, {"texto": LAUDO_VALIDO}]},
    )

    assert resp.status_code == 200
    resultados = resp.json()["resultados"]
    assert len(resultados) == 2
    assert all(r["backend"] == "sklearn" for r in resultados)


def test_predict_batch_acima_do_limite_devolve_422(client):
    laudos = [{"texto": LAUDO_VALIDO}] * 101
    resp = client.post("/predict/batch", json={"laudos": laudos})

    assert resp.status_code == 422


def test_predict_batch_vazio_devolve_422(client):
    resp = client.post("/predict/batch", json={"laudos": []})

    assert resp.status_code == 422


def test_health_responde_ok_e_nao_depende_do_modelo(client):
    resp = client.get("/health")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_confirma_modelo_carregado(client):
    resp = client.get("/ready")

    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_model_info_devolve_backend_versao_e_classes(client):
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
