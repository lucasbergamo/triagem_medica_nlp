"""Testes de `GET /metrics` e da instrumentação — verificam que as 8 séries de observabilidade
aparecem na exposição e que uma predição/erro real incrementa a série certa. Os coletores do
`prometheus_client` vivem no registry global do processo (o mesmo
padrão da aula 04.02) e persistem entre testes do mesmo arquivo — por isso o valor **absoluto**
de uma série nunca é confiável, só o delta antes/depois de uma ação. `_valor_serie` lê o valor
numérico de uma linha específica da exposição (nunca conta ocorrências de substring: cada
Counter grava duas linhas com os mesmos labels — a série `_total` e a série auxiliar
`_created` do OpenMetrics — e contar substring dobraria qualquer incremento).
"""

import inspect
import re

from src.api import main, middleware


def _valor_serie(corpo: str, prefixo_com_labels: str) -> float:
    """`prefixo_com_labels` é `nome_da_serie{labels}` exato, ex.:
    `ml_requests_total{endpoint="/predict",metodo="POST",status_code="200"}`. Devolve 0.0
    quando a série ainda não foi observada nenhuma vez (linha ainda não existe)."""
    padrao = re.escape(prefixo_com_labels) + r" ([0-9.eE+-]+)"
    encontrado = re.search(padrao, corpo)
    return float(encontrado.group(1)) if encontrado else 0.0


def _soma_predicoes_sucesso(corpo: str, endpoint: str) -> float:
    """Soma `ml_predictions_total{endpoint=..., status="sucesso"}` por todas as classes —
    o teste do batch não sabe de antemão qual urgência o modelo sintético vai prever."""
    padrao = (
        rf'ml_predictions_total\{{endpoint="{re.escape(endpoint)}",'
        r'predicted_class="[^"]+",status="sucesso"\} ([0-9.eE+-]+)'
    )
    return sum(float(v) for v in re.findall(padrao, corpo))


def test_metrics_endpoint_devolve_content_type_prometheus(client):
    resp = client.get("/metrics")

    assert resp.status_code == 200
    assert "text/plain" in resp.headers["content-type"]


def test_metrics_endpoint_expoe_as_oito_series(client):
    corpo = client.get("/metrics").text

    # As 4 idênticas à aula 04.02
    assert "ml_predictions_total" in corpo
    assert "ml_prediction_latency_seconds" in corpo
    assert "ml_prediction_confidence" in corpo
    assert "ml_active_requests" in corpo
    # As 4 extensões
    assert "ml_inference_duration_seconds" in corpo
    assert "ml_requests_total" in corpo
    assert "ml_errors_total" in corpo
    assert "ml_model_info" in corpo


def test_metrics_endpoint_nao_e_instrumentado_pelo_middleware(client):
    """O próprio scrape não deve inflar `ml_requests_total` — regra explícita do middleware:
    nenhuma linha de `ml_requests_total` pode trazer `endpoint="/metrics"`."""
    client.get("/metrics")
    corpo = client.get("/metrics").text

    assert 'ml_requests_total{endpoint="/metrics"' not in corpo


def test_predict_incrementa_predictions_total_e_confidence(client):
    laudo = (
        "Paciente do sexo masculino, 58 anos, apresenta dor precordial em aperto com "
        "irradiação para o braço esquerdo, sudorese e dispneia associada."
    )
    serie = 'ml_predictions_total{{endpoint="/predict",predicted_class="{}",status="sucesso"}}'

    resp = client.post("/predict", json={"texto": laudo})
    urgencia = resp.json()["urgencia"]
    antes = _valor_serie(client.get("/metrics").text, serie.format(urgencia))

    client.post("/predict", json={"texto": laudo})
    depois_corpo = client.get("/metrics").text
    depois = _valor_serie(depois_corpo, serie.format(urgencia))

    assert depois == antes + 1
    assert "ml_prediction_confidence_bucket" in depois_corpo


def test_predict_incrementa_inference_duration_com_label_de_backend(client):
    serie = 'ml_inference_duration_seconds_count{backend="sklearn"}'
    antes = _valor_serie(client.get("/metrics").text, serie)

    client.post(
        "/predict",
        json={
            "texto": (
                "Paciente relata dor abdominal difusa, náuseas e episódios de vômito nas "
                "últimas 12 horas, sem febre associada."
            )
        },
    )

    depois = _valor_serie(client.get("/metrics").text, serie)
    assert depois == antes + 1


def test_predict_batch_incrementa_requests_total_uma_vez_por_requisicao_http(client):
    """`ml_requests_total` é métrica de **requisição** — um lote de 2 laudos conta como 1
    chamada HTTP, mesmo gerando 2 observações em `ml_predictions_total`."""
    serie_requests = 'ml_requests_total{endpoint="/predict/batch",metodo="POST",status_code="200"}'
    antes_requests = _valor_serie(client.get("/metrics").text, serie_requests)
    antes_predictions = _soma_predicoes_sucesso(client.get("/metrics").text, "/predict/batch")

    client.post(
        "/predict/batch",
        json={
            "laudos": [
                {"texto": "Paciente com quadro clínico estável, sem queixas agudas no momento."},
                {"texto": "Paciente com quadro clínico estável, sem queixas agudas no momento."},
            ]
        },
    )

    corpo_depois = client.get("/metrics").text
    depois_requests = _valor_serie(corpo_depois, serie_requests)
    depois_predictions = _soma_predicoes_sucesso(corpo_depois, "/predict/batch")

    assert depois_requests == antes_requests + 1
    assert depois_predictions == antes_predictions + 2


def test_texto_curto_incrementa_requests_total_com_status_422(client):
    serie = 'ml_requests_total{endpoint="/predict",metodo="POST",status_code="422"}'
    antes = _valor_serie(client.get("/metrics").text, serie)

    client.post("/predict", json={"texto": "curto"})

    depois = _valor_serie(client.get("/metrics").text, serie)
    assert depois == antes + 1


def test_model_info_reflete_backend_ativo(client):
    corpo = client.get("/metrics").text
    assert 'ml_model_info_info{backend="sklearn"' in corpo


def test_middleware_e_sincrono_mas_endpoints_de_inferencia_continuam(client):
    """O middleware ASGI é `async def` por exigência do framework (é o que ele intercepta),
    mas isso não reintroduz o problema que a regra `def`-only evita: `/predict` e
    `/predict/batch` continuam despachados no threadpool."""
    assert inspect.iscoroutinefunction(middleware.track_request_metrics)
    assert not inspect.iscoroutinefunction(main.predict)
    assert not inspect.iscoroutinefunction(main.predict_batch)
