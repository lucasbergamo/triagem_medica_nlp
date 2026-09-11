"""Avaliação do modelo staged e gate de qualidade.

Macro-F1 é a métrica principal — classes desbalanceadas, e errar "urgente" custa mais do que
errar "normal" (ver docs/dataset.md). O gate em `validar()` é o que transforma a DAG do
Airflow em pipeline de retreino de verdade: um modelo abaixo do piso nunca é promovido.

O gate decide com as métricas do **val**, nunca do **test**. Num pipeline de retreino a
decisão de promover roda a cada execução — se decidisse olhando o test, o test deixaria de
ser uma estimativa imparcial do desempenho (a mesma partição vira, na prática, parte do
critério de seleção do modelo). `run()` calcula os dois blocos; `validar()` só enxerga "val".
O número reportado como resultado final (model card, README) é o de "test", medido depois da
decisão de promoção, nunca usado para tomá-la.
"""

import json
import sys

import pandas as pd
from joblib import load
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from src.data.features import load_gold
from src.utils.config import METRICS_DIR, MODELS_STAGING_DIR, settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

NOME_ARTEFATO_PRINCIPAL = "pipeline.joblib"
NOME_METRICAS = "eval_metrics.json"


def _carregar_staging(nome_arquivo: str = NOME_ARTEFATO_PRINCIPAL):
    path = MODELS_STAGING_DIR / nome_arquivo
    return load(path)


def avaliar(pipeline, x: pd.Series, y: pd.Series) -> dict:
    """Macro-F1, acurácia, relatório de precision/recall por classe e matriz de confusão.

    Genérica quanto à partição — chamada uma vez para "val" e uma vez para "test" em
    `run()`, nunca sabe qual das duas está calculando.
    """
    y_pred = pipeline.predict(x)
    labels = sorted(y.unique())

    return {
        "macro_f1": round(float(f1_score(y, y_pred, average="macro")), 4),
        "acuracia": round(float(accuracy_score(y, y_pred)), 4),
        "classification_report": classification_report(
            y, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y, y_pred, labels=labels).tolist(),
        "labels": labels,
        "n_amostras": len(y),
    }


def _salvar_metrics(metrics: dict) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / NOME_METRICAS
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("metrics_salvas", path=str(path))


def validar(metrics: dict, piso: float | None = None) -> dict:
    """Gate de qualidade: levanta `ValueError` com todas as falhas se macro-F1 (val) < piso.

    Decide com `metrics["val"]`, nunca com `metrics["test"]` — ver o porquê no docstring do
    módulo. Mesma função chamada por `make eval` e pela task `avaliacao` da DAG — a regra
    mora aqui uma única vez. Loga o resultado sempre, aprovado ou não; nunca falha em
    silêncio.
    """
    piso = piso if piso is not None else settings.min_macro_f1
    macro_f1_val = metrics["val"]["macro_f1"]
    falhas = []

    if macro_f1_val < piso:
        falhas.append(f"✗ macro-F1 (val) {macro_f1_val:.4f} abaixo do piso {piso:.4f}")

    logger.info(
        "avaliacao_gate",
        macro_f1_val=macro_f1_val,
        piso=piso,
        aprovado=not falhas,
    )

    if falhas:
        raise ValueError(
            "Modelo não aprovado — artefatos NÃO promovidos.\n  " + "\n  ".join(falhas)
        )

    return metrics


def run() -> dict:
    """Avalia o modelo staged no val e no test, e grava `metrics/eval_metrics.json`.

    Não aplica o gate — isso é `validar()`, que só olha o bloco "val" do retorno. O bloco
    "test" é a estimativa imparcial do desempenho final: calculada aqui, mas não usada para
    decidir nada.
    """
    pipeline = _carregar_staging()
    _train, val, test, _metadata = load_gold()

    metrics = {
        "val": avaliar(pipeline, val["texto"], val["urgencia"]),
        "test": avaliar(pipeline, test["texto"], test["urgencia"]),
    }
    _salvar_metrics(metrics)
    return metrics


def main() -> None:
    metrics = run()
    try:
        validar(metrics)
    except ValueError as exc:
        logger.error("avaliacao_reprovada", erro=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
