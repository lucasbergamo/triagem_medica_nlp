"""Treino do classificador de urgência — TF-IDF + LogisticRegression.

Pipeline de dois estágios escolhido porque converte a inferência para um único GEMM: é o
caso em que a exportação ONNX ganha de verdade (Random Forest, o exemplo do enunciado,
serializa uma árvore por estimador e pode ficar mais lento em ONNX do que em sklearn — ver
docs/model_card.md). O RandomForest treina de qualquer forma, como baseline de comparação
(flag `--baseline`), para o model card mostrar que a escolha foi medida, não assumida.
"""

import argparse
import shutil
import time
from datetime import datetime
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.data.features import load_gold
from src.utils.config import MODELS_STAGING_DIR, settings
from src.utils.logger import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)

NOME_ARTEFATO_PRINCIPAL = "pipeline.joblib"
NOME_ARTEFATO_BASELINE = "pipeline_rf_baseline.joblib"


def _tfidf() -> TfidfVectorizer:
    """Vetorizador compartilhado pelo modelo principal e pelo baseline — só o classificador
    muda entre os dois, para a comparação no model card ser justa."""
    return TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=50_000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )


def build_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", _tfidf()),
            (
                "clf",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, C=1.0, random_state=seed
                ),
            ),
        ]
    )


def build_baseline_pipeline(seed: int) -> Pipeline:
    return Pipeline(
        [
            ("tfidf", _tfidf()),
            (
                "clf",
                RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
            ),
        ]
    )


def _treinar(pipeline: Pipeline, x_train, y_train) -> float:
    inicio = time.perf_counter()
    pipeline.fit(x_train, y_train)
    return time.perf_counter() - inicio


def _salvar_staging(pipeline: Pipeline, nome_arquivo: str) -> Path:
    MODELS_STAGING_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_STAGING_DIR / nome_arquivo
    joblib.dump(pipeline, path)
    return path


def _limpar_staging() -> None:
    """Esvazia `models/staging/` no início de cada run — cada batch começa limpo.

    Sem isso, um retreino que só regrava o classificador principal deixaria artefatos de
    um batch anterior (ex.: um `pipeline.onnx` já exportado) sobrevivendo ao lado do
    `pipeline.joblib` novo, e a promoção levaria os dois para `models/current/` como se
    fossem do mesmo treino — as APIs `sklearn` e `onnx` passariam a servir modelos
    diferentes sem nenhum erro visível.
    """
    if MODELS_STAGING_DIR.exists():
        shutil.rmtree(MODELS_STAGING_DIR)
    MODELS_STAGING_DIR.mkdir(parents=True, exist_ok=True)


def run(batch_id: str | None = None, baseline: bool = False) -> dict:
    """Treina o pipeline principal (e opcionalmente o baseline RF) e salva em `models/staging/`.

    Devolve métricas do treino (tempo, tamanho do artefato) — não métricas de qualidade do
    modelo, que vivem em `src.models.evaluate`. É a função que a task `treino` da DAG do
    Airflow chama, passando o `batch_id` gerado por `prep_execution`.
    """
    set_global_seed()
    seed = settings.seed
    batch_id = batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    _limpar_staging()

    train, _val, _test, _metadata = load_gold()
    x_train, y_train = train["texto"], train["urgencia"]

    pipeline = build_pipeline(seed)
    tempo_treino = _treinar(pipeline, x_train, y_train)
    path = _salvar_staging(pipeline, NOME_ARTEFATO_PRINCIPAL)

    resultado = {
        "batch_id": batch_id,
        "modelo": "logreg_tfidf",
        "tempo_treino_s": round(tempo_treino, 2),
        "tamanho_artefato_kb": round(path.stat().st_size / 1024, 1),
        "n_train": len(x_train),
    }
    logger.info("treino_concluido", **resultado)

    if baseline:
        pipeline_baseline = build_baseline_pipeline(seed)
        tempo_baseline = _treinar(pipeline_baseline, x_train, y_train)
        path_baseline = _salvar_staging(pipeline_baseline, NOME_ARTEFATO_BASELINE)
        resultado["baseline"] = {
            "modelo": "random_forest_tfidf",
            "tempo_treino_s": round(tempo_baseline, 2),
            "tamanho_artefato_kb": round(path_baseline.stat().st_size / 1024, 1),
        }
        logger.info("treino_baseline_concluido", **resultado["baseline"])

    return resultado


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina o classificador de urgência.")
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="Também treina o RandomForest de comparação (usado no model card).",
    )
    args = parser.parse_args()
    run(baseline=args.baseline)


if __name__ == "__main__":
    main()
