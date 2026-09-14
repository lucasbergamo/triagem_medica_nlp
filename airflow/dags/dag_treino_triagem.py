"""DAG de retreino do classificador de urgência.

Cada task é uma casca fina sobre um módulo de `src/` (ou `scripts/`, no caso do benchmark) —
a mesma função que `make train`, `make eval` etc. chamam localmente e que o CI testa. A DAG
não contém lógica de negócio: só encadeia o que já existe.

**Ordem normativa (regra adotada, ver ADR-0005):** `exportacao_onnx → promocao →
benchmark_latencia` — promoção antes do benchmark. `scripts.benchmark_latency` sempre lê os
artefatos de `models/current/`; medir antes da promoção mediria o modelo que está saindo, não
o que este run acabou de treinar. O benchmark é informativo e não gateia nada — quem barra um
modelo abaixo do piso é a task `avaliacao`, antes da exportação.

Staging → produção: `treino` e `exportacao_onnx` escrevem só em `models/staging/`. `promocao`
é a única escrita em `models/current/` — o diretório que a API lê — e só roda se `avaliacao`
aprovou (trigger rule padrão `all_success`: task anterior falhou, a seguinte não executa).
"""

from __future__ import annotations

from datetime import timedelta

import pendulum
from airflow import DAG
from airflow.decorators import task

LOCAL_TZ = pendulum.timezone("America/Sao_Paulo")

default_args = {
    "owner": "ml-engineering",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "execution_timeout": timedelta(minutes=60),
}

with DAG(
    dag_id="treino_triagem",
    start_date=pendulum.datetime(2026, 9, 1, tz=LOCAL_TZ),
    schedule="@weekly",
    catchup=False,
    max_active_runs=1,
    tags=["tc3", "triagem", "ml", "retreino"],
    default_args=default_args,
    doc_md="""
## treino_triagem

Pipeline de retreino do classificador de urgência de laudos médicos: dados → treino →
avaliação (gate) → exportação ONNX → promoção → benchmark de latência.

**Trigger:** semanal (`@weekly`) ou manual.

**Parâmetros opcionais (`dag_run.conf`):** `min_macro_f1` — sobrescreve o piso do gate de
qualidade (default: `MIN_MACRO_F1` das settings, 0,50).

**Fluxo:**
1. `prep_execution` — gera o `batch_id` (timestamp) e lê `dag_run.conf`.
2. `ingestao` — baixa/valida o Medical Abstracts TC Corpus em `data/bronze/`.
3. `preparacao` — bronze → silver → gold (dedup, mapa de urgência, split 70/15/15).
4. `treino` — treina TF-IDF + LogisticRegression, grava em `models/staging/`.
5. `avaliacao` — macro-F1 no val; **gate**: abaixo do piso, a run falha aqui e nada é promovido.
6. `exportacao_onnx` — exporta o classificador para ONNX (fp32 + int8), valida contra o sklearn.
7. `promocao` — copia `models/staging/` → `models/current/` (única escrita nesse diretório).
8. `benchmark_latencia` — mede os 3 backends já promovidos, grava `metrics/latency_benchmark.json`.

**Artefatos:** `models/current/{pipeline.joblib,pipeline.onnx,pipeline.int8.onnx,model_meta.json}`.

**Critério de aprovação:** macro-F1 (val) ≥ `MIN_MACRO_F1`. Reprovado → nenhum artefato é promovido.

**Correspondência com a DAG de referência da disciplina** (`ml_dag.py`,
`prepare → train → evaluate → deploy`): `ingestao`+`preparacao` ⊂ `prepare`, `treino` = `train`,
`avaliacao` = `evaluate` (mesmo padrão de gate — `ValueError` quando a métrica não atinge o
piso), `exportacao_onnx`+`promocao` ⊂ `deploy`. `benchmark_latencia` é extensão deste projeto,
sem equivalente na referência.
""",
) as dag:

    @task
    def prep_execution(**context) -> dict:
        """Define `batch_id` e lê os parâmetros opcionais do run."""
        dag_run = context.get("dag_run")
        conf = (dag_run.conf or {}) if dag_run else {}
        batch_id = pendulum.now(LOCAL_TZ).strftime("%Y%m%d_%H%M%S")
        return {"batch_id": batch_id, "min_macro_f1": conf.get("min_macro_f1")}

    @task
    def ingestao(run_info: dict) -> dict:
        """Baixa e valida os 3 CSVs do corpus em `data/bronze/` (idempotente)."""
        from src.data.load import main as baixar_bronze

        baixar_bronze()
        return run_info

    @task
    def preparacao(run_info: dict) -> dict:
        """Bronze → silver → gold: dedup, mapa de urgência, split estratificado 70/15/15."""
        from src.data.pipeline import run as executar_pipeline_dados

        executar_pipeline_dados()
        return run_info

    @task
    def treino(run_info: dict) -> dict:
        """Treina o pipeline e grava em `models/staging/`. Devolve as métricas de treino."""
        from src.models.train import run as treinar

        return treinar(batch_id=run_info["batch_id"])

    @task
    def avaliacao(run_info: dict, metrics_treino: dict) -> dict:
        """Gate: falha a task se o modelo staged não atinge o piso de macro-F1 (val).

        `metrics_treino` só existe como argumento para amarrar a dependência em `treino` —
        a avaliação lê o modelo staged do disco, não recebe métricas de treino por XCom.
        Critério: macro-F1 ≥ `MIN_MACRO_F1` (classes desbalanceadas; errar "urgente" custa
        mais). `piso` vem de `dag_run.conf["min_macro_f1"]` quando informado, senão do default
        das settings.
        """
        from src.models.evaluate import run as avaliar_staging
        from src.models.evaluate import validar

        metrics = avaliar_staging()
        return validar(metrics, piso=run_info.get("min_macro_f1"))

    @task
    def exportacao_onnx(metrics_avaliacao: dict) -> dict:
        """Exporta o classificador staged para ONNX (fp32 + int8), valida contra o sklearn.

        Só roda se `avaliacao` aprovou — `metrics_avaliacao` é o retorno de `validar()`,
        que levanta antes de chegar aqui se o modelo não passou no gate.
        """
        from src.models.export_onnx import run as exportar_onnx

        return exportar_onnx()

    @task
    def promocao(resultado_exportacao: dict) -> list[str]:
        """Copia `models/staging/` → `models/current/` — única escrita nesse diretório."""
        from src.models.registry import promover

        return promover()

    @task
    def benchmark_latencia(promovidos: list[str]) -> dict:
        """Mede os 3 backends já promovidos e grava `metrics/latency_benchmark.json`.

        Informativo — não gateia a promoção (já concluída na task anterior). Roda depois de
        `promocao` de propósito: `scripts.benchmark_latency` sempre lê `models/current/`, e
        medir antes mediria o modelo que está saindo, não o que este run treinou.
        """
        from scripts.benchmark_latency import medir_comparativo, salvar_comparativo

        resultado = medir_comparativo()
        salvar_comparativo(resultado)
        return resultado["predict_completo"]

    # Encadeamento pelo fluxo de retorno: cada task só roda se a anterior teve sucesso.
    run_info = prep_execution()
    resultado_ingestao = ingestao(run_info)
    resultado_preparacao = preparacao(resultado_ingestao)
    metrics_treino = treino(resultado_preparacao)
    metrics_avaliacao = avaliacao(run_info, metrics_treino)
    resultado_exportacao = exportacao_onnx(metrics_avaliacao)
    artefatos_promovidos = promocao(resultado_exportacao)
    benchmark_latencia(artefatos_promovidos)
