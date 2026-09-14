"""Valida a DAG `treino_triagem` — roda dentro da imagem de `airflow/` (`make dag-test`), não
no pytest do Poetry: o Airflow não entra nas dependências do projeto (conflitaria com as da
API). `pytest.importorskip` abaixo faz este arquivo se pular sozinho quando o pacote
`apache-airflow` não está instalado, para não quebrar `make test`/o job `test` do CI.

`importorskip("airflow.models")`, não `"airflow"`: a pasta `airflow/` deste projeto (o
diretório, não o pacote PyPI) fica na raiz e o `pythonpath = ["."]` do pytest a expõe como
namespace package — `import airflow` "funciona" mesmo sem o pacote real instalado, resolvendo
para esse diretório. `airflow.models` só existe no pacote de verdade, então é o que precisa
ser testado para o skip funcionar fora da imagem do Airflow.
"""

from itertools import pairwise

import pytest

pytest.importorskip("airflow.models")

from airflow.models import DagBag

DAG_ID = "treino_triagem"

TASKS_NA_ORDEM = [
    "prep_execution",
    "ingestao",
    "preparacao",
    "treino",
    "avaliacao",
    "exportacao_onnx",
    "promocao",
    "benchmark_latencia",
]


@pytest.fixture(scope="module")
def dagbag() -> DagBag:
    return DagBag(dag_folder="/opt/airflow/dags", include_examples=False)


def test_dag_sem_erro_de_import(dagbag: DagBag) -> None:
    assert dagbag.import_errors == {}


def test_dag_existe(dagbag: DagBag) -> None:
    assert dagbag.get_dag(DAG_ID) is not None


def test_dag_tem_8_tasks(dagbag: DagBag) -> None:
    dag = dagbag.get_dag(DAG_ID)
    assert len(dag.tasks) == len(TASKS_NA_ORDEM)
    assert {t.task_id for t in dag.tasks} == set(TASKS_NA_ORDEM)


def test_dag_dependencias_sequenciais(dagbag: DagBag) -> None:
    """Cada task só tem a seguinte como downstream direta — cadeia linear de 8 passos."""
    dag = dagbag.get_dag(DAG_ID)
    for anterior, seguinte in pairwise(TASKS_NA_ORDEM):
        assert seguinte in dag.get_task(anterior).downstream_task_ids


def test_avaliacao_depende_de_prep_execution_e_treino(dagbag: DagBag) -> None:
    """`avaliacao` recebe `run_info` (para o piso do gate) e `metrics_treino` (ordenação)."""
    dag = dagbag.get_dag(DAG_ID)
    upstream = dag.get_task("avaliacao").upstream_task_ids
    assert upstream == {"prep_execution", "treino"}


def test_dag_configuracao(dagbag: DagBag) -> None:
    dag = dagbag.get_dag(DAG_ID)
    assert dag.catchup is False
    assert dag.max_active_runs == 1
    assert set(dag.tags) == {"tc3", "triagem", "ml", "retreino"}
