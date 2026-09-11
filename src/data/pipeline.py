"""Orquestra o pipeline completo de dados: bronze → silver → gold.

Delega para o `main()` de cada camada em vez de duplicar lógica — é o mesmo caminho que a
task `preparacao` da DAG do Airflow chama (B6), só que num comando só para uso local/`make
data` (mesmo padrão do TC2).
"""

from src.data.features import main as run_features
from src.data.load import main as run_load
from src.data.preprocess import main as run_preprocess
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run() -> None:
    logger.info("pipeline_started")
    run_load()
    run_preprocess()
    run_features()
    logger.info("pipeline_finished")


if __name__ == "__main__":
    run()
