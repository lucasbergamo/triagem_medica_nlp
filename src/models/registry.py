"""Promoção de modelo — a única escrita em `models/current/` de todo o projeto.

Copia os artefatos aprovados de `models/staging/` para `models/current/` (o diretório que a
API lê). Só deve rodar depois que `src.models.evaluate.validar()` aprovou o modelo — na DAG
do Airflow, a task `promocao` só executa se a task `avaliacao` teve sucesso.
"""

from src.models.store import LocalModelStore
from src.utils.config import MODELS_CURRENT_DIR, MODELS_STAGING_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Artefatos que a API pode servir — um por backend de inferência. Um nome ausente em
# staging é ignorado sem erro: nem todo treino gera os três (ex.: antes da exportação ONNX
# existir, só há `pipeline.joblib`).
ARTEFATOS_SERVIDOS = ["pipeline.joblib", "pipeline.onnx", "pipeline.int8.onnx"]

# Metadado de treino (batch_id, data_treino, macro_f1_val) — não é um modelo servido, mas
# `/model/info` lê `modelo_versao` dele, então precisa viajar junto.
NOME_METADADOS = "model_meta.json"


def promover(store: LocalModelStore | None = None) -> list[str]:
    store = store or LocalModelStore(MODELS_CURRENT_DIR)

    promovidos = []
    for nome in ARTEFATOS_SERVIDOS:
        origem = MODELS_STAGING_DIR / nome
        if origem.exists():
            store.copiar(origem, nome)
            promovidos.append(nome)

    if not promovidos:
        raise FileNotFoundError(
            f"Nenhum artefato encontrado em {MODELS_STAGING_DIR} — treino falhou ou não rodou."
        )

    origem_metadados = MODELS_STAGING_DIR / NOME_METADADOS
    if origem_metadados.exists():
        store.copiar(origem_metadados, NOME_METADADOS)
        promovidos.append(NOME_METADADOS)

    logger.info("modelo_promovido", artefatos=promovidos)
    return promovidos


def main() -> None:
    promover()


if __name__ == "__main__":
    main()
