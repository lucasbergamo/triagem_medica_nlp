"""Download e carregamento do Medical Abstracts TC Corpus (bronze layer).

Fonte pública, sem login/API key/Kaggle — `httpx.get` direto no GitHub raw. Idempotente:
se o CSV já existe em `data/bronze/`, não baixa de novo, mas
sempre valida schema e contagem — inclusive nos arquivos já presentes de um clone.
"""

from pathlib import Path

import httpx
import pandas as pd

from src.utils.config import DATA_BRONZE_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

BASE_URL = "https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/"

# Contagens e colunas verificadas em 01/09/2026 — se mudarem, é sinal de que a fonte
# foi alterada na origem.
ARQUIVOS_ESPERADOS: dict[str, int] = {
    "medical_tc_train.csv": 11_550,
    "medical_tc_test.csv": 2_888,
    "medical_tc_labels.csv": 5,
}

COLUNAS_ESPERADAS: dict[str, list[str]] = {
    "medical_tc_train.csv": ["condition_label", "medical_abstract"],
    "medical_tc_test.csv": ["condition_label", "medical_abstract"],
    "medical_tc_labels.csv": ["condition_label", "condition_name"],
}


def download_bronze(dest_dir: Path = DATA_BRONZE_DIR) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)

    for filename, n_esperado in ARQUIVOS_ESPERADOS.items():
        dest_path = dest_dir / filename

        if dest_path.exists():
            logger.info("bronze_arquivo_ja_existe", arquivo=filename)
        else:
            url = BASE_URL + filename
            logger.info("baixando_bronze", arquivo=filename, url=url)
            response = httpx.get(url, timeout=60)
            response.raise_for_status()
            dest_path.write_bytes(response.content)

        _validar_arquivo(dest_path, filename, n_esperado)

    logger.info("bronze_download_finalizado", dest=str(dest_dir))


def _validar_arquivo(path: Path, filename: str, n_esperado: int) -> None:
    df = pd.read_csv(path)

    colunas = list(df.columns)
    if colunas != COLUNAS_ESPERADAS[filename]:
        raise ValueError(f"{filename}: colunas {colunas}, esperado {COLUNAS_ESPERADAS[filename]}")

    if len(df) != n_esperado:
        raise ValueError(f"{filename}: {len(df)} linhas, esperado {n_esperado}")


def load_bronze() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Lê os 3 CSVs já baixados. Não faz download — chame `download_bronze()` antes."""
    train = pd.read_csv(DATA_BRONZE_DIR / "medical_tc_train.csv")
    test = pd.read_csv(DATA_BRONZE_DIR / "medical_tc_test.csv")
    labels = pd.read_csv(DATA_BRONZE_DIR / "medical_tc_labels.csv")
    logger.info("bronze_carregado", train=len(train), test=len(test), labels=len(labels))
    return train, test, labels


def main() -> None:
    download_bronze()


if __name__ == "__main__":
    main()
