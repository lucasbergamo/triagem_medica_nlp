"""Silver → Gold: split estratificado 70/15/15 por urgência, com hash de rastreabilidade.

Saída: `train/val/test.parquet` + `metadata.parquet` com `n_classes`, `classes`, contagens
e hash do split — o suficiente para detectar se os dados mudaram entre execuções sem
precisar abrir os parquets.
"""

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.preprocess import load_silver
from src.utils.config import DATA_GOLD_DIR, settings
from src.utils.logger import get_logger
from src.utils.reproducibility import set_global_seed

logger = get_logger(__name__)

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def split_estratificado(
    df: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """70/15/15 estratificado por `urgencia`, em dois cortes sucessivos do `train_test_split`."""
    train, resto = train_test_split(
        df, train_size=TRAIN_RATIO, stratify=df["urgencia"], random_state=seed
    )
    val_fracao_do_resto = VAL_RATIO / (VAL_RATIO + TEST_RATIO)
    val, test = train_test_split(
        resto, train_size=val_fracao_do_resto, stratify=resto["urgencia"], random_state=seed
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def _hash_split(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> str:
    """Hash determinístico do conteúdo do split.

    Inclui texto + urgência + qual split cada linha caiu — não só o texto. Hashear só o
    texto (versão anterior) não pegaria uma mudança na regra de resolução de urgência
    nem um texto migrando de split, já que o conjunto de textos em si pode continuar
    idêntico entre duas execuções com regras diferentes.
    """
    linhas = [
        f"{nome}|{texto}|{urgencia}"
        for nome, df in (("train", train), ("val", val), ("test", test))
        for texto, urgencia in zip(df["texto"], df["urgencia"], strict=True)
    ]
    assinatura = "\n".join(sorted(linhas))
    return hashlib.sha256(assinatura.encode("utf-8")).hexdigest()[:16]


def montar_metadata(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    classes = sorted(train["urgencia"].unique())
    return pd.DataFrame(
        [
            {
                "n_classes": len(classes),
                "classes": ",".join(classes),
                "n_train": len(train),
                "n_val": len(val),
                "n_test": len(test),
                "split_hash": _hash_split(train, val, test),
            }
        ]
    )


def save_gold(
    train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame, metadata: pd.DataFrame
) -> None:
    DATA_GOLD_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(DATA_GOLD_DIR / "train.parquet", index=False)
    val.to_parquet(DATA_GOLD_DIR / "val.parquet", index=False)
    test.to_parquet(DATA_GOLD_DIR / "test.parquet", index=False)
    metadata.to_parquet(DATA_GOLD_DIR / "metadata.parquet", index=False)
    logger.info(
        "gold_saved",
        train=len(train),
        val=len(val),
        test=len(test),
        split_hash=metadata.iloc[0]["split_hash"],
    )


def load_gold() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    train = pd.read_parquet(DATA_GOLD_DIR / "train.parquet")
    val = pd.read_parquet(DATA_GOLD_DIR / "val.parquet")
    test = pd.read_parquet(DATA_GOLD_DIR / "test.parquet")
    metadata = pd.read_parquet(DATA_GOLD_DIR / "metadata.parquet").iloc[0].to_dict()
    return train, val, test, metadata


def main() -> None:
    set_global_seed()
    df = load_silver()
    train, val, test = split_estratificado(df, seed=settings.seed)
    metadata = montar_metadata(train, val, test)
    save_gold(train, val, test, metadata)


if __name__ == "__main__":
    main()
