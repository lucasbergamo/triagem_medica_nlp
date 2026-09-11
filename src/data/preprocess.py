"""Bronze → Silver: combina train+test, resolve multi-rótulo e aplica o mapa de urgência.

Saída: `laudos.parquet` com `texto`, `categoria_clinica`, `urgencia`, `n_caracteres`.
O split train/test original do corpus não é usado como split do
projeto — os dois são combinados aqui e o split 70/15/15 próprio é feito na camada gold
(`src/data/features.py`), com a seed global.

Achado ao implementar, não previsto no desenho original do pipeline: o corpus é
**multi-rótulo achatado em linhas** — 2.929 abstracts aparecem repetidos, cada cópia com um
`condition_label` diferente (mesmo texto sob mais de uma categoria clínica). Não é ruído de
anotação: é o formato do dataset. Resolver isso com um `drop_duplicates(keep="first")`
sub-triaria sistematicamente — descartaria uma cópia "urgente" em favor de uma cópia
"normal" só por causa da ordem de leitura do arquivo. Regra adotada em 11/09/2026: **a
maior urgência vence**: entre os rótulos do mesmo texto, fica o de `RANK_URGENCIA`
mais alto; empate (dois rótulos na mesma urgência) desempatado pelo menor
`condition_label`, o que torna o resultado independente da ordem das linhas no arquivo.
Justificativa em duas pontas — ver `docs/dataset.md`: a categoria específica vence a
categoria 5 (residual, "general pathological conditions"), e em triagem a dúvida escala,
já que errar "urgente" custa mais do que errar para baixo (é por isso que macro-F1, não
acurácia, é a métrica principal de avaliação deste projeto).
"""

import pandas as pd

from src.data.load import load_bronze
from src.data.urgency_map import CONDITION_NAMES, RANK_URGENCIA, mapear_urgencia
from src.utils.config import DATA_SILVER_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

MIN_CARACTERES = 50


def combinar_bronze(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Junta os dois splits originais do corpus num único conjunto de laudos."""
    df = pd.concat([train, test], ignore_index=True)
    return df.rename(columns={"medical_abstract": "texto"})


def _resolver_grupo(condition_labels: set[int]) -> int:
    """Entre os `condition_label` de um mesmo texto, escolhe o de maior urgência.

    Empate (dois rótulos mapeando para a mesma urgência) desempatado pelo menor
    `condition_label` — determinístico e independente da ordem de leitura do arquivo.
    """
    return max(
        condition_labels,
        key=lambda label: (RANK_URGENCIA[mapear_urgencia(label)], -label),
    )


def resolver_multirrotulo(df: pd.DataFrame) -> pd.DataFrame:
    """Colapsa cada texto repetido em uma linha, mantendo o `condition_label` vencedor."""
    labels_por_texto = df.groupby("texto")["condition_label"].apply(set)
    vencedor = labels_por_texto.apply(_resolver_grupo)
    return vencedor.rename("condition_label").reset_index()


def limpar(df: pd.DataFrame) -> pd.DataFrame:
    """Remove nulos, resolve multi-rótulo por texto e descarta laudos curtos."""
    n_inicial = len(df)

    df = df.dropna(subset=["texto", "condition_label"]).copy()
    df["condition_label"] = df["condition_label"].astype(int)
    n_apos_nulos = len(df)

    df = resolver_multirrotulo(df)
    n_apos_resolucao = len(df)

    df["n_caracteres"] = df["texto"].str.len()
    df = df[df["n_caracteres"] >= MIN_CARACTERES]
    n_final = len(df)

    logger.info(
        "silver_limpeza",
        linhas_iniciais=n_inicial,
        nulos_removidos=n_inicial - n_apos_nulos,
        linhas_colapsadas_multirrotulo=n_apos_nulos - n_apos_resolucao,
        curtas_removidas=n_apos_resolucao - n_final,
        linhas_finais=n_final,
    )
    return df


def aplicar_mapa(df: pd.DataFrame) -> pd.DataFrame:
    """Deriva `categoria_clinica` e `urgencia` a partir do `condition_label` já resolvido."""
    df = df.copy()
    df["categoria_clinica"] = df["condition_label"].map(CONDITION_NAMES)
    df["urgencia"] = df["condition_label"].apply(mapear_urgencia)
    return df[["texto", "categoria_clinica", "urgencia", "n_caracteres"]].reset_index(drop=True)


def save_silver(df: pd.DataFrame) -> None:
    DATA_SILVER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_SILVER_DIR / "laudos.parquet", index=False)
    logger.info("silver_saved", path=str(DATA_SILVER_DIR), linhas=len(df))


def load_silver() -> pd.DataFrame:
    return pd.read_parquet(DATA_SILVER_DIR / "laudos.parquet")


def main() -> None:
    train, test, _labels = load_bronze()
    df = combinar_bronze(train, test)
    df = limpar(df)
    df = aplicar_mapa(df)
    save_silver(df)


if __name__ == "__main__":
    main()
