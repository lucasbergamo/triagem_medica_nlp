"""Testes do pipeline de dados — funções puras, sem rede (bronze não é baixado aqui)."""

import pandas as pd

from src.data.features import montar_metadata, split_estratificado
from src.data.preprocess import aplicar_mapa, combinar_bronze, limpar, resolver_multirrotulo


def _make_train_test():
    train = pd.DataFrame(
        {
            "condition_label": [1, 2, 3],
            "medical_abstract": ["a" * 60, "b" * 60, "c" * 60],
        }
    )
    test = pd.DataFrame(
        {
            "condition_label": [4, 5],
            "medical_abstract": ["d" * 60, "e" * 60],
        }
    )
    return train, test


def test_combinar_bronze_junta_train_e_test():
    train, test = _make_train_test()
    df = combinar_bronze(train, test)
    assert len(df) == len(train) + len(test)
    assert "texto" in df.columns
    assert "medical_abstract" not in df.columns


def test_resolver_multirrotulo_texto_com_labels_4_e_5_vira_urgente():
    """Cardiovascular (4, urgente) e general pathological (5, normal) — o abstract é o
    mesmo; a maior urgência vence, não a ordem em que as linhas aparecem."""
    texto = "mesmo laudo cardiovascular e residual " * 3
    df = pd.DataFrame({"condition_label": [5, 4], "texto": [texto, texto]})
    resolvido = resolver_multirrotulo(df)
    assert len(resolvido) == 1
    assert resolvido.iloc[0]["condition_label"] == 4  # o rótulo que produz "urgente"


def test_resolver_multirrotulo_e_independente_da_ordem_das_linhas():
    texto = "mesmo laudo cardiovascular e residual " * 3
    ordem_a = pd.DataFrame({"condition_label": [4, 5], "texto": [texto, texto]})
    ordem_b = pd.DataFrame({"condition_label": [5, 4], "texto": [texto, texto]})
    assert (
        resolver_multirrotulo(ordem_a).iloc[0]["condition_label"]
        == resolver_multirrotulo(ordem_b).iloc[0]["condition_label"]
    )


def test_resolver_multirrotulo_empate_de_urgencia_desempata_pelo_menor_label():
    """1 (neoplasms) e 4 (cardiovascular) são ambos "urgente" — empate resolvido
    deterministicamente pelo menor condition_label, não pela ordem de leitura."""
    texto = "mesmo laudo oncológico e cardiovascular " * 3
    df = pd.DataFrame({"condition_label": [4, 1], "texto": [texto, texto]})
    resolvido = resolver_multirrotulo(df)
    assert resolvido.iloc[0]["condition_label"] == 1


def test_resolver_multirrotulo_texto_unico_passa_direto():
    df = pd.DataFrame({"condition_label": [1, 2, 3], "texto": ["a", "b", "c"]})
    resolvido = resolver_multirrotulo(df)
    assert len(resolvido) == 3
    assert set(resolvido["condition_label"]) == {1, 2, 3}


def test_limpar_resolve_multirrotulo_fim_a_fim():
    texto_ambiguo = "mesmo laudo cardiovascular e residual " * 3
    df = pd.DataFrame(
        {
            "condition_label": [5, 4, 2],
            "texto": [texto_ambiguo, texto_ambiguo, "outro laudo bem diferente " * 3],
        }
    )
    limpo = limpar(df)
    assert len(limpo) == 2  # o par ambíguo colapsa em 1 linha + o laudo isolado
    linha_ambigua = limpo[limpo["texto"] == texto_ambiguo].iloc[0]
    assert linha_ambigua["condition_label"] == 4


def test_limpar_descarta_texto_curto():
    df = pd.DataFrame(
        {
            "condition_label": [1, 2],
            "texto": ["laudo longo o bastante para passar do piso de 50 caracteres", "curto"],
        }
    )
    limpo = limpar(df)
    assert len(limpo) == 1
    assert limpo.iloc[0]["n_caracteres"] >= 50


def test_limpar_remove_nulos():
    df = pd.DataFrame(
        {
            "condition_label": [1, None],
            "texto": ["laudo válido " * 5, "laudo com label nulo " * 5],
        }
    )
    limpo = limpar(df)
    assert len(limpo) == 1


def test_aplicar_mapa_deriva_categoria_e_urgencia():
    df = pd.DataFrame(
        {
            "condition_label": [1, 4, 5],
            "texto": ["a", "b", "c"],
            "n_caracteres": [1, 1, 1],
        }
    )
    resultado = aplicar_mapa(df)
    assert list(resultado.columns) == ["texto", "categoria_clinica", "urgencia", "n_caracteres"]
    assert resultado.loc[0, "urgencia"] == "urgente"
    assert resultado.loc[0, "categoria_clinica"] == "neoplasms"
    assert resultado.loc[2, "urgencia"] == "normal"


def _make_gold_input(n_por_classe: int = 40) -> pd.DataFrame:
    linhas = []
    for classe in ("urgente", "atencao", "normal"):
        for i in range(n_por_classe):
            linhas.append({"texto": f"{classe}_{i}", "urgencia": classe})
    return pd.DataFrame(linhas)


def test_split_estratificado_soma_o_total_e_nao_vaza_linhas():
    df = _make_gold_input()
    train, val, test = split_estratificado(df, seed=42)
    assert len(train) + len(val) + len(test) == len(df)
    textos_juntos = set(train["texto"]) | set(val["texto"]) | set(test["texto"])
    assert len(textos_juntos) == len(df)  # nenhuma linha duplicada entre os splits


def test_split_estratificado_preserva_proporcao_70_15_15():
    df = _make_gold_input(n_por_classe=100)  # 300 linhas — múltiplo exato de 70/15/15
    train, val, test = split_estratificado(df, seed=42)
    assert len(train) == 210
    assert len(val) == 45
    assert len(test) == 45


def test_split_estratificado_mantem_as_3_classes_em_cada_parte():
    df = _make_gold_input()
    train, val, test = split_estratificado(df, seed=42)
    for parte in (train, val, test):
        assert set(parte["urgencia"].unique()) == {"urgente", "atencao", "normal"}


def test_split_estratificado_e_deterministico_com_a_mesma_seed():
    df = _make_gold_input()
    train_a, val_a, test_a = split_estratificado(df, seed=42)
    train_b, val_b, test_b = split_estratificado(df, seed=42)
    assert train_a["texto"].tolist() == train_b["texto"].tolist()
    assert val_a["texto"].tolist() == val_b["texto"].tolist()
    assert test_a["texto"].tolist() == test_b["texto"].tolist()


def test_montar_metadata_contem_classes_e_contagens_e_hash():
    df = _make_gold_input()
    train, val, test = split_estratificado(df, seed=42)
    metadata = montar_metadata(train, val, test).iloc[0]
    assert metadata["n_classes"] == 3
    assert set(metadata["classes"].split(",")) == {"urgente", "atencao", "normal"}
    assert metadata["n_train"] == len(train)
    assert metadata["n_val"] == len(val)
    assert metadata["n_test"] == len(test)
    assert isinstance(metadata["split_hash"], str) and len(metadata["split_hash"]) == 16
