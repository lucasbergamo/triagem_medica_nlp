"""Testes do mapa de urgência — proxy determinístico a partir da categoria clínica do corpus."""

import pytest

from src.data.urgency_map import (
    ATENCAO,
    CONDITION_NAMES,
    NORMAL,
    URGENCY_MAP,
    URGENTE,
    mapear_urgencia,
)


@pytest.mark.parametrize(
    ("condition_label", "urgencia_esperada"),
    [
        (1, URGENTE),  # neoplasms
        (4, URGENTE),  # cardiovascular diseases
        (3, ATENCAO),  # nervous system diseases
        (2, ATENCAO),  # digestive system diseases
        (5, NORMAL),  # general pathological conditions
    ],
)
def test_mapear_urgencia_cobre_as_5_categorias(condition_label, urgencia_esperada):
    assert mapear_urgencia(condition_label) == urgencia_esperada


def test_mapear_urgencia_label_desconhecido_levanta_value_error():
    with pytest.raises(ValueError, match="condition_label desconhecido"):
        mapear_urgencia(99)


def test_urgency_map_e_condition_names_tem_as_mesmas_5_chaves():
    assert set(URGENCY_MAP.keys()) == set(CONDITION_NAMES.keys()) == {1, 2, 3, 4, 5}


def test_urgency_map_so_produz_os_3_niveis_documentados():
    assert set(URGENCY_MAP.values()) == {URGENTE, ATENCAO, NORMAL}
