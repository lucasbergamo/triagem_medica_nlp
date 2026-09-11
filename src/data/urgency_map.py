"""Mapa de urgência — proxy determinístico a partir da categoria clínica do corpus.

Módulo isolado de propósito: o corpus traz categoria clínica, não urgência real — nenhum
dataset público de triagem com rótulo de urgência real está acessível sem credenciamento.
Quando um rótulo de triagem validado clinicamente estiver disponível, ele
substitui o mapa abaixo sem alterar nenhuma outra camada do pipeline — silver, gold, treino
e API só conhecem os valores de `URGENCY_MAP`, nunca o `condition_label` original.
"""

URGENTE = "urgente"
ATENCAO = "atencao"
NORMAL = "normal"

CONDITION_NAMES: dict[int, str] = {
    1: "neoplasms",
    2: "digestive system diseases",
    3: "nervous system diseases",
    4: "cardiovascular diseases",
    5: "general pathological conditions",
}

# Justificativa clínica declarada por categoria — ver docs/dataset.md para a tabela completa.
URGENCY_MAP: dict[int, str] = {
    1: URGENTE,  # neoplasms — suspeita oncológica exige estadiamento e conduta em janela curta
    4: URGENTE,  # cardiovascular diseases — evento cardiovascular é tempo-dependente por definição
    3: ATENCAO,  # nervous system diseases — heterogêneo: de cefaleia crônica a AVC
    2: ATENCAO,  # digestive system diseases — majoritariamente subagudo, com exceções
    5: NORMAL,  # general pathological conditions — categoria residual, sem tempo-dependência
}

# Ordem de severidade — usada em src/data/preprocess.py para resolver abstracts
# multi-rótulo (o mesmo texto anotado sob mais de uma categoria clínica no corpus):
# entre os rótulos de um mesmo texto, vence o de maior urgência. Regra adotada em
# 11/09/2026 — justificativa em docs/dataset.md.
RANK_URGENCIA: dict[str, int] = {NORMAL: 0, ATENCAO: 1, URGENTE: 2}


def mapear_urgencia(condition_label: int) -> str:
    """Aplica o proxy determinístico de urgência a partir do `condition_label` do corpus.

    Levanta `ValueError` para qualquer rótulo fora de 1 a 5 — silencioso aqui viraria
    dado sujo em silver sem ninguém perceber.
    """
    try:
        return URGENCY_MAP[condition_label]
    except KeyError as exc:
        raise ValueError(f"condition_label desconhecido: {condition_label!r}") from exc
