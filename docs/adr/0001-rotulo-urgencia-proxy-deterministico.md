# ADR-0001 — Rótulo de urgência como proxy determinístico

## Status

Aceito.

## Contexto

O projeto precisa de um rótulo de **urgência de triagem** para treinar um classificador. Nenhum
dataset público com esse rótulo, anotado por triagem clínica real, está acessível sem
credenciamento (MIMIC-III e equivalentes exigem processo de aprovação institucional). O Medical
Abstracts TC Corpus, usado como fonte de dados deste projeto (`docs/dataset.md`), rotula
**categoria clínica** (`condition_label`, 1 a 5) — uma dimensão diferente de urgência.

Duas alternativas foram avaliadas:

| Caminho | Veredito |
|---|---|
| Buscar um dataset com urgência real, mesmo sob credenciamento | Reprovado. O tempo de aprovação de acesso a dados clínicos reais inviabiliza o cronograma, e o objetivo do projeto é o ciclo de vida do modelo em produção, não a curadoria de um dataset médico |
| Derivar urgência por regra determinística a partir da categoria clínica | **Escolhido.** Não é um rótulo clinicamente validado, mas é auditável, documentável e suficiente para exercitar o pipeline completo — desde que a limitação seja declarada, não escondida |

## Decisão

`src/data/urgency_map.py` mapeia cada `condition_label` para uma classe de urgência
(`urgente` / `atencao` / `normal`) por regra fixa, com justificativa clínica declarada por
categoria:

| `condition_label` | Categoria | Urgência | Justificativa |
|---|---|---|---|
| 1 | neoplasms | urgente | Suspeita oncológica exige estadiamento e conduta em janela curta |
| 4 | cardiovascular diseases | urgente | Evento cardiovascular é tempo-dependente por definição |
| 3 | nervous system diseases | atencao | Heterogêneo — de cefaleia crônica a AVC — priorização intermediária |
| 2 | digestive system diseases | atencao | Majoritariamente subagudo, com exceções que a triagem humana resolve |
| 5 | general pathological conditions | normal | Categoria residual, sem marcador de tempo-dependência |

O módulo é isolado de propósito: trocar o proxy por um rótulo real, quando disponível, altera
só este arquivo — nenhuma outra camada (treino, avaliação, API, monitoramento) depende de como
a urgência é derivada.

### Achado durante a implementação — resolução de multi-rótulo

O corpus se revelou multi-rótulo achatado em linhas: o mesmo abstract aparece repetido sob mais
de uma categoria clínica. A regra de resolução (**a maior urgência vence**; empate desempatado
pelo menor `condition_label`) está registrada e justificada em `docs/dataset.md`, incluindo a
comparação contra a alternativa rejeitada (`keep="first"`, que sub-triaria sistematicamente).

## Consequências

- O modelo treinado **não deve ser usado para decisão médica real** — a ressalva está no
  `README.md`, no `docs/model_card.md` e no `docs/dataset.md`, nos três lugares que um
  avaliador ou um reaproveitamento futuro do código passaria.
- A métrica principal de avaliação é macro-F1, não acurácia — o proxy induz um desbalanceamento
  moderado (~2,55:1) e o custo de errar "para baixo" (perder um caso urgente) é maior que o de
  errar "para cima", o que pesa na escolha de métrica além da questão de classes desbalanceadas.
- Um rótulo real substituiria apenas `src/data/urgency_map.py` — o restante do pipeline (silver,
  gold, treino, gate de qualidade, API, DAG) permanece válido sem alteração estrutural.
