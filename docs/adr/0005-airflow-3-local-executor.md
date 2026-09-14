# ADR-0005 — Airflow 3.1.5 com LocalExecutor

## Status

Aceito.

## Contexto

A orquestração de retreino precisa de um agendador real — não um script chamado por cron —
para expressar o pipeline `dados → treino → avaliação (gate) → exportação → promoção →
benchmark` como um grafo de dependências, com retry, log por task e um gate de qualidade que
efetivamente impede um modelo pior de chegar à produção.

Três caminhos foram avaliados:

| Caminho | Veredito |
|---|---|
| `airflow standalone` + SQLite | Reprovado. Usa `SequentialExecutor`, que não paraleliza tasks, e não reflete como Airflow roda em nenhum ambiente real |
| `LocalExecutor` + Postgres | **Escolhido.** Airflow de verdade — múltiplas tasks em paralelo, banco de metadados real —, sobe em poucos minutos num notebook, e o salto para um executor gerenciado (ex.: MWAA) é mudança de configuração, não de arquitetura |
| `CeleryExecutor` + Redis + workers | Reprovado para este projeto: broker e workers adicionais sem nenhum ganho de paralelismo perceptível numa DAG de 8 tasks sequenciais rodando em uma máquina só |

## Decisão

**Airflow 3.1.5**, executor `LocalExecutor`, metadados em Postgres 16, todos containerizados
sob um perfil separado do compose — a stack de orquestração não sobe junto com a stack de
serving/observabilidade por padrão (memória limitada no ambiente de desenvolvimento local).

A versão 3.1.5 é uma divergência consciente do material de referência da disciplina, que usa
Airflow 2.10.4: a sintaxe de DAG mudou (`schedule` no lugar de `schedule_interval`, entre
outras), mas a estrutura do pipeline — tasks como função fina sobre módulos de aplicação, gate
de qualidade levantando exceção quando a métrica não atinge o piso — é a mesma. O README mapeia
a correspondência entre a DAG deste projeto e a DAG de referência da disciplina.

### Ordem das tasks: promoção antes do benchmark de latência

O desenho original do pipeline previa `exportação → benchmark → promoção`: medir a latência do
modelo recém-exportado antes de promovê-lo. Na prática, o benchmark de latência sempre mede o
modelo em `models/current/` — é o mesmo caminho que a API lê em produção, e reaproveitar esse
caminho (em vez de duplicar a lógica de carregamento de modelo para ler de `models/staging/`)
evita que o script de benchmark e o `Predictor` da API divirjam sobre onde um modelo "ativo" mora.

Com a promoção depois do benchmark, a task mediria o modelo que está **saindo** de produção,
não o que este run acabou de treinar — o número gravado ficaria descolado do próprio retreino
que o gerou. A ordem adotada é `exportação → promoção → benchmark`: o benchmark roda sobre o
modelo que a API vai efetivamente servir a partir desse momento. Isso não muda o critério de
aprovação do modelo — quem decide se um modelo é promovido é exclusivamente a task de
avaliação, antes da exportação; o benchmark continua puramente informativo.

## Consequências

- A imagem do Airflow (`airflow/Dockerfile`) instala as mesmas bibliotecas de ML do projeto
  (`airflow/requirements.txt`), porque cada task importa um módulo de `src/`/`scripts/`
  diretamente — não existe uma segunda cópia de lógica de negócio dentro da DAG.
- Airflow fica fora do Poetry do projeto: instalá-lo no mesmo ambiente da API conflitaria
  dependências. `tests/test_dag.py` roda dentro da imagem do Airflow (que carrega `pytest`
  junto); no pytest do projeto, esse arquivo se pula sozinho via
  `pytest.importorskip("airflow")`.
- Import pesado (scikit-learn, onnxruntime) fica dentro do corpo de cada `@task`, nunca no topo
  do arquivo da DAG — o dag-processor reimporta o módulo a cada ciclo de parsing, e um import
  pesado no topo estoura o `DAGBAG_IMPORT_TIMEOUT` (60s) e a DAG desaparece da UI sem aviso
  claro.
- O perfil `airflow` do compose raiz não sobe por padrão junto com `api-sklearn`/`api-onnx`/
  `prometheus`/`grafana` — as duas stacks competem por RAM no ambiente de desenvolvimento local,
  e nenhuma delas depende da outra estar de pé.
