# ADR-0007 — Escopo de dados e experimentos: sem DVC nem MLflow

## Status

Aceito.

## Contexto

Dois projetos anteriores desta pós-graduação usaram DVC (versionamento de dados/pipeline) e
MLflow (tracking de experimentos e Model Registry) como parte do critério avaliado. Este
projeto usa um dataset público pequeno (18 MB, `docs/dataset.md`), com pipeline que reconstrói
bronze → silver → gold em minutos a partir de um CSV já commitado, e a promoção de modelo já é
o comportamento avaliado pelo critério de orquestração (a task `promocao` da DAG, ADR-0005) —
não uma decisão de negócio que se beneficiasse de um Model Registry separado.

## Decisão

**Sem DVC.** O volume de dados não justifica a dependência extra: o CSV bruto já está commitado
em `data/bronze/` (§"Camadas" de `docs/dataset.md`), e `make data` reconstrói silver e gold do
zero, em qualquer clone, sem depender de um remote externo — o mesmo objetivo que DVC resolveria
aqui, sem o custo de configuração e de curva de aprendizado de uma ferramenta adicional.

**Sem MLflow.** O ciclo de vida do modelo que este projeto avalia é: treinar → avaliar contra um
piso de qualidade → exportar → promover → servir → monitorar — inteiramente expresso pela DAG do
Airflow e pelos módulos de `src/models/` (`train.py`, `evaluate.py`, `registry.py`). Um segundo
sistema de tracking de experimentos duplicaria a responsabilidade que `registry.py` e
`metrics/eval_metrics.json` já cobrem, sem abrir uma capacidade nova que o projeto precise.

## Consequências

- `metrics/` (versionado seletivamente — ver `.gitignore`) é a única fonte de métricas de
  execução: `eval_metrics.json`, `latency_baseline.json`, `latency_benchmark.json`. Não existe
  um servidor de tracking à parte para consultar.
- Reaproveitar este projeto com um dataset maior ou uma decisão de negócio real sobre qual
  modelo promover reintroduziria a discussão sobre DVC/MLflow — a ausência aqui é uma decisão de
  escopo para este dataset e este objetivo de avaliação, não um veredito geral contra as duas
  ferramentas.
- Rastreabilidade do split de dados é garantida por outro mecanismo, mais simples: um hash
  SHA-256 do conteúdo do split gravado em `data/gold/metadata.parquet` (`docs/dataset.md`), não
  por versionamento de arquivo.
