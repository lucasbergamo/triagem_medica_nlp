# Triagem Médica NLP

> Tech Challenge Fase 03 — FIAP Pós-Tech MLET. Triagem automática de laudos médicos por
> urgência, com o ciclo de vida completo do modelo (deploy, monitoramento, CI/CD, orquestração)
> como foco de avaliação — não a acurácia do classificador.

**Este README ainda está em construção.** Badges, diagrama de arquitetura completo e o mapa
dos 6 critérios do enunciado são escritos ao final, quando cada peça já estiver implementada.
Até lá, as decisões de cada etapa ficam documentadas em `docs/` (dataset, model card, ADRs)
conforme os arquivos entram no repo.

## Status

Em construção, bloco por bloco. Ver `CLAUDE.md` para as regras de execução.

## Setup rápido (dev local)

```bash
make install     # poetry install
cp .env.example .env
make validate    # sanity check: Python, pacotes, diretórios, .env
make lint
make test
```

## Docker (paridade CI)

```bash
docker compose --profile ci build
docker compose --profile ci run --rm lint
docker compose --profile ci run --rm ci
```

## API

`FastAPI`, modelo carregado uma única vez no `lifespan` — nunca por requisição.

```bash
make train && make eval && make promote   # gera models/current/pipeline.joblib
make serve                                 # uvicorn local, localhost:8000
# ou
make docker-serve                          # imagem com o modelo embutido (estágio `serve`)
```

| Método | Rota | Papel |
|---|---|---|
| `POST` | `/predict` | Recebe o texto de um laudo, devolve urgência + confiança |
| `POST` | `/predict/batch` | Lote de 1 a 100 laudos |
| `GET` | `/health` | Liveness — processo de pé, nunca toca o modelo |
| `GET` | `/ready` | Readiness — modelo carregado e respondendo (healthcheck de ECS/ALB) |
| `GET` | `/model/info` | Backend ativo, versão do artefato, classes |

```jsonc
// POST /predict → request
{ "texto": "Paciente apresenta dor precordial em aperto com irradiação..." }

// POST /predict → response
{
  "urgencia": "urgente",
  "confianca": 0.87,
  "probabilidades": { "normal": 0.04, "atencao": 0.09, "urgente": 0.87 },
  "latencia_inferencia_ms": 1.83,
  "backend": "sklearn",
  "modelo_versao": "2026-09-11T21:06:11.536960+00:00"
}
```

`texto` é validado por Pydantic (20 a 20.000 caracteres); fora desses limites, `422` com
mensagem em português. Falha de inferência devolve `503`, nunca um `500` mudo. `/predict/batch`
segue o mesmo contrato de item, sob a chave `resultados`.

Todos os endpoints — inclusive os dois de inferência — são `def`, nunca `async def`: a
inferência (TF-IDF + regressão logística) é CPU-bound e síncrona, e dentro de uma coroutine
ela bloquearia o event loop, enfileirando todas as requisições concorrentes. Com `def`, o
FastAPI despacha a chamada para o threadpool do Starlette, liberando o event loop — é o que
mantém a medição de latência sob carga válida.

### Latência de inferência (baseline)

Medição pura de `predictor.predict()` — sem HTTP nem serialização — via
`scripts/benchmark_latency.py`: 200 iterações de aquecimento descartadas, 1.000 medições com
`time.perf_counter_ns`, amostras reais do conjunto de test.

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | Média (ms) | Desvio (ms) | Artefato (KB) |
|---|---|---|---|---|---|---|
| sklearn | 0,7944 | 1,2701 | 1,8117 | 0,8414 | 0,2366 | 3188,1 |

Só o backend `sklearn` existe até aqui — é a baseline contra a qual o ganho do ONNX (bloco de
otimização) vai ser medido, no mesmo script.

## Decisão de nuvem — real-time, não batch nem serverless

O objetivo da triagem é ordenar a fila de atendimento **enquanto o paciente está esperando**,
não gerar um relatório depois do fato. Isso decide a arquitetura antes de qualquer detalhe de
implementação — os três padrões de inferência em nuvem, comparados:

| Padrão | Avaliação |
|---|---|
| **Batch** | Reprovado. Um lote processado periodicamente entrega a classificação depois que a decisão de atendimento já foi tomada — o resultado chega tarde demais para servir ao propósito do produto. |
| **Serverless** (função sob demanda) | Reprovado. O cold start de uma função fria — dezenas de milissegundos — domina o tempo de resposta quando a inferência em si custa menos de 1 ms (ver tabela acima): o overhead de infraestrutura passaria a ser o gargalo, exatamente o que a otimização de latência deste projeto existe para eliminar. |
| **Real-time (API síncrona)** | **Escolhido.** Um serviço sempre ativo, com o modelo já carregado em memória, responde no tempo da requisição. É o único padrão compatível com um SLO de latência apertado (p95 abaixo de 100 ms fim a fim) e com a separação `/health`/`/ready` que permite reciclar instâncias sem impacto no tráfego. |

Consequência direta desta escolha: a API carrega o modelo uma única vez no `lifespan` e o
mantém em memória — recarregar por requisição reintroduziria exatamente o tipo de overhead que
o padrão serverless tem por natureza. O alvo de implantação segue o mesmo raciocínio: um
serviço containerizado sempre ativo (ECS Fargate), atrás de um balanceador que só recebe
tráfego depois que `/ready` confirma o modelo carregado — nunca uma função efêmera por
requisição.
