# ADR-0003 — Contrato da API: `/health` separado de `/ready`, endpoints síncronos

## Status

Aceito.

## Contexto

A API de triagem (`src/api/main.py`) serve inferência em tempo real (ver ADR-0006) e precisa de
dois comportamentos que um único endpoint de saúde não cobre: informar a um orquestrador de
containers que o processo está vivo, e informar, separadamente, que o modelo está carregado e
apto a responder. Confundir os dois faz o orquestrador mandar tráfego para uma instância que
subiu mas ainda não terminou de carregar o modelo — ou que falhou ao carregar e nunca vai ficar
pronta.

A segunda decisão de contrato é como o FastAPI declara os endpoints de inferência: `async def`
ou `def`. A inferência (TF-IDF + regressão logística, ou TF-IDF + ONNX Runtime) é CPU-bound e
síncrona por natureza — não há `await` real dentro dela.

## Decisão

**`/health` (liveness) e `/ready` (readiness) são endpoints distintos.** `/health` só confirma
que o processo está de pé e nunca toca o modelo; `/ready` verifica se o predictor foi carregado
com sucesso no `lifespan` e devolve `503` se não. É o par que permite um orquestrador (ECS, ALB,
Kubernetes) reciclar uma instância que subiu mas não conseguiu carregar o modelo, em vez de
tratá-la como saudável.

**Os endpoints de inferência (`/predict`, `/predict/batch`) são `def`, nunca `async def`** —
norma que vale para todos os endpoints da API, por uniformidade. Dentro de uma coroutine, uma
chamada CPU-bound bloqueia o event loop inteiro do processo: todas as requisições concorrentes
passam a ser atendidas em fila, uma de cada vez, o que invalida qualquer medição de latência sob
carga (`scripts/load_test.py`). Com `def`, o FastAPI despacha a chamada para o threadpool do
Starlette automaticamente, liberando o event loop para outras requisições enquanto uma inferência
roda. Um teste garante a regra: `assert not inspect.iscoroutinefunction(main.predict)`
(`tests/test_api.py`). O `lifespan` continua `async def` — é o que o framework exige para esse
hook, e roda uma única vez na subida, fora do caminho de requisição.

**`modelo_versao` vem de `models/current/model_meta.json`, não do mtime do arquivo.** O treino
grava esse metadado (`data_treino`, macro-F1 do val) em `models/staging/`, e a promoção o copia
junto com os artefatos — metadado de sistema de arquivos não sobrevive a um download de um
object store nem a qualquer cópia que não preserve mtime.

## Consequências

- O modelo é carregado uma única vez no `lifespan`, nunca por requisição — carregar por
  requisição reintroduziria exatamente o overhead que a arquitetura de inferência em tempo real
  (ADR-0006) existe para eliminar.
- Falha ao carregar o modelo não derruba o processo: fica registrada em memória e reportada como
  `503` em `/ready`, mantendo `/health` no ar — o container continua vivo para diagnóstico, mas
  fora do balanceamento de tráfego.
- `/metrics` e `/model/info` seguem a mesma convenção `def` por uniformidade, embora não tenham
  trabalho pesado.
