# Triagem Médica NLP

> Tech Challenge Fase 03 — FIAP Pós-Tech MLET. Triagem automática de laudos médicos por
> urgência, com o ciclo de vida completo do modelo (deploy, monitoramento, CI/CD, orquestração)
> como foco de avaliação — não a acurácia do classificador.

**Este README é um esqueleto (B0).** A versão completa — badges, diagrama de arquitetura,
decisão de nuvem, quick start, tabela de latência e o mapa dos 6 critérios do enunciado — entra
no bloco B9. Enquanto isso, o contrato do projeto vive em
[`ARQUITETURA.md`](../fiap-mlet-graduation/tech-challenge-03/ARQUITETURA.md) (fora deste repo,
no sandbox de processo).

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
