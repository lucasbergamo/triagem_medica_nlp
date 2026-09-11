# CLAUDE.md — triagem_medica_nlp

Repo oficial do Tech Challenge Fase 03 (FIAP Pós-Tech MLET). O contrato técnico completo está em
`ARQUITETURA.md`, no sandbox `/home/lucas/projects/fiap-mlet-graduation/tech-challenge-03/` — não
neste repo. A ordem e o escopo de cada etapa estão em `BLOCOS_EXECUCAO.md`, no mesmo sandbox.

Toda sessão que trabalhar aqui deve ler os dois antes de escrever código.

## Regras que valem para todos os blocos

1. **Branch por bloco.** `git checkout -b <branch do bloco>` a partir da `main` atualizada. Nunca commitar na `main`.
2. **Commit só com autorização explícita do Lucas.** Nunca `git commit` por conta própria. Nunca `Co-Authored-By`.
3. **PR é criado pela sessão, merge é ato do Lucas.** Depois do push, criar o PR com `gh pr create` (título em conventional commits, corpo em português resumindo os commits, sem rodapé de ferramenta) e mandar o link para o Lucas aprovar o merge pela interface web. Nada de `gh pr merge`.
4. **Conventional commits**, escopo entre parênteses: `feat(api):`, `fix(docker):`, `docs(readme):`, `refactor(data):`, `test(models):`, `ci:`.
5. **Walkthrough antes do aceite.** Ao terminar um bloco: arquivo por arquivo, o que mudou, o trecho relevante, o porquê. Nunca só "pronto, funcionou".
6. **Sem `print()`.** `from src.utils.logger import get_logger`.
7. **`set_global_seed()`** antes de qualquer split ou treino.
8. **Definition of Done do bloco é a lista de aceite** — todos os itens verificados **rodando**, não "o arquivo existe".
9. **Se algo do `ARQUITETURA.md` estiver errado na prática, avisar e propor** — não improvisar em silêncio.
10. **Onde cada coisa mora:**
    - Todo o código vive só aqui. Não existe espelho no sandbox.
    - O sandbox `fiap-mlet-graduation/tech-challenge-03/` guarda só processo: `ARQUITETURA.md`, `BLOCOS_EXECUCAO.md`, `PLANO_ACAO_FASE03.md`, `docs/jornada.md`, `docs/defesa_tecnica.md`, o PDF do enunciado.
    - Entregáveis (model card, `dataset.md`, `latencia.md`, `arquitetura_nuvem.md`, ADRs, roteiro do vídeo, prints) vão para `docs/` **deste repo** — sustentam a nota.
    - O arquivamento do código no sandbox acontece uma vez, depois da entrega (B11).
11. **Consultar o material oficial do curso antes de improvisar.** Fica em `fiap-mlet-graduation/aulas/referencias/materiais-mlet-main/fase-03-deploy-e-servir-modelos/`. O mapa "peça nossa → arquivo da aula" está em `ARQUITETURA.md` §11.

## `airflow/`

Já existe no repo, criado antes do B0. **Não sobrescrever nem mover** — Airflow 3.1.5,
`LocalExecutor` + Postgres, decisão já validada pelo Lucas (`ARQUITETURA.md` §6.1). É integrado
ao compose raiz sob o perfil `airflow` no B6.
