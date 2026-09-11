# CLAUDE.md — triagem_medica_nlp

Repo oficial do Tech Challenge Fase 03 (FIAP Pós-Tech MLET) — triagem automática de laudos
médicos por urgência. Este arquivo traz comandos, arquitetura do repo e as regras de trabalho
válidas para toda sessão. Documentos de planejamento (contrato de arquitetura, ordem dos
blocos de execução) não vivem neste repo — uma sessão que precisar deles recebe os caminhos
pelo prompt de abertura.

## Regras que valem para todos os blocos

1. **Branch por bloco.** `git checkout -b <branch do bloco>` a partir da `main` atualizada. Nunca commitar na `main`.
2. **Commit só com autorização explícita do Lucas.** Nunca `git commit` por conta própria. Nunca `Co-Authored-By`.
3. **PR é criado pela sessão, merge é ato do Lucas.** Depois do push, criar o PR com `gh pr create` (título em conventional commits, corpo em português resumindo os commits, sem rodapé de ferramenta) e mandar o link para o Lucas aprovar o merge pela interface web. Nada de `gh pr merge`.
4. **Conventional commits**, escopo entre parênteses: `feat(api):`, `fix(docker):`, `docs(readme):`, `refactor(data):`, `test(models):`, `ci:`.
5. **Walkthrough antes do aceite.** Ao terminar um bloco: arquivo por arquivo, o que mudou, o trecho relevante, o porquê. Nunca só "pronto, funcionou".
6. **Sem `print()`.** `from src.utils.logger import get_logger`.
7. **`set_global_seed()`** antes de qualquer split ou treino.
8. **Definition of Done do bloco é a lista de aceite** — todos os itens verificados **rodando**, não "o arquivo existe".
9. **Se uma decisão de arquitetura já registrada não funcionar na prática, avisar e propor uma alternativa** — não improvisar em silêncio.
10. **Onde cada coisa mora:**
    - Todo o código vive só aqui.
    - Entregáveis (model card, `dataset.md`, `latencia.md`, `arquitetura_nuvem.md`, ADRs, roteiro do vídeo, prints) vão para `docs/` deste repo — sustentam a nota.
11. **O repo é autocontido.** Uma decisão que precisa de justificativa é explicada no próprio arquivo ou citada via `docs/` deste repo (`dataset.md`, `model_card.md`, `adr/`, README) — nunca por referência a um documento ou caminho que só existe fora do repo. O repo vai ficar público; quem avalia não tem acesso a nada fora daqui.

## `airflow/`

Já existe no repo, criado antes do B0. **Não sobrescrever nem mover** — Airflow 3.1.5,
`LocalExecutor` + Postgres, versão e executor já testados de ponta a ponta localmente (o
`airflow standalone` usa `SequentialExecutor`, sem paralelismo, e não reflete como Airflow
roda em produção). Integrado ao compose raiz sob o perfil `airflow` no B6.
