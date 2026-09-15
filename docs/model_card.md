# Model Card — Classificador de Urgência

## Dados

Treinado sobre a camada gold derivada do Medical Abstracts TC Corpus — ver `docs/dataset.md`
para origem, licença e a construção completa do rótulo. Resumo relevante para este modelo:

- **Rótulo:** `urgencia` (`urgente` / `atencao` / `normal`) é um **proxy determinístico**
  derivado da categoria clínica do corpus (`src/data/urgency_map.py`), não um rótulo de
  triagem validado clinicamente. Ver a ressalva completa em `docs/dataset.md`.
- **Split:** 70/15/15 estratificado por `urgencia`, seed fixa — `data/gold/{train,val,test}.parquet`.
- **Treino:** 7.858 laudos. **Val** (decide o gate): 1.684. **Test** (resultado reportado): 1.685.
- **Desbalanceamento:** ≈2,55:1 entre "urgente" e "normal" — tratado com `class_weight="balanced"`,
  sem SMOTE nem reamostragem.

## Modelo

Pipeline de dois estágios (`src/models/train.py`):

```python
Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_features=50_000,
        sublinear_tf=True,
        strip_accents="unicode",
        lowercase=True,
    )),
    ("clf", LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        random_state=seed,
    )),
])
```

Escolhido sobre Random Forest (o exemplo do enunciado) porque converte a inferência inteira
em um GEMM — exatamente o caso em que a exportação para ONNX Runtime ganha desempenho.
RF exportado para ONNX serializa uma árvore por estimador; o artefato explode em tamanho e a
inferência pode ficar mais lenta que em sklearn, o que destruiria a história de latência
medida na etapa de otimização.

## Métricas — val decide, test reporta

`src/models/evaluate.py::run()` avalia o modelo staged nas duas partições que nunca entram
no treino. Elas têm papéis diferentes, de propósito:

- **val** (1.684 laudos) — é o que o **gate de qualidade** (`validar()`) olha para decidir
  se o modelo pode ser promovido. Num pipeline de retreino, essa decisão roda a cada
  execução da DAG; se decidisse olhando o test, o test deixaria de ser uma estimativa
  imparcial — a mesma partição vira, na prática, parte do critério de seleção do modelo.
- **test** (1.685 laudos) — nunca influencia a promoção. É a estimativa imparcial do
  desempenho final, medida **depois** da decisão do gate, e é o número reportado abaixo e
  no README como resultado deste modelo.

| Partição | Papel | Macro-F1 | Acurácia |
|---|---|---|---|
| val | decide o gate | 0,7429 | 0,7803 |
| **test** | **resultado reportado** | **0,7562** | 0,7899 |

As duas ficam próximas (diferença de 0,013 em macro-F1) — esperado, já que vêm do mesmo
`stratify` sobre a mesma distribuição (§`docs/dataset.md`); não é garantia de que sempre
vão ficar assim, é só o que esta execução mostrou.

Por classe (conjunto de **test** — o número reportado):

| Classe | Precision | Recall | F1 | Suporte |
|---|---|---|---|---|
| urgente | 0,9088 | 0,8364 | 0,8711 | 917 |
| atencao | 0,7309 | 0,7721 | 0,7509 | 408 |
| normal | 0,6073 | 0,6917 | 0,6468 | 360 |

Matriz de confusão (linhas = real, colunas = predito; ordem `atencao, normal, urgente`):

| | atencao | normal | urgente |
|---|---|---|---|
| **atencao** | 315 | 73 | 20 |
| **normal** | 54 | 249 | 57 |
| **urgente** | 62 | 88 | 767 |

**Resultado acima da expectativa registrada em arquitetura (0,55–0,68).** A hipótese mais
provável: o proxy de urgência deste projeto deriva diretamente da categoria clínica do
corpus (§`docs/dataset.md`), e a categoria clínica é justamente o que o texto do abstract
descreve com mais vocabulário distintivo — "neoplasms" e "cardiovascular" carregam termos
lexicais próprios (tumor, câncer, infarto, coronário) que um TF-IDF captura bem. Um rótulo de
urgência clínica real, anotado por triagem humana e não derivado da categoria, tende a ser
mais difícil de separar por léxico — a classe "normal" (a mais fraca aqui, F1 0,6468, herdada
da categoria residual "general pathological conditions") já sugere isso: é a categoria com
vocabulário menos específico, e é onde o modelo mais confunde.

O erro dominante da matriz é **normal → urgente** (88 casos) e **atencao → urgente** (57
casos) — o modelo super-classifica como urgente. Em triagem, esse é o erro mais barato dos
dois possíveis: mandar um caso normal para revisão prioritária custa tempo; classificar um
caso urgente como normal custa uma vida. A métrica principal (macro-F1, que pesa as três
classes igualmente) já reflete essa prioridade melhor que acurácia simples.

## Comparação com o baseline (Random Forest)

Mesmo `TfidfVectorizer`, classificador trocado — treinado em `src/models/train.py --baseline`:

| Modelo | Macro-F1 | Acurácia | Tempo de treino | Tamanho do artefato |
|---|---|---|---|---|
| **LogisticRegression** (escolhido) | **0,7562** | 0,7899 | 8,08 s | 3,1 MB |
| RandomForest (baseline) | 0,7296 | 0,7751 | 4,22 s | 32,6 MB |

RandomForest treina mais rápido neste corpus, mas perde em macro-F1 e produz um artefato
**~10× maior** — o tamanho vem do número de árvores serializadas, e é justamente o que
inviabiliza uma exportação ONNX enxuta (§ acima). A escolha da LogisticRegression não é só
sobre acurácia: é sobre o artefato inteiro ser leve o bastante para a etapa de otimização de
latência ter uma história para contar.

## Gate de qualidade

`src/models/evaluate.py::validar()` bloqueia a promoção se **o macro-F1 do val** ficar
abaixo de `MIN_MACRO_F1` (default 0,50, configurável via `.env`) — nunca o do test, pelo
motivo explicado na seção acima. `metrics/eval_metrics.json` grava os dois blocos
(`"val"` e `"test"`) lado a lado, para auditoria. Testado neste bloco com
`MIN_MACRO_F1=0.99`: o `make eval` falha com exit code 1, a mensagem cita explicitamente
"macro-F1 (val)" e lista a falha — nenhum artefato é promovido.

## Reprodutibilidade entre ambientes — local vs. DAG do Airflow

Os números publicados neste documento e no README (val 0,7429 / test 0,7562) vêm de `make train`
rodando no ambiente do projeto (`poetry.lock`: scikit-learn 1.9.1, numpy 2.4.6, scipy 1.17.1). A
DAG `treino_triagem` (ADR-0005) treina o mesmo código, sobre os mesmos dados, dentro da imagem do
Airflow — cujo ecossistema (`airflow/constraints.txt`, da Apache) fixa numpy 1.26.4 e scipy
1.16.3. Medido: a DAG produz val 0,7420 / test 0,7530 — uma diferença de ~0,003, na terceira
casa decimal.

A causa é o backend numérico (BLAS/LAPACK) por trás de `LogisticRegression.fit`, não código nem
dado: o split de dados é idêntico nos dois ambientes, confirmado pelo mesmo `split_hash`
(`7e8de5fa1d2760a2`) em `data/gold/metadata.parquet`. O scikit-learn foi explicitamente alinhado
para 1.9.1 nas duas pontas (`airflow/Dockerfile` instala essa versão à parte do
`constraints.txt`, que fixaria 1.8.0) justamente para reduzir essa superfície de divergência — o
que sobra é numpy/scipy, que o `constraints.txt` do Airflow fixa e que trocar exigiria uma versão
de numpy fora do que aquele ecossistema testa e garante.

É por isso que o gate de qualidade (`src/models/evaluate.py::validar()`) compara contra um
**piso** (`macro-F1 ≥ MIN_MACRO_F1`), não uma igualdade com um número fixo: um pipeline de
retreino que exigisse bater exatamente 0,7562 quebraria na primeira variação legítima de
ambiente, mesmo com o modelo continuando bom o suficiente para produção.

## Limitações

- **O rótulo é um proxy, não triagem clínica real** — ver ressalva completa em `docs/dataset.md`.
  Este modelo não deve ser usado para decisão médica.
- **Classe "normal" é a mais fraca** (F1 0,6468) — é a categoria residual do corpus original,
  com o vocabulário menos específico das três.
- **Treinado só em abstracts em inglês do corpus público** — não validado contra laudos em
  português nem contra o vocabulário de um pronto-socorro real.
- **Sem calibração de probabilidade** — `predict_proba` da LogisticRegression não foi
  calibrado (`CalibratedClassifierCV` ou similar); a confiança reportada é a saída bruta do
  modelo.

## Uso indevido

Não usar para: decisão de atendimento médico real, priorização de fila hospitalar sem
supervisão humana, ou qualquer contexto em que o rótulo de urgência precise refletir
julgamento clínico validado. O propósito deste projeto é demonstrar o ciclo de vida de um
modelo em produção (deploy, monitoramento, CI/CD, orquestração) — não substituir triagem
humana.
