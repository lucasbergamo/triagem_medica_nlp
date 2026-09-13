# Latência — metodologia, resultados e de onde o ganho vem

## Metodologia

`scripts/benchmark_latency.py --comparar` mede três coisas, sempre com o mesmo protocolo —
200 iterações de aquecimento descartadas (JIT do scipy, cache de alocador do ONNX Runtime),
1.000 medições, `time.perf_counter_ns`, amostras reais de `data/gold/test.parquet` (nunca
string sintética):

1. **`predict()` completo** — vetorização + classificação, por backend.
2. **Decomposição** — vetorização (TF-IDF) e classificação medidas separadamente. A
   classificação é medida sobre a saída do TF-IDF já calculada fora do cronômetro (senão a
   etapa de vetorização seria contada duas vezes). Para onnx/onnx-int8, a densificação
   (`toarray().astype(np.float32)`) entra no tempo do classificador de propósito: é trabalho
   real que `OnnxPredictor.predict()` faz por requisição, não é grátis.
3. **Tamanho por componente** — vetorizador e classificador serializados isoladamente (em
   memória, nunca em disco — não são artefatos do projeto), porque `pipeline.joblib` empacota
   os dois juntos e o tamanho desse arquivo sozinho não diz quanto é de cada um.

Mede só `predictor.predict()`/componentes internos, sem HTTP nem serialização — a latência
fim a fim (com FastAPI) é papel de `scripts/load_test.py`; confundir as duas é o erro clássico
aqui. `metrics/latency_baseline.json` é a baseline do sklearn, congelada desde o bloco de API
e nunca sobrescrita por este script. `metrics/latency_benchmark.json` tem os três resultados
completos, incluindo p99, média e desvio padrão.

**Hardware:** WSL2 (Ubuntu 24.04), 18 CPUs, 9,7 GB de RAM, x86_64. Medições in-process,
single-thread por requisição — sem concorrência simulada (isso é papel do `load_test.py`).

## Resultado 1 — `predict()` completo

| Backend | p50 (ms) | p95 (ms) | p99 (ms) | Speedup p95 |
|---|---|---|---|---|
| sklearn | 0,796 | 1,187 | 1,641 | 1,00x |
| onnx | 0,749 | 1,188 | 1,467 | 1,00x |
| onnx-int8 | 0,710 | 1,064 | 1,551 | 1,12x |

Ganho fim a fim modesto — 12% em p95 para o int8, praticamente nenhum para o fp32. O resto
deste documento existe para explicar por quê, com números, não com suposição.

## Resultado 2 — de onde o tempo é gasto (decomposição)

| Etapa | p50 (ms) | p95 (ms) |
|---|---|---|
| Vetorização TF-IDF (comum aos 3 backends) | 0,557 | 1,047 |
| Classificador (sklearn) | 0,129 | 0,169 |
| Classificador (onnx) | 0,080 | 0,116 |
| Classificador (onnx-int8) | 0,055 | 0,075 |

**O TF-IDF domina.** Em p50, ele é 70% do tempo do backend sklearn (0,557 de 0,796 ms),
subindo para 74% no onnx e 78% no int8 — a fração cresce porque o denominador (o total)
encolhe conforme o classificador acelera, não porque o TF-IDF em si mudou (ele é o mesmo
código Python nos três backends). O restante de cada total — a diferença entre "vetorização +
classificação" e o `predict()` medido no Resultado 1 — é overhead fixo de Python por chamada
(empacotar `[texto]`, montar o `dict`/`Predicao`, os dois `time.perf_counter()`): consistente
em torno de 14% nos três backends, o que confirma que é overhead estrutural, não algo que um
backend paga e outro não.

Isso é Lei de Amdahl medida no próprio sistema: o classificador é a única etapa que os
backends onnx/onnx-int8 trocam de motor de execução, mas ele nunca foi mais que ~16% do
tempo total (sklearn) — mesmo acelerando-o de verdade, o teto do ganho fim a fim é baixo
porque a parcela otimizável é pequena.

**O classificador acelera de verdade** — isolado, o ganho é muito maior que os 12% do
`predict()` completo:

| Comparação (p50) | Speedup |
|---|---|
| onnx (fp32) vs. sklearn | 1,61x |
| onnx-int8 vs. sklearn | 2,33x |
| onnx-int8 vs. onnx (fp32) | 1,45x |

### O achado que não bate com a hipótese inicial: onnx fp32 já é mais rápido que sklearn, antes de qualquer quantização

A expectativa antes de medir era que o classificador onnx (fp32) fosse **mais lento** que o
sklearn — o `Gemm` recebe um vetor denso de 50.000 floats, enquanto `LogisticRegression`
multiplica direto sobre a matriz esparsa que o TF-IDF já produz, e multiplicação esparsa
devia ganhar numa matriz com a maioria das posições zeradas. Medido três vezes de forma
independente, o resultado é o oposto: onnx fp32 (0,080 ms) é consistentemente mais rápido que
sklearn (0,129 ms) — antes de qualquer int8.

A explicação mais provável não é o formato dos dados, é o **overhead fixo por chamada**:
`LogisticRegression.predict_proba()` do scikit-learn carrega validação de entrada, despacho
para `scipy.special.softmax` e o caminho genérico de `predict_proba` da API do sklearn a cada
chamada. `InferenceSession.run()` do ONNX Runtime, sobre um grafo de dois nós já compilado
(`Gemm` + `Softmax`), não tem esse caminho — e nessa escala (uma matriz 1×50.000 por
3 classes), o cálculo em si é trivial nos dois casos: o que domina é o overhead do
**caminho de código** em volta do cálculo, não a densidade dos dados. A quantização int8
reduz ainda mais — provavelmente por um tensor de pesos menor (147 KB vs. 586 KB) caber
melhor em cache, mais que por qualquer efeito de aritmética inteira nessa escala de matriz.

## Resultado 3 — tamanho por componente (a correção mais importante deste documento)

Uma versão anterior deste documento comparava o **pipeline sklearn inteiro** (vetorizador +
classificador, 3.188 KB) contra o **classificador onnx sozinho** (586 KB / 147 KB) e chamava
isso de "21,6x menor". Está errado: é comparar coisas de tamanho diferente por natureza,
não uma redução real. A comparação correta é por componente:

| Componente | Tamanho |
|---|---|
| Vetorizador TF-IDF (comum aos 3 backends) | 2.015,3 KB |
| Classificador sklearn | 1.172,9 KB |
| Classificador onnx (fp32) | 586,2 KB |
| Classificador onnx-int8 | 147,4 KB |

Aqui sim o ganho é real: o classificador onnx-int8 é **8x menor** que o classificador
sklearn (1.172,9 → 147,4 KB) — o número que a arquitetura original citava, só que aplicado
à peça certa.

### O footprint total em disco — e por que os backends onnx não são menores

`OnnxPredictor` (`src/models/predictor.py`) carrega o TF-IDF do **mesmo `pipeline.joblib`**
que o backend sklearn usa — inclusive o classificador sklearn embutido nele, que fica sem uso
nesses dois backends. Isso significa que o footprint total em disco de cada backend é:

| Backend | Footprint total | Composição |
|---|---|---|
| sklearn | 3.188,1 KB | `pipeline.joblib` (vetorizador + classificador, os dois em uso) |
| onnx | **3.774,3 KB** | `pipeline.joblib` inteiro (só o TF-IDF é usado) + `pipeline.onnx` |
| onnx-int8 | **3.335,5 KB** | `pipeline.joblib` inteiro (só o TF-IDF é usado) + `pipeline.int8.onnx` |

**Os backends onnx e onnx-int8 ocupam mais espaço em disco que o sklearn, não menos** — 18%
e 5% a mais, respectivamente — porque carregam um classificador sklearn que nunca chamam,
só para reaproveitar o vetorizador de dentro do mesmo arquivo. Reduzir o footprint total de
verdade exigiria um artefato "vetorizador sozinho" (`tfidf.joblib`, sem o `clf` dentro) — não
implementado neste bloco; fica registrado aqui como a otimização óbvia que falta, não como
algo que o desenho atual já entrega.

## De onde o ganho vem — e de onde não vem

O ganho de latência é real, mas modesto (**12% em p95** para o onnx-int8) porque a etapa cara
desta inferência é a vetorização TF-IDF, não a classificação, e o TF-IDF continua em Python
nos três backends (ver seção seguinte para o porquê). Isolado, o classificador acelera de
verdade — até 2,33x — mas ele nunca foi mais que ~16% do tempo total, então mesmo um ganho
grande ali move pouco o número fim a fim. Em tamanho, o ganho por componente é real e grande
(8x no classificador), mas não se traduz em footprint total menor, porque o desenho atual
ainda depende do artefato sklearn completo para a vetorização.

A leitura honesta, sem inflar nada: com a inferência já em regime sub-milissegundo antes de
qualquer otimização (baseline sklearn: p50 0,79 ms — `metrics/latency_baseline.json`), mover
só o classificador para ONNX entrega uma melhora mensurável, mas pequena, no fim a fim — e
nenhuma melhora de footprint em disco, dado como o TF-IDF é reaproveitado hoje. O valor desta
etapa está mais em **latência de cauda** (p95/p99) e em **custo de CPU por requisição em
escala** (o que este benchmark não mede diretamente, mas o overhead de código por chamada
medido acima sugere) do que em latência típica percebida ponta a ponta, onde o overhead de
HTTP/FastAPI domina de qualquer forma.

### Por que o TF-IDF não está no grafo ONNX

A tentativa original — `skl2onnx.convert_sklearn` sobre o `Pipeline` inteiro — esbarrou em
duas limitações contornáveis e uma que não é, todas medidas contra os dados reais de test
(1.685 amostras), não supostas:

1. `strip_accents='unicode'` não é suportado pelo conversor de `TfidfVectorizer`.
   Contornável: strip de acentos em Python antes do ONNX Runtime, já que o vocabulário
   ajustado no treino reflete esse pré-processamento.
2. O nó `StringNormalizer` que o conversor gera para `lowercase=True` exige uma locale de
   sistema (`en_US.UTF-8` por padrão) ausente em imagens Debian/WSL sem `locale-gen`.
   Contornável: sobrescrever o atributo `locale` do nó para `C.utf8` (locale builtin do
   glibc, sem pacote extra).
3. **Não contornável:** o tokenizador do `skl2onnx` diverge do tokenizador do scikit-learn em
   ngrams e casos de borda (números, hífens, abreviações como "stage A"). Medido: até 12 das
   1.685 predições de classe trocavam de classe, e a diferença máxima de probabilidade
   passava de 0,10 — muito acima da tolerância de equivalência (1e-4). A causa raiz é a
   contagem de tokens por documento: o vetorizador ONNX consistentemente extrai menos ngrams
   que o sklearn no mesmo texto.

Diante do item 3, o TF-IDF ficou em Python, dentro de `OnnxPredictor`
(`src/models/predictor.py`) — os backends `onnx`/`onnx-int8` reabrem o mesmo
`pipeline.joblib` do backend `sklearn` só para a etapa de vetorização (e é exatamente isso
que infla o footprint total, ver acima), e só o classificador (`LogisticRegression`) vira
ONNX (`src/models/export_onnx.py`).

### Por que o classificador foi reconstruído à mão (Gemm + Softmax)

O conversor padrão do `skl2onnx` para `LogisticRegression` gera um nó `LinearClassifier` do
domínio `ai.onnx.ml` — equivalente ao sklearn com precisão de ~1e-7, mas **não é alvo da
quantização dinâmica do ONNX Runtime**: rodar `quantize_dynamic` sobre esse grafo produzia um
artefato "int8" do mesmo tamanho do fp32, sem ganho nenhum.

A solução foi reconstruir a etapa de classificação manualmente como dois operadores do
domínio padrão `ai.onnx` — `Gemm` (pesos) seguido de `Softmax` (probabilidades) —
matematicamente idênticos à regressão logística multinomial do scikit-learn
(`softmax(X @ coef_.T + intercept_)`, diferença numérica 0,0 nos testes) e, ao contrário do
`LinearClassifier`, um alvo válido para quantização.

## Equivalência entre backends — o que foi validado, e o que não é esperado ser idêntico

`src/models/export_onnx.py` valida dois critérios diferentes, de propósito:

- **fp32 vs. sklearn:** equivalência estrita. Tolerância de 1e-4 na probabilidade máxima e
  **zero** predições trocadas nas 1.685 amostras de test — é o que a exportação manual
  (Gemm + Softmax) garante por construção.
- **int8 vs. sklearn:** gate de degradação, não de equivalência. Quantização dinâmica troca
  precisão numérica por tamanho/latência **por desenho** — exigir "idêntico ao fp32" anularia
  o próprio ponto de quantizar. O critério adotado é a queda de macro-F1 no test real ficar
  abaixo de 0,02; a medição atual (0,0009) fica bem dentro da margem.

Consequência prática: os 3 backends concordam na esmagadora maioria das predições (medido:
11/1.685 divergências do int8 frente ao sklearn), mas não é esperado — nem exigido — que
sejam bit-a-bit idênticos para o backend quantizado. `tests/test_predictor.py` cobre os 3
backends concordando na mesma entrada usando um pipeline sintético pequeno, onde as margens de
decisão são largas o bastante para não haver divergência nenhuma nos casos testados.
