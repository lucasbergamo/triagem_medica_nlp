# ADR-0004 — Exportação ONNX: classificador isolado, TF-IDF em Python

## Status

Aceito.

## Contexto

A otimização de latência é o critério de maior peso técnico deste projeto, e o plano original
(`src/models/export_onnx.py`) era exportar o **pipeline inteiro** — vetorizador TF-IDF incluído
— para ONNX, porque a tokenização rodando em C++ dentro do ONNX Runtime, em vez de Python, é de
onde viria o ganho maior: não "o modelo ficou menor", mas "o pré-processamento saiu do Python".

Na prática, a conversão do `Pipeline` completo via `skl2onnx` esbarrou em três problemas,
medidos contra as 1.685 amostras reais do conjunto de test, não supostos:

1. `strip_accents="unicode"` não é suportado pelo conversor do `TfidfVectorizer` — contornável.
2. O nó `StringNormalizer` gerado para `lowercase=True` exige uma locale de sistema ausente em
   imagens Debian/WSL sem `locale-gen` — contornável.
3. **Não contornável:** o tokenizador do `skl2onnx` diverge do tokenizador do scikit-learn em
   ngrams e casos de borda (números, hífens, abreviações). Até 12 das 1.685 predições trocavam
   de classe, e a diferença máxima de probabilidade passava de 0,10 — muito acima da tolerância
   de equivalência adotada (1e-4).

Um problema adicional apareceu no classificador: o conversor padrão do `skl2onnx` para
`LogisticRegression` gera um nó `LinearClassifier` (domínio `ai.onnx.ml`) que **não é alvo da
quantização dinâmica** do ONNX Runtime — `quantize_dynamic` sobre esse grafo produzia um
artefato "int8" do mesmo tamanho do fp32, sem ganho nenhum.

## Decisão

**O TF-IDF permanece em Python**, dentro de `OnnxPredictor` (`src/models/predictor.py`) — os
backends `onnx` e `onnx-int8` reabrem o mesmo `pipeline.joblib` do backend `sklearn` só para a
etapa de vetorização. Só o classificador vira ONNX, e é **reconstruído manualmente** como dois
operadores do domínio padrão `ai.onnx` — `Gemm` (pesos) seguido de `Softmax` — matematicamente
idênticos à regressão logística multinomial do scikit-learn e, ao contrário do
`LinearClassifier` gerado automaticamente, um alvo válido para quantização.

Dois critérios de equivalência diferentes, de propósito (`src/models/export_onnx.py`):

- **fp32 vs. sklearn:** equivalência estrita — tolerância de 1e-4 na probabilidade máxima e
  **zero** predições trocadas nas amostras de test.
- **int8 vs. sklearn:** gate de **degradação**, não de equivalência — quantização troca precisão
  por tamanho/latência por desenho; o critério é a queda de macro-F1 no test ficar abaixo de
  0,02 (medido: 0,0009).

A análise completa — metodologia, decomposição de onde o tempo é gasto, e por que o ganho fim a
fim é modesto mesmo com o classificador acelerando de verdade — está em `docs/latencia.md`.

## Consequências

- Este é um plano B assumido, não um fallback silencioso: o ganho de latência vem só da etapa de
  classificação, que nunca foi mais que ~16% do tempo total de inferência medido — o teto do
  ganho fim a fim é baixo por construção, não por um erro de implementação.
- O footprint em disco dos backends `onnx`/`onnx-int8` **não** encolhe em relação ao `sklearn`:
  eles ainda carregam o `pipeline.joblib` inteiro (vetorizador + um classificador sklearn nunca
  usado) só para reaproveitar o TF-IDF — ficam 5–18% maiores, não menores. Um artefato
  "vetorizador sozinho" resolveria isso; fica registrado como otimização pendente, não
  implementada.
- `tests/test_predictor.py` cobre os três backends concordando na mesma entrada, sobre um
  pipeline sintético pequeno onde as margens de decisão são largas o bastante para não haver
  divergência.
