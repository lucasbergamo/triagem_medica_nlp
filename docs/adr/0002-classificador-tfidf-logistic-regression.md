# ADR-0002 — Classificador: TF-IDF + Logistic Regression, não Random Forest

## Status

Aceito.

## Contexto

O critério de maior peso deste projeto é o ciclo de vida do modelo em produção (deploy,
monitoramento, CI/CD, orquestração), não a acurácia do classificador — a modelagem só precisa
ser boa o bastante para a demonstração ser honesta e leve o bastante para a otimização de
latência (exportação ONNX, quantização) ter uma história real para contar. Essa restrição pesa
na escolha do algoritmo tanto quanto a métrica de qualidade.

## Decisão

Pipeline de dois estágios (`src/models/train.py`):

```
TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000,
                sublinear_tf=True, strip_accents="unicode", lowercase=True)
→ LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0)
```

Random Forest — o algoritmo usado como exemplo na disciplina — entra como **baseline de
comparação** (`train.py --baseline`), não como candidato principal:

| Modelo | Macro-F1 (test) | Tempo de treino | Tamanho do artefato |
|---|---|---|---|
| LogisticRegression (escolhido) | 0,7562 | 8,08 s | 3,1 MB |
| RandomForest (baseline) | 0,7296 | 4,22 s | 32,6 MB |

RF treina mais rápido neste corpus, mas perde em macro-F1 e produz um artefato ~10× maior — o
tamanho vem do número de árvores serializadas. Exportado para ONNX, cada estimador de uma
floresta vira uma subárvore serializada; o arquivo resultante fica na casa de dezenas de MB e a
inferência via ONNX Runtime pode ficar **mais lenta** que a árvore nativa do scikit-learn,
destruindo a história de latência que o projeto existe para contar. LogisticRegression sobre um
vetor TF-IDF converte a inferência inteira num único `Gemm` (multiplicação de matriz) —
exatamente o padrão em que ONNX Runtime ganha desempenho (ver ADR-0004).

## Consequências

- A escolha do modelo é subordinada à otimização de latência, não o contrário — decisão
  registrada aqui para não parecer, num código isolado, uma escolha arbitrária de acurácia.
- O baseline RF fica no repositório (`train.py --baseline`) como evidência de que a escolha foi
  medida, não assumida — comparação completa em `docs/model_card.md`.
- Expectativa de macro-F1 realista para este corpus e este proxy de rótulo (ADR-0001) era
  0,55–0,68; o resultado medido (0,7562) superou a expectativa — análise da causa provável em
  `docs/model_card.md`.
