# Dataset — Medical Abstracts TC Corpus

## Origem e licença

- **Fonte:** [`sebischair/Medical-Abstracts-TC-Corpus`](https://github.com/sebischair/Medical-Abstracts-TC-Corpus), via GitHub raw — `https://raw.githubusercontent.com/sebischair/Medical-Abstracts-TC-Corpus/main/`.
- **Licença:** Creative Commons Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0), conforme `LICENSE` do repositório de origem.
- **Acesso:** público, sem login, sem API key, sem credenciamento (Kaggle ou similar). Verificado em 01/09/2026, confirmado em execução em 11/09/2026 — `HTTP 200` direto.
- **Arquivos:** `medical_tc_train.csv` (11.550 linhas), `medical_tc_test.csv` (2.888 linhas), `medical_tc_labels.csv` (5 linhas) — 18 MB no total. Baixados para `data/bronze/` por `src/data/load.py` e comitados no git (sem DVC — o volume é pequeno o bastante para não justificar a dependência extra, e garante que um clone reproduza os dados sem precisar baixar nada).
- **Colunas do bronze:** `condition_label` (inteiro, 1–5) e `medical_abstract` (texto livre); `medical_tc_labels.csv` traduz `condition_label` para `condition_name`.

## O rótulo de urgência é um proxy — não um rótulo clínico real

O corpus rotula **categoria clínica**, não urgência de triagem. Nenhum dataset público de
triagem com rótulo de urgência real está acessível sem credenciamento. A saída adotada é um
**proxy determinístico e auditável**, aplicado em `src/data/urgency_map.py`:

| `condition_label` | Categoria (`condition_name`) | Urgência (proxy) | Justificativa clínica declarada |
|---|---|---|---|
| 1 | neoplasms | **urgente** | Suspeita oncológica exige estadiamento e conduta em janela curta |
| 4 | cardiovascular diseases | **urgente** | Evento cardiovascular é tempo-dependente por definição |
| 3 | nervous system diseases | **atencao** | Heterogêneo: de cefaleia crônica a AVC — priorização intermediária |
| 2 | digestive system diseases | **atencao** | Majoritariamente subagudo, com exceções que a triagem humana resolve |
| 5 | general pathological conditions | **normal** | Categoria residual, sem marcador de tempo-dependência |

> O rótulo de urgência é um *proxy* derivado por regra determinística a partir da categoria
> clínica do corpus. Não é um rótulo de triagem validado clinicamente e não deve ser usado
> para decisão médica. A escolha está documentada porque o objetivo deste projeto é o ciclo
> de vida do modelo em produção; um rótulo real, quando disponível, substitui o mapa em
> `src/data/urgency_map.py` sem alterar nenhuma outra camada do sistema.

## Achado de dado — o corpus é multi-rótulo achatado em linhas

Não estava previsto no desenho original do pipeline e apareceu só ao implementar: **2.929
abstracts do corpus aparecem repetidos**, cada cópia com um `condition_label` diferente — 6.140 linhas ao
todo (3.211 "a mais" sobre o total de textos únicos). Não é ruído de anotação nem duplicata
no sentido usual: é o mesmo abstract anotado pelo autor original sob **mais de uma categoria
clínica**, achatado em uma linha por combinação em vez de guardado como multi-rótulo.

As combinações mais frequentes entre as 2.929 (22 combinações distintas ao todo):

| Combinação (`condition_label`) | Categorias | Grupos |
|---|---|---|
| (4, 5) | cardiovascular + general pathological | 738 |
| (1, 5) | neoplasms + general pathological | 486 |
| (2, 5) | digestive + general pathological | 475 |
| (3, 5) | nervous system + general pathological | 446 |
| (1, 3) | neoplasms + nervous system | 139 |
| (1, 2) | neoplasms + digestive | 134 |
| (3, 4) | nervous system + cardiovascular | 114 |
| — | outras 15 combinações (3 a 4 rótulos) | 397 |

Padrão visível na tabela: a maioria dos pares envolve a categoria 5 (*general pathological
conditions*, residual) somada a uma categoria específica — o autor anotou o caso concreto
**e também** o guarda-chuva residual.

### A regra: a maior urgência vence

Resolver isso com dedup ingênuo (`keep="first"`, manter a primeira linha lida) sub-triaria
sistematicamente: descartaria a cópia "urgente" (`4: cardiovascular`) em favor da cópia
"normal" (`5: general pathological`) só por causa da ordem em que `train.csv` e `test.csv`
são lidos — 1.446 laudos teriam saído com urgência mais baixa do que o corpus permite.

Regra adotada em 11/09/2026 (`src/data/urgency_map.py::RANK_URGENCIA` +
`src/data/preprocess.py::_resolver_grupo`): **entre os rótulos do mesmo texto, vence o de
maior urgência** (`normal < atencao < urgente`); empate entre dois rótulos da mesma urgência
é desempatado pelo menor `condition_label` — determinístico e independente da ordem das
linhas no arquivo. Duas justificativas:

1. **A categoria específica vence a categoria residual.** Nos 4 pares mais comuns da tabela
   acima, o autor anotou o abstract tanto pela condição concreta quanto pela categoria 5
   (`general pathological conditions`, sem marcador de tempo-dependência por definição). A
   categoria específica é a informação clinicamente relevante.
2. **Em triagem, a dúvida escala.** Quando o próprio corpus não sabe decidir entre duas
   categorias, o custo de errar para baixo (perder um caso urgente) é maior que o custo de
   errar para cima — é por isso que macro-F1, não acurácia, é a métrica principal de
   avaliação deste projeto. Diante de ambiguidade real na fonte, a política do sistema
   escala, não abaixa.

### Os splits originais do corpus se sobrepõem — mais um motivo para o re-split próprio

988 textos do corpus aparecem **tanto no `medical_tc_train.csv` quanto no
`medical_tc_test.csv`** originais — provavelmente pela mesma causa do multi-rótulo (a mesma
publicação, anotada sob categorias diferentes, caiu em splits diferentes do autor). Usar o
split original do autor vazaria treino para teste. Confirma a decisão já tomada na
§"Camadas" abaixo: os dois arquivos são combinados e o split 70/15/15 deste projeto é feito
do zero, na camada gold, depois da resolução de multi-rótulo — nunca o split original do
corpus.

## Camadas

```
data/bronze/                    data/silver/                 data/gold/
medical_tc_{train,test}.csv →   laudos.parquet           →   {train,val,test}.parquet
medical_tc_labels.csv           texto, categoria_clinica,    + metadata.parquet
                                 urgencia, n_caracteres
load.py                         preprocess.py                 features.py
```

- **bronze** — os 3 CSVs originais, imutáveis, `train.csv` e `test.csv` do corpus combinados
  aqui (não usamos o split original do autor — o split deste projeto é o da camada gold).
- **silver** (`laudos.parquet`) — bronze combinado, sem nulos, multi-rótulo resolvido por
  texto (maior urgência vence — ver seção acima), sem laudos com menos de 50 caracteres,
  com `categoria_clinica` e `urgencia` derivadas do `condition_label` vencedor.
- **gold** — `train.parquet` / `val.parquet` / `test.parquet`, split estratificado por
  `urgencia` em 70/15/15 com a seed global (`src/utils/reproducibility.py`).
  `metadata.parquet` guarda `n_classes`, `classes`, as 3 contagens e um hash SHA-256 (16
  caracteres) do conteúdo do split, para detectar se os dados mudaram entre execuções.

## Distribuição

**Bruta, no `medical_tc_train.csv` original** — estimativa inicial, calculada antes da
resolução multi-rótulo (seção acima), portanto ilustrativa do desbalanceamento de origem,
não a distribuição final deste projeto: urgente 4.971 · normal 3.844 · atenção 2.735.

**Distribuição final, depois de bronze combinado (train+test) → silver com a resolução de
multi-rótulo** (14.438 → 11.227 linhas; `nulos_removidos=0`,
`linhas_colapsadas_multirrotulo=3.211`, `curtas_removidas=0` — nenhum abstract do corpus tem
menos de 50 caracteres). Esta tabela substitui a estimativa inicial acima:

| Urgência | Linhas |
|---|---|
| urgente | 6.113 |
| atencao | 2.720 |
| normal | 2.394 |

A resolução por maior urgência desloca a distribuição na direção esperada: mais laudos
classificados como "urgente" do que uma dedup ingênua produziria (5.140 com `keep="first"`
— 1.446 casos que sairiam sub-triados, ver seção acima) e menos como "normal" (a categoria
residual perde nos empates contra qualquer categoria específica).

**Na camada gold**, a proporção é preservada nos 3 splits pelo `stratify` (medido na
execução de 11/09/2026, `split_hash=7e8de5fa1d2760a2`):

| Split | Linhas | urgente | atencao | normal |
|---|---|---|---|---|
| train | 7.858 | 54,45% | 24,23% | 21,32% |
| val | 1.684 | 54,45% | 24,23% | 21,32% |
| test | 1.685 | 54,42% | 24,21% | 21,36% |

Desbalanceamento moderado (≈2,55:1 entre "urgente" e "normal", maior que a estimativa
inicial de 1,8:1 calculada antes da resolução multi-rótulo), tratável com
`class_weight="balanced"` no treino — ainda não exige SMOTE nem reamostragem, mas é um
ponto para o model card observar: o modelo verá bem mais exemplos de "urgente" que de
"normal".
