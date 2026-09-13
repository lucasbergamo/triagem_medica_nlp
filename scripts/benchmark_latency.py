"""Baseline de latência de inferência pura.

Mede só o tempo de `predictor.predict()`, sem HTTP nem serialização — medição fim a fim
(incluindo FastAPI) é responsabilidade de um script separado, para não confundir as duas.
Amostras são textos reais do conjunto de test, nunca string sintética. Warmup descartado
(JIT do scipy/cache de alocador), 1000 medições, percentis calculados sobre
`time.perf_counter_ns`.

Dois modos:
- `--backend <nome>` (default `sklearn`): mede um único backend e grava/sobrescreve
  `metrics/latency_baseline.json` — o mesmo caminho do bloco de API, mantido para não quebrar
  `make benchmark`. Não usar com `onnx`/`onnx-int8`: sobrescreveria a baseline congelada do
  sklearn com números de outro backend.
- `--comparar`: mede os 3 backends (sklearn, onnx, onnx-int8), decompõe `predict()` em
  vetorização (TF-IDF) + classificação, e mede o tamanho de cada componente isolado. Grava
  tudo em `metrics/latency_benchmark.json`. **Nunca** toca em `latency_baseline.json` — essa é
  a baseline versionada no bloco de API, contra a qual as medições aqui servem de comparação,
  não de substituição.

A decomposição existe porque a tabela de `predict()` completo, sozinha, é enganosa: o TF-IDF
é Python puro nos 3 backends (ver `src/models/predictor.py::OnnxPredictor`), e é ele quem
domina o tempo — o classificador, que é a única etapa que os backends onnx/onnx-int8 realmente
trocam de motor, é uma fração pequena do total. Sem separar as duas etapas, um ganho real e
grande no classificador (medido abaixo) aparece como um ganho pequeno no fim a fim, sem
explicação. Pelo mesmo motivo, `tamanhos_por_componente()` separa o peso do vetorizador do
peso do classificador: o arquivo `pipeline.joblib` no disco empacota os dois juntos, e reportar
o tamanho desse arquivo como "o artefato do backend onnx" superestimaria a redução — o backend
onnx ainda depende desse mesmo arquivo para o TF-IDF.
"""

import argparse
import io
import json
import statistics
import time
from pathlib import Path

import joblib
import numpy as np
from onnxruntime import InferenceSession

from src.data.features import load_gold
from src.models.predictor import (
    NOME_ARTEFATO_ONNX,
    NOME_ARTEFATO_ONNX_INT8,
    NOME_ARTEFATO_SKLEARN,
    get_predictor,
)
from src.utils.config import METRICS_DIR, MODELS_CURRENT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

N_WARMUP = 200
N_MEDICOES = 1000
NOME_SAIDA = "latency_baseline.json"
NOME_SAIDA_COMPARATIVO = "latency_benchmark.json"
BACKENDS_COMPARADOS = ["sklearn", "onnx", "onnx-int8"]

ARTEFATOS_POR_BACKEND = {
    "sklearn": NOME_ARTEFATO_SKLEARN,
    "onnx": NOME_ARTEFATO_ONNX,
    "onnx-int8": NOME_ARTEFATO_ONNX_INT8,
}


def _amostras_reais(n: int) -> list[str]:
    """`n` textos do conjunto de test — repete a amostra em ciclo se `n` exceder o tamanho do
    test, mas nunca gera string sintética: todo texto medido veio de um laudo de verdade."""
    _train, _val, test, _metadata = load_gold()
    textos = test["texto"].tolist()
    return [textos[i % len(textos)] for i in range(n)]


def _percentil(valores_ordenados: list[float], p: float) -> float:
    idx = min(int(len(valores_ordenados) * p), len(valores_ordenados) - 1)
    return valores_ordenados[idx]


def _tamanho_artefato_kb(backend: str) -> float | None:
    nome = ARTEFATOS_POR_BACKEND.get(backend)
    if nome is None:
        return None
    path = MODELS_CURRENT_DIR / nome
    return round(path.stat().st_size / 1024, 1) if path.exists() else None


def _resumo(latencias_ms: list[float]) -> dict:
    ordenadas = sorted(latencias_ms)
    return {
        "p50_ms": round(_percentil(ordenadas, 0.50), 4),
        "p95_ms": round(_percentil(ordenadas, 0.95), 4),
        "p99_ms": round(_percentil(ordenadas, 0.99), 4),
        "media_ms": round(statistics.mean(latencias_ms), 4),
        "desvio_padrao_ms": round(statistics.stdev(latencias_ms), 4),
    }


def medir(backend: str = "sklearn") -> dict:
    predictor = get_predictor(backend)
    amostras = _amostras_reais(N_WARMUP + N_MEDICOES)

    for texto in amostras[:N_WARMUP]:
        predictor.predict([texto])

    latencias_ms = []
    for texto in amostras[N_WARMUP:]:
        inicio = time.perf_counter_ns()
        predictor.predict([texto])
        latencias_ms.append((time.perf_counter_ns() - inicio) / 1_000_000)

    resultado = {
        "backend": backend,
        "n_medicoes": N_MEDICOES,
        "n_warmup": N_WARMUP,
        **_resumo(latencias_ms),
        "tamanho_artefato_kb": _tamanho_artefato_kb(backend),
    }
    logger.info("benchmark_concluido", **resultado)
    return resultado


def _tamanho_serializado_kb(obj) -> float:
    """Serializa `obj` isolado em memória — nunca em disco, não é um artefato do projeto, só
    uma medição — para saber quanto cada componente pesa sozinho. `pipeline.joblib` empacota
    vetorizador e classificador juntos; sem isso não dá pra separar quanto é de cada um."""
    buffer = io.BytesIO()
    joblib.dump(obj, buffer)
    return round(len(buffer.getvalue()) / 1024, 1)


def tamanhos_por_componente() -> dict:
    """Tamanho isolado do vetorizador e de cada classificador, e o footprint total em disco
    de cada backend. O footprint dos backends onnx/onnx-int8 não é só o artefato `.onnx`: eles
    ainda carregam `pipeline.joblib` inteiro — vetorizador **e** o classificador sklearn, esse
    último sem uso — só para ter o TF-IDF (`src/models/predictor.py::OnnxPredictor`). O
    footprint deles é maior que o do sklearn, não menor."""
    pipeline = joblib.load(MODELS_CURRENT_DIR / NOME_ARTEFATO_SKLEARN)
    vetorizador_kb = _tamanho_serializado_kb(pipeline.named_steps["tfidf"])
    classificador_sklearn_kb = _tamanho_serializado_kb(pipeline.named_steps["clf"])
    pipeline_kb = _tamanho_artefato_kb("sklearn")
    onnx_kb = _tamanho_artefato_kb("onnx")
    int8_kb = _tamanho_artefato_kb("onnx-int8")

    return {
        "vetorizador_tfidf_kb": vetorizador_kb,
        "classificador_kb": {
            "sklearn": classificador_sklearn_kb,
            "onnx": onnx_kb,
            "onnx-int8": int8_kb,
        },
        "footprint_total_kb": {
            "sklearn": pipeline_kb,
            "onnx": round(pipeline_kb + onnx_kb, 1),
            "onnx-int8": round(pipeline_kb + int8_kb, 1),
        },
    }


def medir_vetorizacao(tfidf) -> dict:
    """TF-IDF isolado — a etapa comum aos 3 backends, medida uma única vez."""
    amostras = _amostras_reais(N_WARMUP + N_MEDICOES)
    for texto in amostras[:N_WARMUP]:
        tfidf.transform([texto])

    latencias_ms = []
    for texto in amostras[N_WARMUP:]:
        inicio = time.perf_counter_ns()
        tfidf.transform([texto])
        latencias_ms.append((time.perf_counter_ns() - inicio) / 1_000_000)
    return _resumo(latencias_ms)


def medir_classificador(nome: str, tfidf, classificar) -> dict:
    """Classificador isolado. A vetorização acontece **antes** do cronômetro começar — ela já
    tem sua própria medição em `medir_vetorizacao`; contá-la aqui de novo inflaria os três
    backends pelo mesmo tanto e escoderia a diferença real entre eles."""
    amostras = _amostras_reais(N_WARMUP + N_MEDICOES)
    entradas = [tfidf.transform([texto]) for texto in amostras]

    for x in entradas[:N_WARMUP]:
        classificar(x)

    latencias_ms = []
    for x in entradas[N_WARMUP:]:
        inicio = time.perf_counter_ns()
        classificar(x)
        latencias_ms.append((time.perf_counter_ns() - inicio) / 1_000_000)
    return {"backend": nome, **_resumo(latencias_ms)}


def medir_decomposicao() -> dict:
    """Decompõe `predict()` em vetorização (TF-IDF) + classificação — a etapa que os backends
    onnx/onnx-int8 realmente trocam de motor de execução. `classificar` recebe a saída esparsa
    do TF-IDF: para sklearn isso já é a entrada de `predict_proba`; para onnx/onnx-int8, a
    densificação (`toarray().astype(np.float32)`) entra no tempo medido do classificador
    porque é trabalho real que `OnnxPredictor.predict()` faz por requisição — não é grátis."""
    pipeline = joblib.load(MODELS_CURRENT_DIR / NOME_ARTEFATO_SKLEARN)
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    sessao_onnx = InferenceSession(
        str(MODELS_CURRENT_DIR / NOME_ARTEFATO_ONNX), providers=["CPUExecutionProvider"]
    )
    sessao_int8 = InferenceSession(
        str(MODELS_CURRENT_DIR / NOME_ARTEFATO_ONNX_INT8), providers=["CPUExecutionProvider"]
    )

    def _classificar_onnx(sessao: InferenceSession):
        entrada = sessao.get_inputs()[0].name

        def _fn(x_esparso):
            x_denso = x_esparso.toarray().astype(np.float32)
            sessao.run(None, {entrada: x_denso})

        return _fn

    vetorizacao = medir_vetorizacao(tfidf)
    classificador = {
        "sklearn": medir_classificador("sklearn", tfidf, lambda x: clf.predict_proba(x)),
        "onnx": medir_classificador("onnx", tfidf, _classificar_onnx(sessao_onnx)),
        "onnx-int8": medir_classificador("onnx-int8", tfidf, _classificar_onnx(sessao_int8)),
    }
    logger.info(
        "decomposicao_concluida",
        vetorizacao_p50_ms=vetorizacao["p50_ms"],
        classificador_sklearn_p50_ms=classificador["sklearn"]["p50_ms"],
        classificador_onnx_p50_ms=classificador["onnx"]["p50_ms"],
        classificador_int8_p50_ms=classificador["onnx-int8"]["p50_ms"],
    )
    return {"vetorizacao_tfidf": vetorizacao, "classificador": classificador}


def salvar(resultado: dict) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / NOME_SAIDA
    path.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("baseline_salva", path=str(path))
    return path


def medir_comparativo() -> dict:
    """Mede os 3 backends com a mesma metodologia de `medir()`, decompõe cada `predict()` em
    vetorização + classificação, e mede o tamanho de cada componente isolado — as três
    perguntas que, juntas, explicam o ganho: quanto cada backend leva de ponta a ponta, onde
    o tempo é gasto dentro desse total, e quanto cada peça pesa em disco."""
    predict_completo = {backend: medir(backend) for backend in BACKENDS_COMPARADOS}
    referencia_p95 = predict_completo["sklearn"]["p95_ms"]
    for resultado in predict_completo.values():
        resultado["speedup_p95_vs_sklearn"] = round(referencia_p95 / resultado["p95_ms"], 2)

    return {
        "predict_completo": predict_completo,
        "decomposicao": medir_decomposicao(),
        "tamanhos": tamanhos_por_componente(),
    }


def salvar_comparativo(resultados: dict) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / NOME_SAIDA_COMPARATIVO
    path.write_text(json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("benchmark_comparativo_salvo", path=str(path))
    return path


def tabela_markdown_comparativa(resultados: dict) -> str:
    """Só `predict()` completo (vetorização + classificação). Tamanho de artefato não entra
    aqui de propósito — ver `tabela_markdown_tamanhos`, que separa vetorizador de
    classificador em vez de misturar os dois num único número por backend."""
    predict_completo = resultados["predict_completo"]
    cabecalho = "| Backend | p50 (ms) | p95 (ms) | p99 (ms) | Speedup p95 |"
    linhas = [cabecalho, "|---|---|---|---|---|"]
    for backend in BACKENDS_COMPARADOS:
        r = predict_completo[backend]
        linhas.append(
            f"| {backend} | {r['p50_ms']} | {r['p95_ms']} | {r['p99_ms']} | "
            f"{r['speedup_p95_vs_sklearn']}x |"
        )
    return "\n".join(linhas)


def tabela_markdown_decomposicao(decomposicao: dict) -> str:
    v = decomposicao["vetorizacao_tfidf"]
    linhas = [
        "| Etapa | p50 (ms) | p95 (ms) |",
        "|---|---|---|",
        f"| Vetorização TF-IDF (comum aos 3 backends) | {v['p50_ms']} | {v['p95_ms']} |",
    ]
    for backend in BACKENDS_COMPARADOS:
        c = decomposicao["classificador"][backend]
        linhas.append(f"| Classificador ({backend}) | {c['p50_ms']} | {c['p95_ms']} |")
    return "\n".join(linhas)


def tabela_markdown_tamanhos(tamanhos: dict) -> str:
    ft = tamanhos["footprint_total_kb"]
    linhas = [
        "| Componente | Tamanho |",
        "|---|---|",
        f"| Vetorizador TF-IDF (comum aos 3 backends) | {tamanhos['vetorizador_tfidf_kb']} KB |",
        f"| Classificador sklearn | {tamanhos['classificador_kb']['sklearn']} KB |",
        f"| Classificador onnx (fp32) | {tamanhos['classificador_kb']['onnx']} KB |",
        f"| Classificador onnx-int8 | {tamanhos['classificador_kb']['onnx-int8']} KB |",
        "",
        "| Backend | Footprint total em disco | Composição |",
        "|---|---|---|",
        f"| sklearn | {ft['sklearn']} KB | pipeline.joblib (vetorizador + classificador) |",
        f"| onnx | {ft['onnx']} KB | pipeline.joblib inteiro (p/ o TF-IDF) + pipeline.onnx |",
        f"| onnx-int8 | {ft['onnx-int8']} KB | pipeline.joblib inteiro (p/ o TF-IDF) "
        "+ pipeline.int8.onnx |",
    ]
    return "\n".join(linhas)


def tabela_markdown(resultado: dict) -> str:
    cabecalho = (
        "| Backend | p50 (ms) | p95 (ms) | p99 (ms) | Média (ms) | Desvio (ms) | Artefato (KB) |"
    )
    return "\n".join(
        [
            cabecalho,
            "|---|---|---|---|---|---|---|",
            (
                f"| {resultado['backend']} | {resultado['p50_ms']} | {resultado['p95_ms']} | "
                f"{resultado['p99_ms']} | {resultado['media_ms']} | "
                f"{resultado['desvio_padrao_ms']} | {resultado['tamanho_artefato_kb']} |"
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline de latência de inferência pura.")
    parser.add_argument(
        "--backend",
        default="sklearn",
        help="Backend a medir isoladamente — grava/sobrescreve latency_baseline.json.",
    )
    parser.add_argument(
        "--comparar",
        action="store_true",
        help=(
            "Mede sklearn, onnx e onnx-int8 e grava metrics/latency_benchmark.json "
            "com o speedup relativo. Nunca sobrescreve latency_baseline.json."
        ),
    )
    args = parser.parse_args()

    if args.comparar:
        resultados = medir_comparativo()
        salvar_comparativo(resultados)
        print(tabela_markdown_comparativa(resultados))
        print()
        print(tabela_markdown_decomposicao(resultados["decomposicao"]))
        print()
        print(tabela_markdown_tamanhos(resultados["tamanhos"]))
        return

    resultado = medir(args.backend)
    salvar(resultado)
    print(tabela_markdown(resultado))


if __name__ == "__main__":
    main()
