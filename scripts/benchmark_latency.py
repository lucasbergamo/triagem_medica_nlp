"""Baseline de latência de inferência pura.

Mede só o tempo de `predictor.predict()`, sem HTTP nem serialização — medição fim a fim
(incluindo FastAPI) é responsabilidade de um script separado, para não confundir as duas.
Amostras são textos reais do conjunto de test, nunca string sintética. Warmup descartado
(JIT do scipy/cache de alocador), 1000 medições, percentis calculados sobre
`time.perf_counter_ns`.

Só o backend `sklearn` existe até aqui; `--backend` já aceita `onnx`/`onnx-int8` porque a
factory (`get_predictor`) já sabe rejeitá-los com uma mensagem clara — adicionar um backend
novo não muda uma linha deste script.

`metrics/latency_baseline.json` é a baseline versionada contra a qual medições futuras (outro
backend, outra otimização) podem ser comparadas para detectar regressão.
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from src.data.features import load_gold
from src.models.predictor import NOME_ARTEFATO_SKLEARN, get_predictor
from src.utils.config import METRICS_DIR, MODELS_CURRENT_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

N_WARMUP = 200
N_MEDICOES = 1000
NOME_SAIDA = "latency_baseline.json"

ARTEFATOS_POR_BACKEND = {"sklearn": NOME_ARTEFATO_SKLEARN}


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

    latencias_ordenadas = sorted(latencias_ms)

    resultado = {
        "backend": backend,
        "n_medicoes": N_MEDICOES,
        "n_warmup": N_WARMUP,
        "p50_ms": round(_percentil(latencias_ordenadas, 0.50), 4),
        "p95_ms": round(_percentil(latencias_ordenadas, 0.95), 4),
        "p99_ms": round(_percentil(latencias_ordenadas, 0.99), 4),
        "media_ms": round(statistics.mean(latencias_ms), 4),
        "desvio_padrao_ms": round(statistics.stdev(latencias_ms), 4),
        "tamanho_artefato_kb": _tamanho_artefato_kb(backend),
    }
    logger.info("benchmark_concluido", **resultado)
    return resultado


def salvar(resultado: dict) -> Path:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / NOME_SAIDA
    path.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("baseline_salva", path=str(path))
    return path


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
        "--backend", default="sklearn", help="Backend a medir (só 'sklearn' neste bloco)."
    )
    args = parser.parse_args()

    resultado = medir(args.backend)
    salvar(resultado)
    print(tabela_markdown(resultado))


if __name__ == "__main__":
    main()
