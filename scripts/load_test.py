"""Gerador de carga fim a fim — `httpx` assíncrono contra `/predict`, amostrando laudos reais
do conjunto de test (nunca string sintética, mesma regra de `benchmark_latency.py`).

Existe por dois motivos, nenhum deles é medir inferência pura:
1. Popular os painéis do Grafana para a gravação do vídeo — sem tráfego, os 6 painéis
   ficam vazios.
2. Medir latência **fim a fim** (HTTP + serialização + FastAPI) sob concorrência real —
   diferente da medição isolada de `scripts/benchmark_latency.py`. Confundir as duas é o erro
   clássico a evitar: a diferença entre elas é justamente o overhead de rede/framework.

Uso:
    poetry run python scripts/load_test.py --rps 10 --duracao 60 --backend sklearn
    poetry run python scripts/load_test.py --rps 10 --duracao 60 --backend onnx
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx

from src.data.features import load_gold
from src.utils.logger import get_logger

logger = get_logger(__name__)

PORTA_POR_BACKEND = {"sklearn": 8000, "onnx": 8001}


def _amostras_reais(n: int) -> list[str]:
    """`n` textos do conjunto de test, em ciclo se `n` exceder o tamanho do test — mesma regra
    de `benchmark_latency.py`: todo texto disparado veio de um laudo de verdade."""
    _train, _val, test, _metadata = load_gold()
    textos = test["texto"].tolist()
    return [textos[i % len(textos)] for i in range(n)]


def _percentil(valores_ordenados: list[float], p: float) -> float:
    idx = min(int(len(valores_ordenados) * p), len(valores_ordenados) - 1)
    return valores_ordenados[idx]


async def _disparar(
    cliente: httpx.AsyncClient, base_url: str, texto: str, resultados: list[dict]
) -> None:
    inicio = time.perf_counter()
    try:
        resp = await cliente.post(f"{base_url}/predict", json={"texto": texto})
        latencia_ms = (time.perf_counter() - inicio) * 1000
        resultados.append({"latencia_ms": latencia_ms, "status_code": resp.status_code})
    except httpx.HTTPError as exc:
        latencia_ms = (time.perf_counter() - inicio) * 1000
        resultados.append({"latencia_ms": latencia_ms, "status_code": None, "erro": str(exc)})


async def gerar_carga(base_url: str, rps: int, duracao_s: int) -> dict:
    """Dispara `rps` requisições por segundo por `duracao_s` segundos, sem esperar a resposta
    de uma requisição antes de disparar a próxima do mesmo segundo — é isso que testa
    concorrência de verdade, em vez de um cliente sequencial."""
    n_total = rps * duracao_s
    amostras = _amostras_reais(n_total)
    resultados: list[dict] = []

    async with httpx.AsyncClient(timeout=10.0) as cliente:
        indice = 0
        for _segundo in range(duracao_s):
            inicio_segundo = time.perf_counter()
            tarefas = [
                _disparar(cliente, base_url, amostras[indice + i], resultados) for i in range(rps)
            ]
            indice += rps
            await asyncio.gather(*tarefas)
            decorrido = time.perf_counter() - inicio_segundo
            if decorrido < 1.0:
                await asyncio.sleep(1.0 - decorrido)

    latencias_ms = sorted(r["latencia_ms"] for r in resultados)
    n_erros = sum(1 for r in resultados if r.get("status_code") != 200)
    resumo = {
        "n_requisicoes": len(resultados),
        "n_erros": n_erros,
        "taxa_erro": round(n_erros / len(resultados), 4) if resultados else 0.0,
        "p50_ms": round(_percentil(latencias_ms, 0.50), 3),
        "p95_ms": round(_percentil(latencias_ms, 0.95), 3),
        "p99_ms": round(_percentil(latencias_ms, 0.99), 3),
        "media_ms": round(statistics.mean(latencias_ms), 3) if latencias_ms else 0.0,
    }
    logger.info("load_test_concluido", base_url=base_url, rps=rps, duracao_s=duracao_s, **resumo)
    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(description="Gerador de carga fim a fim contra /predict.")
    parser.add_argument("--rps", type=int, default=10, help="Requisições por segundo.")
    parser.add_argument("--duracao", type=int, default=30, help="Duração em segundos.")
    parser.add_argument(
        "--backend",
        default="sklearn",
        choices=sorted(PORTA_POR_BACKEND),
        help="Qual API atacar — sklearn (porta 8000) ou onnx (porta 8001).",
    )
    parser.add_argument(
        "--host", default="localhost", help="Host onde as APIs estão escutando (default localhost)."
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{PORTA_POR_BACKEND[args.backend]}"
    resumo = asyncio.run(gerar_carga(base_url, args.rps, args.duracao))
    print(
        f"{args.backend}: {resumo['n_requisicoes']} requisições, "
        f"{resumo['n_erros']} erros ({resumo['taxa_erro']:.1%}) — "
        f"p50 {resumo['p50_ms']} ms · p95 {resumo['p95_ms']} ms · p99 {resumo['p99_ms']} ms"
    )


if __name__ == "__main__":
    main()
