"""Valida que o ambiente está configurado corretamente antes de rodar o pipeline."""

import importlib
import sys
from pathlib import Path

REQUIRED_PACKAGES = [
    "sklearn",
    "pandas",
    "pyarrow",
    "fastapi",
    "uvicorn",
    "prometheus_client",
    "pydantic_settings",
    "structlog",
    "skl2onnx",
    "onnxruntime",
    "httpx",
]

REQUIRED_DIRS = [
    "data/bronze",
    "data/silver",
    "data/gold",
    "models",
    "metrics",
    "configs",
]


def check_python_version() -> bool:
    ok = sys.version_info >= (3, 11)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] Python {sys.version_info.major}.{sys.version_info.minor}")
    return ok


def check_packages() -> bool:
    all_ok = True
    for pkg in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(pkg)
            version = getattr(mod, "__version__", "?")
            print(f"[OK]   {pkg} ({version})")
        except ImportError:
            print(f"[FAIL] {pkg} — não instalado")
            all_ok = False
    return all_ok


def check_directories() -> bool:
    root = Path(__file__).resolve().parents[1]
    all_ok = True
    for d in REQUIRED_DIRS:
        path = root / d
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"[CRIADO] {d}/")
        else:
            print(f"[OK]     {d}/")
    return all_ok


def check_env_file() -> bool:
    root = Path(__file__).resolve().parents[1]
    env_file = root / ".env"
    if not env_file.exists():
        print("[WARN]  .env não encontrado — copie .env.example para .env")
        return False
    print("[OK]   .env encontrado")
    return True


def main() -> None:
    print("=" * 50)
    print("Validação de Ambiente — Triagem Médica NLP")
    print("=" * 50)

    results = [
        check_python_version(),
        check_packages(),
        check_directories(),
        check_env_file(),
    ]

    print("=" * 50)
    if all(results):
        print("Ambiente OK — pronto para rodar o pipeline.")
        sys.exit(0)
    else:
        print("Ambiente com problemas — corrija os itens marcados [FAIL] ou [WARN].")
        sys.exit(1)


if __name__ == "__main__":
    main()
