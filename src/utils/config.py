"""Configuração central do projeto — Pydantic Settings lidas de ambiente/.env.

Todo caminho usado pelo projeto (dados, modelos, métricas) deriva de PROJECT_ROOT aqui,
nunca de caminho relativo ao cwd — a task da DAG do Airflow roda com cwd diferente
(ver ARQUITETURA.md §6.3, regra 11).
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    seed: int = 42
    log_level: str = "INFO"

    # Strategy de inferência selecionável em runtime (src/models/predictor.py, B3/B4)
    model_backend: str = "sklearn"  # sklearn | onnx | onnx-int8

    # Gate de qualidade do treino (src/models/evaluate.py, B2)
    min_macro_f1: float = 0.50

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_BRONZE_DIR = PROJECT_ROOT / "data" / "bronze"
DATA_SILVER_DIR = PROJECT_ROOT / "data" / "silver"
DATA_GOLD_DIR = PROJECT_ROOT / "data" / "gold"

MODELS_DIR = PROJECT_ROOT / "models"
MODELS_STAGING_DIR = MODELS_DIR / "staging"
MODELS_CURRENT_DIR = MODELS_DIR / "current"

METRICS_DIR = PROJECT_ROOT / "metrics"
