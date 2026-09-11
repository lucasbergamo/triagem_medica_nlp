"""Smoke test: verifica que src importa e que Settings carrega com os defaults certos."""


def test_import_src():
    import src  # noqa: F401


def test_settings_loads():
    from src.utils.config import settings

    assert settings.seed == 42
    assert settings.model_backend == "sklearn"
    assert 0.0 < settings.min_macro_f1 < 1.0


def test_project_paths_derive_from_root():
    from src.utils.config import DATA_BRONZE_DIR, PROJECT_ROOT

    assert DATA_BRONZE_DIR == PROJECT_ROOT / "data" / "bronze"


def test_get_logger_returns_bound_logger():
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    assert logger is not None


def test_set_global_seed_is_deterministic():
    import numpy as np

    from src.utils.reproducibility import set_global_seed

    set_global_seed(123)
    a = np.random.rand(3)
    set_global_seed(123)
    b = np.random.rand(3)
    assert (a == b).all()
