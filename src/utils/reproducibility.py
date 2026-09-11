import random

import numpy as np

from src.utils.config import settings


def set_global_seed(seed: int | None = None) -> None:
    """Fixa a seed global. Chamar antes de qualquer split ou treino."""
    s = seed if seed is not None else settings.seed
    random.seed(s)
    np.random.seed(s)
