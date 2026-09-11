"""ModelStore — abstração de onde os artefatos de modelo moram.

`LocalModelStore` hoje; `S3ModelStore` entra junto com o deploy AWS, selecionado por
variável de ambiente. Uma interface, duas implementações — o resto do código nunca sabe
onde o modelo mora, só que existe um `ModelStore` que copia e verifica artefatos.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelStore(Protocol):
    def copiar(self, origem: Path, destino_nome: str) -> None: ...

    def existe(self, nome: str) -> bool: ...


class LocalModelStore:
    """Copia arquivos entre diretórios locais — implementação usada até o deploy em nuvem."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

    def copiar(self, origem: Path, destino_nome: str) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        destino = self.base_dir / destino_nome
        shutil.copy2(origem, destino)
        logger.info("artefato_copiado", origem=str(origem), destino=str(destino))

    def existe(self, nome: str) -> bool:
        return (self.base_dir / nome).exists()
