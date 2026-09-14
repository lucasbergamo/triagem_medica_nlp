"""Gera airflow/.env a partir de airflow/.env.example na primeira vez que alguém sobe o
Airflow — sem isso, quem clona o repo precisa copiar o arquivo e gerar 3 segredos à mão antes
de conseguir ver a DAG rodando. Não sobrescreve um airflow/.env já existente.
"""

import base64
import os
import secrets
from pathlib import Path

AIRFLOW_DIR = Path(__file__).resolve().parent.parent / "airflow"
ENV_EXAMPLE = AIRFLOW_DIR / ".env.example"
ENV_FILE = AIRFLOW_DIR / ".env"


def gerar_segredos() -> dict[str, str]:
    return {
        "POSTGRES_PASSWORD": secrets.token_urlsafe(32),
        "AIRFLOW_ADMIN_PASSWORD": secrets.token_urlsafe(32),
        "AIRFLOW__CORE__FERNET_KEY": base64.urlsafe_b64encode(os.urandom(32)).decode(),
        "AIRFLOW__API_AUTH__JWT_SECRET": secrets.token_urlsafe(32),
        "AIRFLOW__API__SECRET_KEY": secrets.token_urlsafe(32),
    }


def main() -> None:
    if ENV_FILE.exists():
        print(f"{ENV_FILE} já existe — nada a fazer.")
        return

    segredos = gerar_segredos()
    linhas_geradas = []
    for linha in ENV_EXAMPLE.read_text(encoding="utf-8").splitlines():
        chave, separador, valor = linha.partition("=")
        if separador and chave in segredos and valor == "":
            linha = f"{chave}={segredos[chave]}"
        linhas_geradas.append(linha)

    ENV_FILE.write_text("\n".join(linhas_geradas) + "\n", encoding="utf-8")
    print(
        f"{ENV_FILE} criado com segredos gerados automaticamente — "
        "a senha do admin da UI está lá dentro."
    )


if __name__ == "__main__":
    main()
