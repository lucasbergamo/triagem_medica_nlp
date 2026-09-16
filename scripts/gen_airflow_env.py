"""Gera airflow/.env a partir de airflow/.env.example na primeira vez que alguém sobe o
Airflow. Sem isso, quem clona o repo precisa copiar o arquivo e gerar 3 chaves criptográficas
à mão antes de conseguir ver a DAG rodando. Não sobrescreve um airflow/.env já existente.

Só as chaves que ninguém digita são geradas aqui. As credenciais que uma pessoa usa (senha do
Postgres, usuário e senha da UI) ficam com valor fixo no .env.example, porque valor aleatório
nelas custa caro e não protege nada num stack local: o Postgres não escuta no host, e uma
senha aleatória na UI obriga quem só quer avaliar o projeto a caçar o arquivo antes de logar.
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
    print(f"{ENV_FILE} criado, com as chaves criptográficas geradas automaticamente.")
    print("Login da UI em localhost:8080: admin / admin.")


if __name__ == "__main__":
    main()
