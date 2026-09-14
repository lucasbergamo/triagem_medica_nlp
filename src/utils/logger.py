import structlog


def get_logger(name: str) -> structlog.BoundLogger:
    """Logger estruturado em JSON no stdout.

    JSON em vez do ConsoleRenderer legível para humano (diferença do TC2): é o que
    permite o CloudWatch parsear o log sem adaptador quando isto rodar no ECS. O Airflow
    também captura o stdout da task, então o mesmo logger serve os dois lugares sem mudança.

    `logger_factory` fixado explicitamente: todo parâmetro que `structlog.configure()` não
    recebe fica com o que já estava configurado antes, não volta ao default — o Airflow 3
    configura o próprio `logger_factory` (orientado a bytes, para o pipe de log da task de
    volta ao scheduler) antes do código da task rodar. Sem fixar o nosso aqui, os `processors`
    abaixo (terminando em `JSONRenderer`, que produz `str`) rodam sobre o factory do Airflow,
    e a escrita do log quebra com `TypeError: can only concatenate str (not "bytes") to str`
    na primeira chamada dentro de uma task do LocalExecutor.
    """
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
    )
    return structlog.get_logger(name)
