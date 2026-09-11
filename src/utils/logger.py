import structlog


def get_logger(name: str) -> structlog.BoundLogger:
    """Logger estruturado em JSON no stdout.

    JSON em vez do ConsoleRenderer legível para humano (diferença do TC2): é o que
    permite o CloudWatch parsear o log sem adaptador quando isto rodar no ECS. O Airflow
    também captura o stdout da task, então o mesmo logger serve os dois lugares sem mudança.
    """
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.processors.JSONRenderer(),
        ]
    )
    return structlog.get_logger(name)
