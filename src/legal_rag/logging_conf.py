"""Structured JSON logging configuration with correlation ID tracking."""

import contextvars
import logging
import sys

from pythonjsonlogger import jsonlogger

# Context variable to hold correlation ID across async requests
correlation_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "correlation_id", default=None
)


def get_correlation_id() -> str:
    """Retrieve the current request's correlation ID or return a placeholder."""
    return correlation_id_ctx.get() or "no-correlation-id"


def set_correlation_id(corr_id: str) -> None:
    """Set the correlation ID for the current context."""
    correlation_id_ctx.set(corr_id)


class CorrelationJsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter that injects correlation_id and standardized fields."""

    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = self.formatTime(record, self.datefmt)
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["correlation_id"] = get_correlation_id()


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structured JSON logging for stdout."""
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level.upper())

    # Remove existing handlers
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    formatter = CorrelationJsonFormatter(
        fmt="%(timestamp)s %(level)s %(logger)s %(message)s %(correlation_id)s"
    )
    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Suppress chatty 3rd-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
