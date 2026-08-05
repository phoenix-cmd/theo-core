"""Structured logging — structlog-based implementation.

Provides JSON-serializable structured logs in production and
colored human-readable output in development.
"""

from __future__ import annotations

import logging
import sys
from typing import cast

import structlog


def configure_logging(
    level: str = "INFO",
    format_style: str = "json",
) -> None:
    """Configure the global logging stack.

    Args:
        level: The log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_style: Output format — "json" for production, "console" for dev.

    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if format_style == "console":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(colors=True)
    else:
        renderer = structlog.processors.JSONRenderer()

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger bound to the given name.

    Args:
        name: Logger name (typically __name__).

    Returns:
        A bound structured logger instance.

    """
    return cast("structlog.stdlib.BoundLogger", structlog.get_logger(name))
