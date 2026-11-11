"""Structured logging configuration based on structlog."""

import logging
import sys
from typing import cast

import structlog
from structlog.typing import FilteringBoundLogger


def configure_logging(level: str | int = "INFO", *, json: bool = False) -> None:
    """Configure structlog for human readable console output or JSON lines.

    Every event carries its level and an ISO timestamp and is written to the standard error. Events
    below `level` are dropped.

    Args:
        level: Lowest level written, as a standard level name in any case or its numeric value.
        json: Whether to render each event as one JSON object per line instead of console text.

    Raises:
        KeyError: If `level` is a string that names no standard logging level.
    """
    if isinstance(level, str):
        level = logging.getLevelNamesMapping()[level.upper()]
    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(sys.stderr),
        cache_logger_on_first_use=False,
    )


def get_logger(component: str) -> FilteringBoundLogger:
    """Logger bound to a named component.

    Args:
        component: Name recorded under the `component` key of every event, usually the module name.

    Returns:
        A structlog logger that adds `component` to every event it emits.
    """
    return cast(FilteringBoundLogger, structlog.get_logger().bind(component=component))
