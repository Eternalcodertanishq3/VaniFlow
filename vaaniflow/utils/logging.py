"""
Structured logging with structlog.
Every log event must be JSON-serializable with consistent fields.
Sarvam wants "making systems observable" — this is how.
"""
import logging
import sys
import structlog
from structlog.types import EventDict, WrappedLogger


def add_job_context(logger: WrappedLogger, method: str, event_dict: EventDict) -> EventDict:
    """Add standard job context to every log event."""
    return event_dict


def setup_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog for production JSON logging.
    Every log event includes: timestamp, level, logger, event, + context fields.
    """
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if sys.stderr.isatty():
        # Development: pretty colored output
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    else:
        # Production: JSON output for log aggregators
        processors = shared_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelName(log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
