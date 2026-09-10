"""Application logging setup.

The service units send stdout/stderr to journald, which is unreadable
without ``systemd-journal`` group membership.  Setting ``LEX_LLM_LOG_FILE``
writes the same records to a file the ``apps`` user can already reach
(e.g. ``/apps/.logs/lex_llm_app.log``), so a run can be analysed without
extra privileges.

Environment:
    LEX_LLM_LOG_FILE   Path to write to.  Falls back to stderr (journal).
    LEX_LLM_LOG_LEVEL  Level name, default INFO.
"""

import logging
import os

LOGGER_NAME = "lex_llm"

_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def get_logger() -> logging.Logger:
    """Get the application logger."""
    return logging.getLogger(LOGGER_NAME)


def setup_logging() -> logging.Logger:
    """Configure the application logger.  Idempotent."""
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    level = getattr(
        logging, os.getenv("LEX_LLM_LOG_LEVEL", "INFO").upper(), logging.INFO
    )
    logger.setLevel(level)
    # Don't also hand records to the root logger; uvicorn owns that.
    logger.propagate = False

    log_file = os.getenv("LEX_LLM_LOG_FILE")
    handler: logging.Handler
    if log_file:
        try:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            handler = logging.FileHandler(log_file)
        except OSError:
            # An unwritable path must never take the service down.
            handler = logging.StreamHandler()
    else:
        handler = logging.StreamHandler()

    handler.setFormatter(logging.Formatter(_FORMAT))
    logger.addHandler(handler)
    return logger
