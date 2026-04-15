"""
utils/logger.py — Centralised logging setup.

Every module imports `get_logger(__name__)` to get a properly-configured
logger that writes to both the console and a rotating log file.
"""

import logging
import sys
from logging.handlers import RotatingFileHandler
from config import LOG_LEVEL, LOG_FILE


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger with the given name.
    On first call the root handler is configured; subsequent calls are cheap.
    """
    logger = logging.getLogger(name)

    # Only configure handlers once (on the root logger)
    if not logging.getLogger().handlers:
        _configure_root_logger()

    return logger


def _configure_root_logger() -> None:
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(fmt)

    # Rotating file handler (5 MB × 3 backups)
    file_handler = RotatingFileHandler(
        LOG_FILE, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(console)
    root.addHandler(file_handler)
