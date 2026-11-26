"""
Structured Logging Utility for BEANWISE.

Provides centralized, thread-safe logging with formatted console and file outputs.
Supports INFO, WARNING, ERROR, and DEBUG levels with automatic rotation avoidance
and duplicate handler prevention.
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from config.settings import LOGS_DIR, PROJECT_NAME

# Standard format for professional log records
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_LOGGING_INITIALIZED = False


def setup_logging(
    log_file_name: str = "beanwise.log",
    log_level: int = logging.INFO,
    logs_dir: Optional[Path] = None,
    force_new_file: bool = False,
) -> Path:
    """
    Initializes the root logging configuration for the project.
    Directs INFO and higher logs to console, and all DEBUG and higher logs to file.

    Args:
        log_file_name: Name of the log file inside the logs directory.
        log_level: Default logging level for console output.
        logs_dir: Target logs directory (defaults to config.settings.LOGS_DIR).
        force_new_file: If True, adds a new FileHandler even if one already exists.

    Returns:
        Path to the active log file.
    """
    global _LOGGING_INITIALIZED

    target_dir = logs_dir or LOGS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    log_path = (target_dir / log_file_name).resolve()

    root_logger = logging.getLogger(PROJECT_NAME)
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler (standard output) - ensure only one is present
    has_stream_handler = any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
        for h in root_logger.handlers
    )
    if not has_stream_handler:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler check
    existing_file_handlers = [
        h for h in root_logger.handlers if isinstance(h, logging.FileHandler)
    ]
    file_handler_exists = any(
        Path(h.baseFilename).resolve() == log_path for h in existing_file_handlers
    )

    if not file_handler_exists or force_new_file:
        try:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as exc:
            root_logger.warning(f"Could not initialize file logging handler at {log_path}: {exc}")

    _LOGGING_INITIALIZED = True
    return log_path


def get_logger(module_name: Optional[str] = None) -> logging.Logger:
    """
    Returns a configured logger for a given module.

    Args:
        module_name: Name of the module or component requesting the logger.

    Returns:
        Configured logging.Logger instance.
    """
    if not _LOGGING_INITIALIZED:
        setup_logging()

    if module_name:
        return logging.getLogger(f"{PROJECT_NAME}.{module_name}")
    return logging.getLogger(PROJECT_NAME)
