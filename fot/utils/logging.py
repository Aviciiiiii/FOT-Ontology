from __future__ import annotations

import logging
import sys
from typing import Optional

from . import paths


def get_logger(name: str, level: str = "INFO", to_file: bool = False, file_key: str = "artifacts.logs") -> logging.Logger:
    """Create or retrieve a configured logger.

    - level: string level (e.g., INFO, DEBUG)
    - to_file: if True, also write logs to a file under the directory configured by file_key
    - file_key: dot key into configs/paths.yaml to locate the logs directory
    """
    logger = logging.getLogger(name)
    logger.propagate = False

    # Clear existing handlers to avoid duplicate logs on repeated calls
    if logger.handlers:
        logger.handlers.clear()

    lvl = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(lvl)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(lvl)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if to_file:
        log_path = paths.expand(file_key, f"{name}.log")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(lvl)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
