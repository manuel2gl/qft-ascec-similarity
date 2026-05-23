"""Centralised logging configuration for COSMIC ASCEC.

Submodules obtain their logger via `logging.getLogger(__name__)`. The CLI layer
calls `configure_logging()` once at startup. No `print()` calls anywhere outside
the `cli/` package.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

ROOT_LOGGER_NAME: Final[str] = "cosmic_ascec"
_DEFAULT_FORMAT: Final[str] = "[%(asctime)s] %(levelname)-7s %(name)s: %(message)s"
_DEFAULT_DATEFMT: Final[str] = "%Y-%m-%d %H:%M:%S"


def configure_logging(verbosity: int = 0, *, quiet: bool = False) -> None:
    """Set up the package's root logger.

    Args:
        verbosity: 0 = WARNING (default), 1 = INFO, 2+ = DEBUG.
        quiet: If True, force level to ERROR regardless of verbosity.
    """
    if quiet:
        level = logging.ERROR
    elif verbosity >= 2:
        level = logging.DEBUG
    elif verbosity == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    root = logging.getLogger(ROOT_LOGGER_NAME)
    root.setLevel(level)

    # Replace any pre-existing handlers (idempotent across re-calls in tests).
    for handler in list(root.handlers):
        root.removeHandler(handler)

    stream_handler = logging.StreamHandler(stream=sys.stderr)
    stream_handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT))
    root.addHandler(stream_handler)

    # Prevent propagation to Python's root logger to avoid duplicate output.
    root.propagate = False
