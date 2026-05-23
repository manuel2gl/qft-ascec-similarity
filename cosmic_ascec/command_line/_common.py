"""Shared command-line helpers (version flag, banner, common argparse pieces)."""

from __future__ import annotations

import argparse

from cosmic_ascec import __version__


def add_version_flag(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--version",
        action="version",
        version=f"cosmic-ascec {__version__}",
    )
