"""I/O for COSMIC ASCEC: ``.asc`` parser/schema and the run-output writers.

The annealing-callback writers (``SummaryWriter``, ``TvseWriter``,
``TrajectoryWriter``) are part of this package's public surface, but they
subclass :class:`cosmic_ascec.annealing.AnnealingCallback`, which in turn
imports :class:`AscConfig` from this package. To break the import cycle
they are exposed **lazily** via :pep:`562` ``__getattr__`` — the writer
module is loaded only on first attribute access, by which point both
packages are fully initialised. Callers can still write
``from cosmic_ascec.file_formats import SummaryWriter`` without thinking
about the cycle.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from cosmic_ascec.file_formats.asc_parser import parse_asc
from cosmic_ascec.file_formats.asc_schema import (
    AscConfig,
    BoxSpec,
    CycleSpec,
    EmbeddedTemplate,
    GeometricSchedule,
    LinearSchedule,
    MoleculeSpec,
    MoveSpec,
    ProtocolSpec,
    QMProgram,
    QMSpec,
    QuenchingRoute,
    ScheduleSpec,
    SimulationMode,
)

# Writer name -> module that defines it. Resolved on first access (see below).
_LAZY_WRITERS = {
    "SummaryWriter": "cosmic_ascec.file_formats.summary",
    "TrajectoryWriter": "cosmic_ascec.file_formats.trajectory_writer",
    "TvseWriter": "cosmic_ascec.file_formats.tvse_writer",
}

if TYPE_CHECKING:  # let type-checkers and IDEs see the writers statically.
    from cosmic_ascec.file_formats.summary import SummaryWriter
    from cosmic_ascec.file_formats.trajectory_writer import TrajectoryWriter
    from cosmic_ascec.file_formats.tvse_writer import TvseWriter


def __getattr__(name: str) -> Any:
    """Lazily import a writer the first time it is accessed (PEP 562)."""
    module_name = _LAZY_WRITERS.get(name)
    if module_name is not None:
        return getattr(importlib.import_module(module_name), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AscConfig",
    "BoxSpec",
    "CycleSpec",
    "EmbeddedTemplate",
    "GeometricSchedule",
    "LinearSchedule",
    "MoleculeSpec",
    "MoveSpec",
    "ProtocolSpec",
    "QMProgram",
    "QMSpec",
    "QuenchingRoute",
    "ScheduleSpec",
    "SimulationMode",
    "SummaryWriter",
    "TrajectoryWriter",
    "TvseWriter",
    "parse_asc",
]
