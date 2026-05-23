"""Simulated-annealing engine: temperature schedule, acceptance, main loop.

Re-exports the public surface of the submodules so callers can write
``from cosmic_ascec.annealing import anneal, AnnealingCallback`` without
caring about the internal split (**D-009**).
"""

from cosmic_ascec.annealing.engine import (
    AnnealingCallback,
    AnnealingResult,
    ConfigAccepted,
    DEFAULT_CYCLE_FLOOR,
    MAXSTEP_REDUCTION_FACTOR,
    RunFinish,
    RunStart,
    TemperatureComplete,
    TemperatureStart,
    anneal,
)
from cosmic_ascec.annealing.metropolis import (
    BOLTZMANN_HARTREE_PER_KELVIN,
    metropolis_decision,
)
from cosmic_ascec.annealing.schedule import (
    TEMPERATURE_FLOOR,
    build_temperature_schedule,
    geometric_quench,
    linear_quench,
)

__all__ = [
    "AnnealingCallback",
    "AnnealingResult",
    "BOLTZMANN_HARTREE_PER_KELVIN",
    "ConfigAccepted",
    "DEFAULT_CYCLE_FLOOR",
    "MAXSTEP_REDUCTION_FACTOR",
    "RunFinish",
    "RunStart",
    "TEMPERATURE_FLOOR",
    "TemperatureComplete",
    "TemperatureStart",
    "anneal",
    "build_temperature_schedule",
    "geometric_quench",
    "linear_quench",
    "metropolis_decision",
]
