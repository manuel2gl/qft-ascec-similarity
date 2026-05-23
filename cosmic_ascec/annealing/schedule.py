"""Temperature quenching schedules for the annealing engine.

v04 supports two cooling routes (ascec-v04.py lines 20635-20640, inside the
main annealing loop):

* **Linear** (``quenching_routine == 1``) — subtract a fixed decrement each
  step: ``T <- max(T - dT, floor)``.
* **Geometric** (``quenching_routine == 2``) — multiply by a cooling factor
  each step: ``T <- max(T * factor, floor)``.

v04 applies the quench *before* the Monte Carlo cycles of every step except
the first (``if step_num > 0``), so the initial temperature is used as-is for
step 0. This module pre-computes the whole temperature list once with
:func:`build_temperature_schedule`, which keeps the engine loop free of any
route branching.

The geometric ``factor`` stored on :class:`GeometricSchedule` is already the
multiplicative factor (v04 converts the on-disk percentage — ``5.0`` — into
``0.95`` while parsing, see ``file_formats/asc_schema.py``), so this module
multiplies by it directly.
"""

from __future__ import annotations

from cosmic_ascec.exceptions import WorkflowError
from cosmic_ascec.file_formats.asc_schema import QuenchingRoute, ScheduleSpec

# v04 clamps every quenched temperature to this floor so the Metropolis
# exponential never divides by zero (ascec-v04.py lines 20638 / 20640:
# ``max(..., 0.001)``).
TEMPERATURE_FLOOR: float = 0.001


def linear_quench(temperature: float, decrement: float) -> float:
    """Return the next temperature on a linear cooling route.

    Mirrors v04 ``max(state.current_temp - state.linear_temp_decrement, 0.001)``.
    """
    return max(temperature - decrement, TEMPERATURE_FLOOR)


def geometric_quench(temperature: float, factor: float) -> float:
    """Return the next temperature on a geometric cooling route.

    Mirrors v04 ``max(state.current_temp * state.geom_temp_factor, 0.001)``.
    """
    return max(temperature * factor, TEMPERATURE_FLOOR)


def build_temperature_schedule(schedule: ScheduleSpec) -> tuple[float, ...]:
    """Pre-compute the temperature used at every annealing step.

    The returned tuple has one entry per annealing step (its length is the
    ``num_steps`` of the route selected by ``schedule.route``). Entry ``0`` is
    the initial temperature; entry ``i`` is entry ``i-1`` quenched once. This
    matches v04's "quench before the cycles, but not on step 0" behaviour.

    Raises:
        WorkflowError: if the route is neither linear nor geometric, or the
            selected route declares fewer than one step.
    """
    if schedule.route is QuenchingRoute.LINEAR:
        spec = schedule.linear
        num_steps = spec.num_steps
        quench = lambda t: linear_quench(t, spec.delta_temperature)  # noqa: E731
    elif schedule.route is QuenchingRoute.GEOMETRIC:
        spec = schedule.geometric
        num_steps = spec.num_steps
        quench = lambda t: geometric_quench(t, spec.factor)  # noqa: E731
    else:  # pragma: no cover - QuenchingRoute is a closed IntEnum
        raise WorkflowError(f"unknown quenching route: {schedule.route!r}")

    if num_steps < 1:
        raise WorkflowError(
            f"quenching route {schedule.route.name} declares {num_steps} steps; "
            "at least 1 is required"
        )

    temperatures = [float(spec.initial_temperature)]
    for _ in range(num_steps - 1):
        temperatures.append(quench(temperatures[-1]))
    return tuple(temperatures)


__all__ = [
    "TEMPERATURE_FLOOR",
    "build_temperature_schedule",
    "geometric_quench",
    "linear_quench",
]
