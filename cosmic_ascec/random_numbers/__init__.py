"""The single run RNG and the seed flow (D-038).

v05 uses **one** ``numpy.random.RandomState`` per run, built from the run seed
and threaded explicitly as an ``rng`` argument — never a module global. v04's
``ran0`` generator is deleted: it was dead code on the live annealing path.

See :mod:`cosmic_ascec.random_numbers.rng` and ``docs/rng_and_seed_flow.md``.
"""

from cosmic_ascec.random_numbers.rng import (
    SEED_ENV_VAR,
    SEED_MAX,
    SEED_MIN,
    draw_seed,
    make_rng,
    resolve_run_seed,
)

__all__ = [
    "SEED_ENV_VAR",
    "SEED_MAX",
    "SEED_MIN",
    "draw_seed",
    "make_rng",
    "resolve_run_seed",
]
