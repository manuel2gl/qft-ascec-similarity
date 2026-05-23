"""The single random-number stream for one COSMIC ASCEC run (D-038).

v04 (``ascec-v04.py``) drew randomness from **three** generators, all seeded
from one integer ``random_seed``:

* ``SystemState.ran0_method`` — a Numerical Recipes ``ran0`` port. Only the
  ``trans``/``rotac`` subroutines call it, and those are dead code on the live
  annealing path (see ``docs/rng_and_seed_flow.md`` §2). ``ran0`` is therefore
  deleted in v05.
* the global ``numpy.random`` — initial placement (``config_molecules``,
  v04 lines 1275-1280) and the live move proposal (``propose_unified_move``,
  v04 lines 3811-3882).
* the global stdlib ``random`` — the standard-Metropolis acceptance draw
  (v04 line 20703) and the run-seed auto-draw (v04 line 20360).

D-038 consolidates these to **one** ``numpy.random.RandomState``, constructed
from the run seed and threaded explicitly as an ``rng`` argument — never a
module global. v04's ``np.random.<fn>(...)`` calls become a mechanical
``rng.<fn>(...)`` rename, and v04's ``random.random()`` Metropolis draw becomes
``rng.random_sample()``.

This module owns exactly two things: constructing that ``RandomState``
(:func:`make_rng`) and the run-seed flow (:func:`resolve_run_seed`,
:func:`draw_seed`).

Seed flow
---------
Faithful to v04: a run's seed is one integer in ``[100000, 999999]``. v04 has
no way to *inject* a seed — ``random_seed`` is always the ``-1`` sentinel and
is always auto-drawn with ``random.randint(100000, 999999)`` (v04 line 20360).
v05 keeps this exactly: :func:`draw_seed` is the auto-draw, the seed is printed
as ``Seed = NNNNNN`` and suffixes the output files (``result_<seed>.xyz``,
``rless_<seed>.out``, ``tvse_<seed>.dat``), and it constructs the run's
``RandomState``.

The single addition is a **test-only** seed-pin hook: :func:`resolve_run_seed`
honours the ``COSMIC_ASCEC_SEED`` environment variable when set. This is *not*
a user-facing feature — v04 has none, so D-021's ``--seed`` flag is reverted —
it exists only so the parity harness can replay a run at a seed observed from
a v04 run. Production runs never set it and behave exactly as v04.
"""

from __future__ import annotations

import os
import random

import numpy as np

#: Environment variable read by :func:`resolve_run_seed`. Test-only — set by
#: the parity harness to pin v05 to a seed observed from a v04 run. It is unset
#: in every production run, so production seed handling is identical to v04.
SEED_ENV_VAR = "COSMIC_ASCEC_SEED"

#: Inclusive bounds of the run-seed range — v04's ``random.randint(100000,
#: 999999)`` (ascec-v04.py line 20360).
SEED_MIN = 100_000
SEED_MAX = 999_999


def draw_seed() -> int:
    """Auto-draw a fresh 6-digit run seed.

    A verbatim port of v04's ``random.randint(100000, 999999)`` (ascec-v04.py
    line 20360). It uses the *unseeded* stdlib ``random`` global, so every
    process draws a fresh seed from OS entropy — exactly as v04 does, where the
    draw happens before any ``random.seed`` call. The drawn integer is not
    itself reproducible; it is the run's entropy source and its identity.
    """
    return random.randint(SEED_MIN, SEED_MAX)


def resolve_run_seed() -> int:
    """Return the seed for this run.

    Honours the ``COSMIC_ASCEC_SEED`` test-only override (see
    :data:`SEED_ENV_VAR`); otherwise auto-draws via :func:`draw_seed`. v04 has
    no seed-injection path, so production runs never set the variable and this
    is identical to v04's behaviour.

    Raises:
        ValueError: if ``COSMIC_ASCEC_SEED`` is set but not an integer.
    """
    override = os.environ.get(SEED_ENV_VAR)
    if override is not None and override.strip():
        try:
            return int(override)
        except ValueError as exc:
            raise ValueError(
                f"{SEED_ENV_VAR}={override!r} is not an integer"
            ) from exc
    return draw_seed()


def make_rng(seed: int) -> np.random.RandomState:
    """Construct the run's single ``numpy.random.RandomState`` from *seed* (D-038).

    ``numpy.random.RandomState(seed)`` produces the identical stream that v04's
    global ``numpy.random`` produces after ``np.random.seed(seed)`` — so every
    draw v04 made through ``np.random`` (initial placement, move proposal) is
    reproduced bit-for-bit when v05 threads this object and calls the same
    method. v04's separate ``random.random()`` Metropolis draws are folded into
    this one stream; that is the only interleaving change D-038 introduces (see
    ``docs/rng_and_seed_flow.md`` §5).

    Args:
        seed: A non-negative run seed (normally from :func:`resolve_run_seed`).

    Returns:
        A freshly constructed, independent ``numpy.random.RandomState``.
    """
    return np.random.RandomState(int(seed))


__all__ = [
    "SEED_ENV_VAR",
    "SEED_MAX",
    "SEED_MIN",
    "draw_seed",
    "make_rng",
    "resolve_run_seed",
]
