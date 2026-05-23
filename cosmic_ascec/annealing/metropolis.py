"""Metropolis acceptance criterion — re-exported from :mod:`monte_carlo.trial`.

The Session 5 plan lists ``annealing/metropolis.py`` as a module, but the
acceptance logic already shipped in Session 3 inside
:mod:`cosmic_ascec.monte_carlo.trial` (it has to live there: the trial driver
needs it, and the trial driver is one layer below annealing). Rather than
duplicate v04's two-branch criterion — which would invite the two copies to
drift — this module simply re-exports it.

v04's criterion has two branches (ascec-v04.py lines 20691-20717), selected by
``state.use_standard_metropolis``:

* **Standard Metropolis** — accept an uphill move iff ``rng.random_sample()
  < pE`` where ``pE = exp(-ΔE / (k_B · T))``.
* **Modified Metropolis** (v04 default) — accept iff
  ``ΔE / |E_proposed| < pE``; deterministic given the energies and ``T``.

Downhill moves (``ΔE <= 0``) always accept. See
:func:`cosmic_ascec.monte_carlo.trial.metropolis_decision` for the
implementation and :data:`BOLTZMANN_HARTREE_PER_KELVIN` for v04's ``B2``.
"""

from __future__ import annotations

from cosmic_ascec.monte_carlo.trial import (
    BOLTZMANN_HARTREE_PER_KELVIN,
    metropolis_decision,
)

__all__ = ["BOLTZMANN_HARTREE_PER_KELVIN", "metropolis_decision"]
