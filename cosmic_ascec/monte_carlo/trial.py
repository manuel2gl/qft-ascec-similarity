"""One Monte Carlo trial — propose, evaluate, accept/reject.

A trial is the unit of work the annealing engine spends one QM call on:

1. Propose a new full-system geometry via
   :func:`~cosmic_ascec.monte_carlo.moves.propose_unified_move`
   (every molecule moved at once).
2. Evaluate its energy via the injected ``energy_fn`` (one QM call).
3. Decide acceptance with the Metropolis criterion.

There is **no overlap gate** here — a clashing rigid-move proposal goes
straight to QM. The only overlap check on the live move path is the
intramolecular one inside the conformational branch of
:func:`~cosmic_ascec.monte_carlo.moves.propose_unified_move`. This is
deliberate: spending QM calls *is* the search budget, and rejecting moves
before the call would distort the temperature schedule's QM accounting.

Acceptance criterion (selected by ``use_standard_metropolis``):

* ``delta_e <= 0`` → always accept, criterion ``"LwE"``.
* otherwise compute ``pE = exp(-delta_e / (B2 * T))`` and either:
  * **standard Metropolis** (``--standard``): accept iff a uniform draw is
    ``< pE`` — criterion ``"EMpol"``. The draw comes from the run RNG.
  * **modified Metropolis** (default): accept iff
    ``delta_e / |proposed_energy| < pE`` — criterion ``"Mpol"``.
    Deterministic; draws no randomness.

``B2`` is Boltzmann's constant in Hartree/Kelvin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from cosmic_ascec.geometry.molecule import Cluster
from cosmic_ascec.monte_carlo.moves import (
    MoveParams,
    RotatableBondCache,
    propose_unified_move,
)

BOLTZMANN_HARTREE_PER_KELVIN: float = 3.166811563e-6
"""v04 ``B2`` (ascec-v04.py line 68). Used inside :func:`metropolis_decision`."""


# --------------------------------------------------------------------------- #
# Energy contract                                                             #
# --------------------------------------------------------------------------- #


EnergyFn = Callable[[Cluster], float]
"""Signature of any energy provider plugged into :func:`trial`.

The real :class:`QuantumChemistryAdapter` is wired behind this signature so the
engine never has to learn about adapters. A failed QM evaluation raises
:class:`~cosmic_ascec.exceptions.QMError`; :func:`trial` lets that propagate so
the engine can count the consumed QM cycle and reject the move — exactly v04's
``jo_status == 0`` path (ascec-v04.py line 20681).
"""


def zero_energy(cluster: Cluster) -> float:  # noqa: ARG001 - parity stub
    """Constant-zero energy. Lets the trial loop be exercised without QM."""
    return 0.0


# --------------------------------------------------------------------------- #
# Metropolis                                                                  #
# --------------------------------------------------------------------------- #


def metropolis_decision(
    delta_energy: float,
    proposed_energy: float,
    *,
    temperature_kelvin: float,
    rng: np.random.RandomState,
    use_standard_metropolis: bool = False,
) -> tuple[bool, str]:
    """Apply v04's acceptance criterion. Returns ``(accept, criterion)``.

    Criteria mirror v04's strings (ascec-v04.py lines 20691-20717):

    * ``"LwE"`` — proposal lowers (or equals) the energy.
    * ``"EMpol"`` — proposal accepted by standard Metropolis exponential.
    * ``"Mpol"`` — proposal accepted by v04's *modified* Metropolis
      ``delta_e / |proposed_energy| < pE``.
    * ``"Rejected"`` — none of the above.

    The standard branch draws one ``rng.random_sample()`` (v04's
    ``random.random()``, folded into the single RNG, D-038); the modified
    branch is deterministic and draws nothing. The downhill branch
    (``delta_energy <= 0``) draws nothing in either mode.
    """
    if delta_energy <= 0.0:
        return True, "LwE"

    if temperature_kelvin < 1e-6:
        pE = 0.0
    else:
        pE = math.exp(
            -delta_energy / (BOLTZMANN_HARTREE_PER_KELVIN * temperature_kelvin)
        )

    if use_standard_metropolis:
        if rng.random_sample() < pE:
            return True, "EMpol"
        return False, "Rejected"

    # Modified Metropolis (v04 default) — deterministic given the energies/T.
    if abs(proposed_energy) < 1e-12:
        crt = math.inf
    else:
        crt = delta_energy / abs(proposed_energy)
    if crt < pE:
        return True, "Mpol"
    return False, "Rejected"


# --------------------------------------------------------------------------- #
# Trial driver                                                                #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrialResult:
    """Outcome of one :func:`trial` call.

    On acceptance ``cluster``/``energy`` are the proposed geometry and its
    energy; on rejection they are the *original* cluster and energy (v04 never
    mutates ``state.rp`` for a rejected proposal — line 20779). ``criterion``
    is one of ``"LwE"``, ``"Mpol"``, ``"EMpol"``, ``"Rejected"``.
    """

    accepted: bool
    cluster: Cluster
    energy: float
    criterion: str
    delta_energy: float
    proposed_energy: float
    move_type: str
    last_moved_molecule: int


def trial(
    cluster: Cluster,
    *,
    current_energy: float,
    energy_fn: EnergyFn,
    rng: np.random.RandomState,
    temperature_kelvin: float,
    params: MoveParams,
    cache: RotatableBondCache,
) -> TrialResult:
    """Run one propose → energy → accept step (v04 loop body, lines 20665-20717).

    Raises whatever ``energy_fn`` raises — notably
    :class:`~cosmic_ascec.exceptions.QMError` on a failed QM evaluation, which
    the engine treats as a consumed-but-rejected cycle.
    """
    move = propose_unified_move(cluster, rng=rng, params=params, cache=cache)
    proposed_cluster = move.cluster

    proposed_energy = float(energy_fn(proposed_cluster))
    delta = proposed_energy - current_energy
    accepted, criterion = metropolis_decision(
        delta,
        proposed_energy,
        temperature_kelvin=temperature_kelvin,
        rng=rng,
        use_standard_metropolis=params.use_standard_metropolis,
    )

    if accepted:
        return TrialResult(
            accepted=True,
            cluster=proposed_cluster,
            energy=proposed_energy,
            criterion=criterion,
            delta_energy=delta,
            proposed_energy=proposed_energy,
            move_type=move.move_type,
            last_moved_molecule=move.last_moved_molecule,
        )

    return TrialResult(
        accepted=False,
        cluster=cluster,
        energy=current_energy,
        criterion=criterion,
        delta_energy=delta,
        proposed_energy=proposed_energy,
        move_type=move.move_type,
        last_moved_molecule=move.last_moved_molecule,
    )


__all__ = [
    "BOLTZMANN_HARTREE_PER_KELVIN",
    "EnergyFn",
    "TrialResult",
    "metropolis_decision",
    "trial",
    "zero_energy",
]
