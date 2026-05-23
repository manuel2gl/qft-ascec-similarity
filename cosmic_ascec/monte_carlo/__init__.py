"""Monte Carlo move library and trial driver.

The MC layer is deliberately thin: one move proposer (every molecule moves
each cycle — either a rigid translation+rotation or, for molecules with
rotatable bonds and a successful probability draw, a dihedral rotation),
one acceptance test (Metropolis), and one trial driver that threads the
two together. The annealing engine drives this layer; the QM layer is
injected as a ``Cluster -> float`` energy function. That's the whole
machinery — nothing else is needed for a working annealing loop.

Re-exports the public surface of :mod:`moves` and :mod:`trial` so callers
can import from this package without caring about the internal split.
"""

from cosmic_ascec.monte_carlo.moves import (
    INITIAL_QM_RETRIES,
    MoveParams,
    RotatableBond,
    RotatableBondCache,
    UnifiedMoveResult,
    initialize_rotatable_bond_cache,
    propose_unified_move,
    rotate_around_bond,
)
from cosmic_ascec.monte_carlo.trial import (
    BOLTZMANN_HARTREE_PER_KELVIN,
    EnergyFn,
    TrialResult,
    metropolis_decision,
    trial,
    zero_energy,
)

__all__ = [
    "BOLTZMANN_HARTREE_PER_KELVIN",
    "EnergyFn",
    "INITIAL_QM_RETRIES",
    "MoveParams",
    "RotatableBond",
    "RotatableBondCache",
    "TrialResult",
    "UnifiedMoveResult",
    "initialize_rotatable_bond_cache",
    "metropolis_decision",
    "propose_unified_move",
    "rotate_around_bond",
    "trial",
    "zero_energy",
]
