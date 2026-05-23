"""Physical and unit-conversion constants used throughout COSMIC ASCEC.

Values match the conventions of the v04 codebase to preserve numerical parity.
"""

from __future__ import annotations

from typing import Final

# Energy
HARTREE_TO_KCAL_PER_MOL: Final[float] = 627.5094740631
HARTREE_TO_EV: Final[float] = 27.211386245988
HARTREE_TO_KJ_PER_MOL: Final[float] = 2625.499638
KCAL_PER_MOL_TO_HARTREE: Final[float] = 1.0 / HARTREE_TO_KCAL_PER_MOL

# Boltzmann
BOLTZMANN_HARTREE_PER_K: Final[float] = 3.166811563e-6  # k_B in Hartree / Kelvin

# Length
BOHR_TO_ANGSTROM: Final[float] = 0.529177210903
ANGSTROM_TO_BOHR: Final[float] = 1.0 / BOHR_TO_ANGSTROM

# Mass
AMU_TO_KG: Final[float] = 1.66053906660e-27

# Defaults used in v04
DEFAULT_TEMPERATURE_K: Final[float] = 298.15

# Note: the overlap-scale factor and max-placement-attempts values are *not*
# kept here. The geometry layer pins its own thresholds as module-local named
# constants next to the code that uses them (D-006): see
# DEFAULT_INTERMOLECULAR_OVERLAP_SCALE in geometry/overlap.py and
# PlacementSettings.max_attempts in geometry/placement.py. The stale values
# that once lived here (0.85 / 1000) never matched v04 and were removed in
# Session 5 to keep a single source of truth.
