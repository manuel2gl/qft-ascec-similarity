"""Energy selection for representative picking and Boltzmann populations.

What counts as a structure's "energy" depends on what was computed:

* Just an SCF energy → rank on electronic energy.
* Plus thermochemistry → rank on Gibbs free energy.
* Plus composite refinement → rank on the composite Gibbs energy
  (:mod:`~cosmic_ascec.clustering.composite_energies`).

The choice flows through every downstream step (cluster representative
picking, Boltzmann populations, XYZ comment lines), so it is captured once
in an immutable :class:`EnergyMode` and threaded explicitly through the
pipeline — no hidden globals.

The Boltzmann constant and Hartree→kcal/mol factor are pinned here as
module-local constants rather than imported from
:mod:`cosmic_ascec.constants`. The values differ from ``constants.py`` in
the 7th-8th significant figure; a Boltzmann population is the tie-break that
orders motifs by rank, so the clustering layer keeps its own copy to stay
bit-for-bit reproducible against historical runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

# Boltzmann constant in Hartree per Kelvin. Pinned here, deliberately not
# imported from cosmic_ascec.constants (which differs at the 1e-8 digit).
BOLTZMANN_CONSTANT_HARTREE_PER_K: float = 3.1668114e-6

# Display-unit conversions used by the printable cluster reports.
HARTREE_TO_KCAL_MOL: float = 627.509474
HARTREE_TO_EV: float = 27.211386245988


@dataclass(frozen=True)
class EnergyMode:
    """Which energy ranks structures, and what the XYZ comment lines say.

    Mirrors cosmic-v01's ``_DATASET_HAS_FREQ`` / ``_DATASET_HAS_COMPOSITE``
    globals, but immutable and passed explicitly.

    Attributes
    ----------
    has_freq:
        ``True`` when at least one structure carries a Gibbs free energy —
        cosmic-v01's freq mode. ``False`` is opt-only mode (rank by electronic
        energy).
    has_composite:
        ``True`` once :func:`cosmic_ascec.clustering.composite_energies` has
        attached ``composite_gibbs`` values from a previous stage.
    """

    has_freq: bool = True
    has_composite: bool = False


def sorting_energy(mol_data: Mapping[str, Any], mode: EnergyMode) -> float:
    """Return the energy used to sort/select a representative.

    Verbatim logic from cosmic-v01's ``_sorting_energy`` (lines 92-112):
    a composite Gibbs energy wins if present, else Gibbs (freq mode) or
    electronic energy (opt-only); a missing value sorts the structure last.
    """
    composite = mol_data.get("composite_gibbs")
    if composite is not None:
        return composite
    if mode.has_freq:
        gibbs = mol_data.get("gibbs_free_energy")
        return gibbs if gibbs is not None else float("inf")
    electronic = mol_data.get("final_electronic_energy")
    return electronic if electronic is not None else float("inf")


def hartree_to_kcal_mol(energy_hartree: float) -> float:
    """Hartree → kcal/mol (cosmic-v01.py line 115-117)."""
    return energy_hartree * HARTREE_TO_KCAL_MOL


def hartree_to_ev(energy_hartree: float) -> float:
    """Hartree → eV (cosmic-v01.py line 119-121)."""
    return energy_hartree * HARTREE_TO_EV


__all__ = [
    "BOLTZMANN_CONSTANT_HARTREE_PER_K",
    "HARTREE_TO_EV",
    "HARTREE_TO_KCAL_MOL",
    "EnergyMode",
    "hartree_to_ev",
    "hartree_to_kcal_mol",
    "sorting_energy",
]
