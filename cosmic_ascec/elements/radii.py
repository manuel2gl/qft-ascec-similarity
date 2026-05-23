"""Atomic-radius lookup used during initial geometry placement.

Ported from v04's `get_radius` (ascec-v04.py line 369). Rule: monatomic species
use the Bondi/Alvarez van-der-Waals radius; everything else uses the Cordero
covalent radius. Defaults match v04 exactly so the box-length calculator and
overlap checker stay parity-clean.
"""

from __future__ import annotations

from cosmic_ascec.elements.data import COVALENT_RADII, VDW_RADII

_COVALENT_DEFAULT = 1.50
_VDW_FALLBACK_SCALE = 1.2
_VDW_HARD_DEFAULT = 1.70


def get_radius(atomic_number: int, *, monatomic: bool = False) -> float:
    """Return the radius (Å) appropriate for box-length and overlap checks.

    Args:
        atomic_number: Element Z.
        monatomic: True when this Z represents a single-atom species (an atom
            or atomic ion) — then we use the VDW radius. Polyatomic molecules
            use the covalent radius. v04 inferred this from
            ``len(mol_def.atoms_coords) == 1``; passing a flag keeps the
            geometry module decoupled from data structures.

    The fallback chain for an unknown ``Z`` matches v04: VDW lookup falls back
    to ``r_atom[Z] * 1.2`` then to 1.70 Å; covalent lookup falls back to 1.50 Å.
    """
    if monatomic:
        if atomic_number in VDW_RADII:
            return VDW_RADII[atomic_number]
        covalent = COVALENT_RADII.get(atomic_number)
        if covalent is not None:
            return covalent * _VDW_FALLBACK_SCALE
        return _VDW_HARD_DEFAULT
    return COVALENT_RADII.get(atomic_number, _COVALENT_DEFAULT)


__all__ = ["get_radius"]
