"""Mass-weighted molecular descriptors: COM, RoG, principal moments, rot. consts.

Ported from ``cosmic-v01.py`` lines 425-533 (and the corresponding helper at
``ascec-v04.py`` line 1180 for ``calculate_mass_center``). All formulas are
identical to the v04 + cosmic-v01 sources — same physical constants, same
ascending-eigenvalue ordering, same A ≥ B ≥ C convention.

Conventions:
  - Coordinates in Ångström.
  - Atomic masses in atomic mass units (amu); pulled from
    :data:`cosmic_ascec.elements.data.ATOMIC_WEIGHTS`.
  - Inertia tensor returned in amu·Å² (not SI).
  - Rotational constants returned in cm⁻¹, sorted descending (A ≥ B ≥ C).
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple

import numpy as np

from cosmic_ascec.elements.data import ATOMIC_WEIGHTS


# Physical constants (cosmic-v01.py lines 511-515).
_PLANCK_J_S: float = 6.62607015e-34
_SPEED_OF_LIGHT_CM_PER_S: float = 2.99792458e10
_AMU_TO_KG: float = 1.66053906660e-27
_ANGSTROM_TO_M: float = 1e-10
_AMU_ANG2_TO_KG_M2: float = _AMU_TO_KG * (_ANGSTROM_TO_M ** 2)
_INERTIA_TO_ROT_CONST_FACTOR: float = _PLANCK_J_S / (8.0 * math.pi ** 2)


def _masses_for(atomic_numbers: Sequence[int]) -> np.ndarray:
    """Look up atomic weights; unknown Z returns 0 (matches v04's ``.get`` fallback)."""
    return np.asarray(
        [ATOMIC_WEIGHTS.get(int(z), 0.0) for z in atomic_numbers],
        dtype=np.float64,
    )


def calculate_mass_center(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Centre of mass; returns ``[0, 0, 0]`` when total mass is zero.

    Matches ``calculate_mass_center`` at ascec-v04.py line 1180.
    """
    coords = np.asarray(coords, dtype=np.float64)
    masses = np.asarray(masses, dtype=np.float64)
    total = float(np.sum(masses))
    if total == 0.0:
        return np.zeros(3, dtype=np.float64)
    return np.sum(coords * masses[:, np.newaxis], axis=0) / total


def calculate_radius_of_gyration(
    atomic_numbers: Sequence[int],
    coords: np.ndarray,
) -> float:
    """Mass-weighted RoG (Å). Returns 0 for total-mass-zero inputs (cosmic-v01.py line 446)."""
    masses = _masses_for(atomic_numbers)
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N, 3); got {coords.shape}")

    total = float(np.sum(masses))
    if total == 0.0:
        return 0.0

    com = np.sum(coords.T * masses, axis=1) / total
    rg_sq = float(np.sum(masses * np.sum((coords - com) ** 2, axis=1)) / total)
    return math.sqrt(rg_sq)


def calculate_inertia_tensor(
    atomic_numbers: Sequence[int],
    coords: np.ndarray,
) -> np.ndarray:
    """Inertia tensor in amu·Å², centred on the centre of mass.

    Exposed so callers can derive their own principal axes or angular
    descriptors; the matrix is the same one ``cosmic-v01.py`` builds inline
    in ``calculate_rotational_constants`` (lines 493-504).
    """
    masses = _masses_for(atomic_numbers)
    coords = np.asarray(coords, dtype=np.float64)
    com = calculate_mass_center(coords, masses)
    r = coords - com

    Ixx = float(np.sum(masses * (r[:, 1] ** 2 + r[:, 2] ** 2)))
    Iyy = float(np.sum(masses * (r[:, 0] ** 2 + r[:, 2] ** 2)))
    Izz = float(np.sum(masses * (r[:, 0] ** 2 + r[:, 1] ** 2)))
    Ixy = -float(np.sum(masses * r[:, 0] * r[:, 1]))
    Ixz = -float(np.sum(masses * r[:, 0] * r[:, 2]))
    Iyz = -float(np.sum(masses * r[:, 1] * r[:, 2]))

    return np.array(
        [[Ixx, Ixy, Ixz],
         [Ixy, Iyy, Iyz],
         [Ixz, Iyz, Izz]],
        dtype=np.float64,
    )


def calculate_principal_moments(
    atomic_numbers: Sequence[int],
    coords: np.ndarray,
) -> np.ndarray:
    """Principal moments of inertia (ascending), in amu·Å²."""
    tensor = calculate_inertia_tensor(atomic_numbers, coords)
    return np.linalg.eigvalsh(tensor)


def calculate_rotational_constants(
    atomic_numbers: Sequence[int],
    coords: np.ndarray,
) -> np.ndarray:
    """Rotational constants ``[A, B, C]`` in cm⁻¹ (descending order).

    A principal moment of zero (linear molecules, monatomics) maps to a
    rotational constant of zero — same convention as cosmic-v01.py line 521.
    """
    eigenvalues = calculate_principal_moments(atomic_numbers, coords)
    constants: list[float] = []
    for I_val in eigenvalues:
        if I_val <= 0.0:
            constants.append(0.0)
        else:
            I_si = float(I_val) * _AMU_ANG2_TO_KG_M2
            freq_hz = _INERTIA_TO_ROT_CONST_FACTOR / I_si
            constants.append(freq_hz / _SPEED_OF_LIGHT_CM_PER_S)
    constants.sort(reverse=True)
    return np.asarray(constants, dtype=np.float64)


__all__ = [
    "calculate_inertia_tensor",
    "calculate_mass_center",
    "calculate_principal_moments",
    "calculate_radius_of_gyration",
    "calculate_rotational_constants",
]
