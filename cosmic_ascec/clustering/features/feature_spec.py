"""Feature-vector specification — the parity-critical column contract.

The COSMIC clustering pipeline turns a directory of quantum-chemistry outputs
into one numeric feature vector per candidate structure. *Which* features, in
*what order*, and under *what weights* is a contract: cluster IDs and motif
folders are only reproducible if every run builds the matrix with the same
columns in the same order. This module is the single source of truth for that
contract.

Everything here is ported verbatim (values, names, ordering) from
``cosmic-v01.py``:

* :data:`FEATURE_MAPPING` — cosmic-v01.py lines 3614-3630.
* :data:`CLUSTERING_NUMERICAL_FEATURES` — cosmic-v01.py lines 3672-3684.
* :data:`ROTATIONAL_CONSTANT_SUBFEATURES` — cosmic-v01.py lines 3686-3691.
* :data:`SEMIEMPIRICAL_WEIGHTS` / :data:`DEFAULT_WEIGHTS` — lines 3697-3712.
* :data:`FEATURE_UNITS` — cosmic-v01.py ``_EXTRACT_FEATURE_UNITS`` lines 3717-3733.
* :func:`parse_weights_argument` — cosmic-v01.py lines 3632-3648.

:data:`FEATURE_COLUMNS` is the new name for cosmic-v01's local
``feature_columns = CLUSTERING_NUMERICAL_FEATURES + ROTATIONAL_CONSTANT_SUBFEATURES``
(``run_data_extraction``, cosmic-v01.py line 3780). It is the pinned column
order of the DataFrame :mod:`cosmic_ascec.clustering.features.extractor`
produces (**D-027**).
"""

from __future__ import annotations

import logging
import re
from types import MappingProxyType
from typing import Dict, Mapping, Tuple

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Feature-name mapping                                                         #
# --------------------------------------------------------------------------- #
#
# Maps a clustering feature name → the key it is stored under in cosmic-v01's
# parsed ``extracted_props`` dict. Most names are identical;
# ``electronic_energy`` is the exception (the parsers call it
# ``final_electronic_energy``). The three ``rotational_constants_*`` entries
# are documented here for completeness but are handled specially — they index
# into the 3-element ``rotational_constants`` array rather than naming a key.

FEATURE_MAPPING: Mapping[str, str] = MappingProxyType(
    {
        "electronic_energy": "final_electronic_energy",
        "gibbs_free_energy": "gibbs_free_energy",
        "homo_energy": "homo_energy",
        "lumo_energy": "lumo_energy",
        "homo_lumo_gap": "homo_lumo_gap",
        "dipole_moment": "dipole_moment",
        "radius_of_gyration": "radius_of_gyration",
        # Array elements — special-cased in the extractor (see ``rotational_constants``).
        "rotational_constants_A": "rotational_constants_0",
        "rotational_constants_B": "rotational_constants_1",
        "rotational_constants_C": "rotational_constants_2",
        "first_vib_freq": "first_vib_freq",
        "last_vib_freq": "last_vib_freq",
        "average_hbond_distance": "average_hbond_distance",
        "average_hbond_angle": "average_hbond_angle",
        # Mapped for completeness; not a clustering feature column.
        "num_hydrogen_bonds": "num_hydrogen_bonds",
    }
)


# --------------------------------------------------------------------------- #
# Column order — the parity contract                                           #
# --------------------------------------------------------------------------- #

CLUSTERING_NUMERICAL_FEATURES: Tuple[str, ...] = (
    "electronic_energy",
    "gibbs_free_energy",
    "homo_energy",
    "lumo_energy",
    "homo_lumo_gap",
    "dipole_moment",
    "radius_of_gyration",
    "first_vib_freq",
    "last_vib_freq",
    "average_hbond_distance",
    "average_hbond_angle",
)
"""The 11 scalar clustering features, in cosmic-v01's declared order."""

ROTATIONAL_CONSTANT_SUBFEATURES: Tuple[str, ...] = (
    "rotational_constants_A",
    "rotational_constants_B",
    "rotational_constants_C",
)
"""The 3 rotational-constant axes, appended after the scalar features."""

FEATURE_COLUMNS: Tuple[str, ...] = (
    CLUSTERING_NUMERICAL_FEATURES + ROTATIONAL_CONSTANT_SUBFEATURES
)
"""Pinned feature-DataFrame column order (**D-027**).

This is cosmic-v01's local ``feature_columns`` (``run_data_extraction``,
cosmic-v01.py line 3780). The extractor builds every DataFrame with exactly
these 14 columns in this order; the clustering stage (Session 8) and any
parity check against a v04-cosmic reference depend on it not drifting.
"""


# --------------------------------------------------------------------------- #
# Display units                                                                #
# --------------------------------------------------------------------------- #
#
# cosmic-v01's ``_EXTRACT_FEATURE_UNITS`` (lines 3717-3733). Used for labelled
# CSV headers (``electronic_energy_(Hartree)``, …). Note: the parsers store
# ``homo_lumo_gap`` in eV (xTB prints eV; cclib computes ``lumo - homo`` on its
# eV orbital energies; OPI reads the ``:: HOMO-LUMO gap … eV ::`` line). The
# clustering matrix Z-scores every feature, so the eV/Hartree choice does not
# affect cluster IDs; how ``run_data_extraction`` labels the ``--data`` dump is
# audited in R5b (the scaling region, cosmic-v01.py 3892-4543).

FEATURE_UNITS: Mapping[str, str] = MappingProxyType(
    {
        "electronic_energy": "Hartree",
        "gibbs_free_energy": "Hartree",
        "homo_energy": "Hartree",
        "lumo_energy": "Hartree",
        "homo_lumo_gap": "Hartree",
        "dipole_moment": "Debye",
        "radius_of_gyration": "A",
        "first_vib_freq": "cm^-1",
        "last_vib_freq": "cm^-1",
        "average_hbond_distance": "A",
        "average_hbond_angle": "deg",
        "rotational_constants_A": "cm^-1",
        "rotational_constants_B": "cm^-1",
        "rotational_constants_C": "cm^-1",
    }
)


def labelled_column(name: str) -> str:
    """Return ``name_(unit)`` for CSV headers, or bare ``name`` if no unit.

    Mirrors cosmic-v01's ``_extract_labeled`` (lines 3736-3738).
    """
    unit = FEATURE_UNITS.get(name)
    return f"{name}_({unit})" if unit else name


# --------------------------------------------------------------------------- #
# Feature weights                                                              #
# --------------------------------------------------------------------------- #
#
# Tuned per-feature weights for semiempirical / standalone xTB output
# (cosmic-v01.py lines 3693-3712). Values < 1.0 down-weight noisy features so
# geometrically identical structures do not look distinct in the
# Z-standardised feature space. Activated by ``--partialweights``; user
# ``--weights`` overrides layer on top.

SEMIEMPIRICAL_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {
        "electronic_energy": 1.0,        # Direct SCF output, most reliable
        "gibbs_free_energy": 0.9,        # Thermal corrections add noise
        "homo_energy": 0.85,             # Semi-empirical methods show variance
        "lumo_energy": 0.7,              # Noisiest orbital energy, method-dependent
        "homo_lumo_gap": 0.9,            # Correlated with homo/lumo noise
        "dipole_moment": 0.6,            # Sensitive to atom indexing and coord frame
        "radius_of_gyration": 1.0,       # Purely geometric, index-independent
        "first_vib_freq": 0.9,           # Generally stable
        "last_vib_freq": 0.85,           # High-freq modes vary with method/basis
        "average_hbond_distance": 0.7,   # Depends on H-bond detection cutoffs
        "average_hbond_angle": 0.7,      # Sensitive to H-bond detection geometry
        "rotational_constants_A": 1.0,   # Geometric, index-independent
        "rotational_constants_B": 1.0,   # Geometric, index-independent
        "rotational_constants_C": 1.0,   # Geometric, index-independent
    }
)

DEFAULT_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {key: 1.0 for key in SEMIEMPIRICAL_WEIGHTS}
)
"""Uniform default weights — keeps COSMIC method-agnostic unless ``--partialweights``."""


_WEIGHT_PAIR_RE = re.compile(r"\(([^=]+)=([\d.]+)\)")


def parse_weights_argument(weight_str: str | None) -> Dict[str, float]:
    """Parse a ``--weights`` string into ``{feature_name: weight}``.

    The format is a run of ``(key=value)`` pairs, e.g.
    ``"(electronic_energy=0.1)(homo_lumo_gap=0.2)"``. An unparseable value is
    logged at WARNING and skipped — exactly cosmic-v01's behaviour
    (lines 3632-3648), except v04's ``print`` becomes a ``logging`` call
    (**D-003**).
    """
    weights: Dict[str, float] = {}
    if not weight_str:
        return weights
    for key, value in _WEIGHT_PAIR_RE.findall(weight_str):
        try:
            weights[key.strip()] = float(value.strip())
        except ValueError:
            logger.warning("Could not parse weight for '%s=%s'; skipping.", key, value)
    return weights


def parse_abs_tolerance_argument(tolerance_str: str | None) -> Dict[str, float]:
    """Parse a ``--abs-tolerance`` string into ``{feature_name: tolerance}``.

    The format is a run of ``(key=value)`` pairs, e.g.
    ``"(electronic_energy=1e-5)(dipole_moment=1e-3)"``.

    Verbatim port of cosmic-v01's ``parse_abs_tolerance_argument``
    (lines 3650-3665) — its ``print`` warning is kept verbatim (**D-003**:
    cosmic-v01's stdout is a contract).
    """
    tolerances: Dict[str, float] = {}
    if not tolerance_str:
        return tolerances

    matches = re.findall(r'\(([^=]+)=([\d\.eE-]+)\)', tolerance_str)
    for key, value in matches:
        try:
            tolerances[key.strip()] = float(value.strip())
        except ValueError:
            print(f"WARNING: Could not parse absolute tolerance for '{key}={value}'. Skipping this tolerance.")
    return tolerances


__all__ = [
    "CLUSTERING_NUMERICAL_FEATURES",
    "DEFAULT_WEIGHTS",
    "FEATURE_COLUMNS",
    "FEATURE_MAPPING",
    "FEATURE_UNITS",
    "ROTATIONAL_CONSTANT_SUBFEATURES",
    "SEMIEMPIRICAL_WEIGHTS",
    "labelled_column",
    "parse_abs_tolerance_argument",
    "parse_weights_argument",
]
