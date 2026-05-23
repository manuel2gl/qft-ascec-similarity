"""Geometry layer: molecular templates, box sizing, overlap, placement, bonds, inertia.

The package re-exports the public surface of every submodule so the rest of
``cosmic_ascec`` can ``from cosmic_ascec.geometry import Cluster, ...`` without
caring about the internal split. Anything not listed in ``__all__`` is private.
"""

from cosmic_ascec.geometry.bonds import (
    BOND_DISTANCE_TOLERANCE,
    find_connected_atoms,
    find_rotatable_bonds,
)
from cosmic_ascec.geometry.box import (
    BoxLengthRecommendation,
    BoxLengthReport,
    HBondAnalysis,
    calculate_hydrogen_bond_potential,
    calculate_molecular_extent,
    calculate_molecular_volume,
    calculate_optimal_box_length,
    has_primary_hydrogen_bonds,
)
from cosmic_ascec.geometry.inertia import (
    calculate_inertia_tensor,
    calculate_mass_center,
    calculate_principal_moments,
    calculate_radius_of_gyration,
    calculate_rotational_constants,
)
from cosmic_ascec.geometry.molecule import Cluster, Molecule
from cosmic_ascec.geometry.overlap import (
    DEFAULT_INTERMOLECULAR_OVERLAP_SCALE,
    check_intermolecular_overlap,
    check_intramolecular_overlap,
)
from cosmic_ascec.geometry.placement import (
    PlacementOutcome,
    PlacementResult,
    PlacementSettings,
    initialize_cluster,
)


__all__ = [
    "BOND_DISTANCE_TOLERANCE",
    "BoxLengthRecommendation",
    "BoxLengthReport",
    "Cluster",
    "DEFAULT_INTERMOLECULAR_OVERLAP_SCALE",
    "HBondAnalysis",
    "Molecule",
    "PlacementOutcome",
    "PlacementResult",
    "PlacementSettings",
    "calculate_hydrogen_bond_potential",
    "calculate_inertia_tensor",
    "calculate_mass_center",
    "calculate_molecular_extent",
    "calculate_molecular_volume",
    "calculate_optimal_box_length",
    "calculate_principal_moments",
    "calculate_radius_of_gyration",
    "calculate_rotational_constants",
    "check_intermolecular_overlap",
    "check_intramolecular_overlap",
    "find_connected_atoms",
    "find_rotatable_bonds",
    "has_primary_hydrogen_bonds",
    "initialize_cluster",
]
