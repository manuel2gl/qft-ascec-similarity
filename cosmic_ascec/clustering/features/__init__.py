"""COSMIC feature extraction — QM outputs → cosmic-v01 property dicts.

This subpackage is the front half of the clustering pipeline. R5 (D-039
faithful decomposition) re-ports cosmic-v01's clustering feature front-end
verbatim: every clustering candidate is parsed by one of three standalone
parsers, returning cosmic-v01's free-form ``extracted_props`` dict. The pre-R5
v05 routed feature extraction through ascec-v04's annealing QM adapters
(``parse_result`` / ``QMResult.extras``) and assembled a ``pandas`` DataFrame —
neither is on cosmic-v01's clustering path; that was a divergence, now removed.

Layout:

* :mod:`~cosmic_ascec.clustering.features.geometric` — geometry maths and
  geometry-derived features (radius of gyration, rotational constants,
  hydrogen bonds), cosmic-v01's own copies.
* :mod:`~cosmic_ascec.clustering.features.parsers` — the three per-package
  property parsers (``extract_properties_with_{xtb,cclib,opi}``).
* :mod:`~cosmic_ascec.clustering.features.extractor` — file-type detection and
  parser dispatch (``extract_properties_from_logfile``).
* :mod:`~cosmic_ascec.clustering.features.feature_spec` — the parity contract:
  the pinned :data:`FEATURE_COLUMNS` order, ``FEATURE_MAPPING``, weights, units.
"""

from cosmic_ascec.clustering.features.extractor import (
    choose_parser,
    detect_file_type,
    detect_orca_version,
    extract_features_from_file,
    extract_properties_from_logfile,
    list_qm_outputs,
    process_file_parallel_wrapper,
)
from cosmic_ascec.clustering.features.feature_spec import (
    CLUSTERING_NUMERICAL_FEATURES,
    DEFAULT_WEIGHTS,
    FEATURE_COLUMNS,
    FEATURE_MAPPING,
    FEATURE_UNITS,
    ROTATIONAL_CONSTANT_SUBFEATURES,
    SEMIEMPIRICAL_WEIGHTS,
    labelled_column,
    parse_weights_argument,
)
from cosmic_ascec.clustering.features.geometric import (
    atomic_number_to_symbol,
    calculate_radius_of_gyration,
    calculate_rotational_constants,
    detect_hydrogen_bonds,
    extract_xyz_from_output,
)
from cosmic_ascec.clustering.features.parsers import (
    extract_properties_with_cclib,
    extract_properties_with_opi,
    extract_properties_with_xtb,
)

__all__ = [
    "CLUSTERING_NUMERICAL_FEATURES",
    "DEFAULT_WEIGHTS",
    "FEATURE_COLUMNS",
    "FEATURE_MAPPING",
    "FEATURE_UNITS",
    "ROTATIONAL_CONSTANT_SUBFEATURES",
    "SEMIEMPIRICAL_WEIGHTS",
    "atomic_number_to_symbol",
    "calculate_radius_of_gyration",
    "calculate_rotational_constants",
    "choose_parser",
    "detect_file_type",
    "detect_hydrogen_bonds",
    "detect_orca_version",
    "extract_features_from_file",
    "extract_properties_from_logfile",
    "extract_properties_with_cclib",
    "extract_properties_with_opi",
    "extract_properties_with_xtb",
    "extract_xyz_from_output",
    "labelled_column",
    "list_qm_outputs",
    "parse_weights_argument",
    "process_file_parallel_wrapper",
]
