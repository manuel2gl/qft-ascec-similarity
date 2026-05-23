"""COSMIC clustering pipeline.

The clustering stage is the back half of the COSMIC ASCEC protocol. After
annealing/optimization has produced many candidate structures, COSMIC reads
their QM outputs, builds a physicochemical feature vector per structure, and
groups them by hierarchical (UPGMA) clustering into a small set of
representative *motifs*.

Submodule responsibilities (each one is a self-contained step you can change
without touching the others):

* :mod:`~cosmic_ascec.clustering.features` — QM outputs → feature dicts
  (energy, dipole, rotational constants, etc.) per structure.
* :mod:`~cosmic_ascec.clustering.console` — shared verbose/print helpers.
* :mod:`~cosmic_ascec.clustering.scaling` — Z-standardisation and
  per-feature weights (``--weights``, ``--partialweights``).
* :mod:`~cosmic_ascec.clustering.thresholds` — choosing the dendrogram cut
  height (mojena / knee / explicit).
* :mod:`~cosmic_ascec.clustering.rmsd` — geometric RMSD post-processing
  that splits over-merged clusters.
* :mod:`~cosmic_ascec.clustering.filters` — drop imaginary-frequency and
  non-converged structures before clustering.
* :mod:`~cosmic_ascec.clustering.matching` — reduced-feature-tier matcher
  for structures missing some properties.
* :mod:`~cosmic_ascec.clustering.dat_writer` — per-cluster ``.dat`` reports.
* :mod:`~cosmic_ascec.clustering.motifs` — motif directories and
  representative ``.xyz`` files.
* :mod:`~cosmic_ascec.clustering.dendrogram` — dendrogram and threshold
  diagnostic plots (``--diagram``).
* :mod:`~cosmic_ascec.clustering.data_extraction` — ``--data`` dump of the
  raw feature matrix.
* :mod:`~cosmic_ascec.clustering.composite_energies` — composite-Gibbs
  reconstruction ``G = E_eref + (G_prev − E_prev)``.
* :mod:`~cosmic_ascec.clustering.orchestrator` — the top-level driver that
  threads all of the above into one ``perform_clustering_and_analysis`` call.
"""

from cosmic_ascec.clustering.energies import EnergyMode, sorting_energy
from cosmic_ascec.clustering.features import (
    FEATURE_COLUMNS,
    extract_features_from_file,
    extract_properties_from_logfile,
)
from cosmic_ascec.clustering.orchestrator import (
    get_cpu_count_fast,
    perform_clustering_and_analysis,
)
from cosmic_ascec.clustering.scaling import (
    apply_weights,
    build_feature_vectors,
    zscore_scale,
)
from cosmic_ascec.clustering.thresholds import resolve_clustering_threshold

__all__ = [
    "EnergyMode",
    "FEATURE_COLUMNS",
    "apply_weights",
    "build_feature_vectors",
    "extract_features_from_file",
    "extract_properties_from_logfile",
    "get_cpu_count_fast",
    "perform_clustering_and_analysis",
    "resolve_clustering_threshold",
    "sorting_energy",
    "zscore_scale",
]
