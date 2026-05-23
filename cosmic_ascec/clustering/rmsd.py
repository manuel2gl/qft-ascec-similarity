"""RMSD post-processing — geometric refinement of property-based clusters.

Two structures can have nearly identical physicochemical feature vectors yet
be genuinely different geometric arrangements (stereoisomers, mirror images,
loosely-coupled conformers). The optional RMSD pass catches that: within each
property cluster, heavy-atom RMSD is measured between every member and the
lowest-energy representative; members beyond the threshold are split out and
re-clustered on a pure RMSD distance matrix.

Public functions:

* :func:`calculate_rmsd` — heavy-atom RMSD via Kabsch alignment.
* :func:`post_process_clusters_with_rmsd` — first pass: detect outliers
  inside each property cluster.
* :func:`perform_second_rmsd_clustering` — re-cluster the outliers from the
  first pass into their own groups.

The clustering record dicts are mutated in place; energy selection is
threaded explicitly via an
:class:`~cosmic_ascec.clustering.energies.EnergyMode` argument so the same
function works for runs with and without thermochemistry.
"""

from __future__ import annotations

from typing import Any, Dict, List, MutableMapping, Sequence, Tuple

import numpy as np

from cosmic_ascec.clustering.energies import EnergyMode, sorting_energy

Record = MutableMapping[str, Any]
Cluster = List[Record]


def calculate_rmsd(
    atomnos1: Sequence[int],
    coords1: np.ndarray,
    atomnos2: Sequence[int],
    coords2: np.ndarray,
) -> Any:
    """Heavy-atom RMSD between two structures via Kabsch alignment.

    Hydrogen atoms (Z = 1) are excluded. Returns the minimised RMSD, or
    ``None`` when the heavy-atom counts differ or alignment fails.

    Verbatim port of cosmic-v01's ``calculate_rmsd`` (lines 1961-2026).
    """
    from scipy.spatial.transform import Rotation as R  # Import only when needed

    # Filter out hydrogen atoms (atomic number 1)
    heavy_indices1 = [i for i, z in enumerate(atomnos1) if z != 1]
    heavy_coords1 = coords1[heavy_indices1]

    heavy_indices2 = [i for i, z in enumerate(atomnos2) if z != 1]
    heavy_coords2 = coords2[heavy_indices2]

    if len(heavy_indices1) == 0 or len(heavy_indices2) == 0:
        return None

    if len(heavy_indices1) != len(heavy_indices2):
        return None

    # Ensure coordinates are numpy arrays and float64
    coords1_filtered = np.asarray(heavy_coords1, dtype=np.float64)
    coords2_filtered = np.asarray(heavy_coords2, dtype=np.float64)

    if coords1_filtered.shape[0] == 0 or coords2_filtered.shape[0] == 0:
        return None

    try:
        # Step 1: Center the coordinates (move to origin)
        center1 = np.mean(coords1_filtered, axis=0)
        centered_coords1 = coords1_filtered - center1

        center2 = np.mean(coords2_filtered, axis=0)
        centered_coords2 = coords2_filtered - center2

        # Step 2: Perform Kabsch alignment to find the optimal rotation
        # R.align_vectors(a, b) finds rotation + RMSD to transform a onto b.
        result = R.align_vectors(centered_coords2, centered_coords1)
        rmsd_value = result[1]  # The RMSD is always the second value returned

        return rmsd_value

    except Exception as e:
        print(f"  ERROR during heavy atom RMSD calculation: {e}")
        return None


def post_process_clusters_with_rmsd(
    initial_clusters: Sequence[Cluster],
    rmsd_validation_threshold: float,
    mode: EnergyMode,
) -> Tuple[List[Cluster], List[Record]]:
    """First RMSD pass — validate each property cluster against its representative.

    Returns ``(validated_main_clusters, individual_outliers)``: members whose
    RMSD to the lowest-energy representative exceeds the threshold are pulled
    out as individual outliers (to be re-clustered by
    :func:`perform_second_rmsd_clustering`).

    Verbatim port of cosmic-v01's ``post_process_clusters_with_rmsd``
    (lines 2028-2121). cosmic-v01's ``_sorting_energy`` global becomes the
    explicit *mode* argument (**D-007**).
    """
    validated_main_clusters: List[Cluster] = []
    individual_outliers: List[Record] = []

    print(f"  Initiating first pass RMSD validation with threshold: {rmsd_validation_threshold:.3f} Å...")

    for cluster_idx, current_property_cluster in enumerate(initial_clusters):
        if not current_property_cluster:
            continue

        if len(current_property_cluster) == 1:
            # Single member clusters are passed directly to validated_main_clusters

            current_property_cluster[0]['_rmsd_pass_origin'] = 'first_pass_validated'
            validated_main_clusters.append(current_property_cluster)
            continue

        print(f"    Validating initial property cluster {current_property_cluster[0].get('_parent_global_cluster_id', 'N/A')} with {len(current_property_cluster)} configurations...")

        # Select the lowest energy configuration as the representative for this property cluster

        representative_conf = min(current_property_cluster,
                                  key=lambda x: (sorting_energy(x, mode), x['filename']))

        current_validated_sub_cluster = [representative_conf]  # Start new validated cluster with representative
        processed_members_filenames = {representative_conf['filename']}

        representative_conf['_rmsd_pass_origin'] = 'first_pass_validated'

        coords_rep = representative_conf.get('final_geometry_coords')
        atomnos_rep = representative_conf.get('final_geometry_atomnos')

        if coords_rep is None or atomnos_rep is None:
            print(f"    WARNING: Representative {representative_conf['filename']} has missing geometry. Skipping RMSD validation for this property cluster. All members kept together for now.")
            # If skipping, mark all as first_pass_validated
            for conf_member in current_property_cluster:
                conf_member['_rmsd_pass_origin'] = 'first_pass_validated'
            validated_main_clusters.append(current_property_cluster)
            continue

        other_members = [conf for conf in current_property_cluster if conf != representative_conf]

        for conf_member in other_members:
            if conf_member['filename'] in processed_members_filenames:
                continue

            coords_member = conf_member.get('final_geometry_coords')
            atomnos_member = conf_member.get('final_geometry_atomnos')

            if coords_member is None or atomnos_member is None:
                print(f"    WARNING: {conf_member['filename']} has missing geometry data. Treating as an individual outlier for now.")
                conf_member['_parent_global_cluster_id'] = representative_conf['_parent_global_cluster_id']
                conf_member['_rmsd_pass_origin'] = 'second_pass_formed'
                individual_outliers.append(conf_member)  # Collect this as an outlier
                processed_members_filenames.add(conf_member['filename'])
                continue

            rmsd_val = calculate_rmsd(
                atomnos_rep, coords_rep,
                atomnos_member, coords_member
            )

            if rmsd_val is not None and rmsd_val <= rmsd_validation_threshold:
                current_validated_sub_cluster.append(conf_member)
                conf_member['_rmsd_pass_origin'] = 'first_pass_validated'
                processed_members_filenames.add(conf_member['filename'])
            else:
                print(f"    {conf_member['filename']} (RMSD={rmsd_val:.3f} Å) is an outlier from {representative_conf['filename']} (Threshold={rmsd_validation_threshold:.3f} Å).")
                conf_member['_parent_global_cluster_id'] = representative_conf['_parent_global_cluster_id']
                conf_member['_rmsd_pass_origin'] = 'second_pass_formed'
                individual_outliers.append(conf_member)  # Collect this as an outlier
                processed_members_filenames.add(conf_member['filename'])

        if current_validated_sub_cluster:
            validated_main_clusters.append(current_validated_sub_cluster)

    return validated_main_clusters, individual_outliers


def perform_second_rmsd_clustering(
    cluster_members_to_refine: Cluster,
    rmsd_threshold: float,
    mode: EnergyMode,
) -> List[Cluster]:
    """Second RMSD pass — re-cluster first-pass outliers on an RMSD distance matrix.

    A pairwise heavy-atom RMSD matrix feeds a UPGMA linkage; the tree is cut at
    *rmsd_threshold*. Each resulting sub-cluster records its representative and
    the RMSD of every member to it.

    Verbatim port of cosmic-v01's ``perform_second_rmsd_clustering``
    (lines 2604-2693). cosmic-v01's ``_sorting_energy`` global becomes the
    explicit *mode* argument (**D-007**).
    """
    from scipy.cluster.hierarchy import linkage, fcluster  # Import only when needed

    if len(cluster_members_to_refine) <= 1:
        for m in cluster_members_to_refine:
            m['_second_rmsd_sub_cluster_id'] = m.get('_initial_cluster_label')
            m['_second_rmsd_context_listing'] = [{'filename': m['filename'], 'rmsd_to_rep': 0.0}]
            m['_second_rmsd_rep_filename'] = m['filename']
            m['_rmsd_pass_origin'] = 'second_pass_formed'
        return [[m] for m in cluster_members_to_refine]

    num_members = len(cluster_members_to_refine)
    rmsd_matrix = np.zeros((num_members, num_members))

    for i in range(num_members):
        for j in range(i + 1, num_members):
            conf1 = cluster_members_to_refine[i]
            conf2 = cluster_members_to_refine[j]

            coords1 = conf1.get('final_geometry_coords')
            atomnos1 = conf1.get('final_geometry_atomnos')
            coords2 = conf2.get('final_geometry_coords')
            atomnos2 = conf2.get('final_geometry_atomnos')

            rmsd = calculate_rmsd(atomnos1, coords1, atomnos2, coords2)
            if rmsd is None:
                rmsd = float('inf')
            rmsd_matrix[i, j] = rmsd_matrix[j, i] = rmsd

    condensed_distances = []
    for i in range(num_members):
        for j in range(i + 1, num_members):
            condensed_distances.append(rmsd_matrix[i, j])

    if not condensed_distances:
        for m in cluster_members_to_refine:
            m['_second_rmsd_sub_cluster_id'] = m.get('_initial_cluster_label')
            m['_second_rmsd_context_listing'] = [{'filename': m['filename'], 'rmsd_to_rep': 0.0}]
            m['_second_rmsd_rep_filename'] = m['filename']
            m['_rmsd_pass_origin'] = 'second_pass_formed'
        return [[m] for m in cluster_members_to_refine]

    linkage_matrix = linkage(condensed_distances, method='average', metric='euclidean')
    second_cluster_labels = fcluster(linkage_matrix, t=rmsd_threshold, criterion='distance')

    second_level_clusters_data: Dict[Any, Cluster] = {}
    for i, label in enumerate(second_cluster_labels):
        cluster_members_to_refine[i]['_second_rmsd_sub_cluster_id'] = label
        cluster_members_to_refine[i]['_rmsd_pass_origin'] = 'second_pass_formed'
        second_level_clusters_data.setdefault(label, []).append(cluster_members_to_refine[i])

    final_sub_clusters: List[Cluster] = []
    for label, sub_cluster_members in second_level_clusters_data.items():
        if not sub_cluster_members:
            continue

        sub_cluster_rep = min(sub_cluster_members,
                              key=lambda x: (sorting_energy(x, mode), x['filename']))

        sub_cluster_rmsd_listing = []
        if sub_cluster_rep.get('final_geometry_coords') is not None and sub_cluster_rep.get('final_geometry_atomnos') is not None:
            for member_conf in sub_cluster_members:
                if member_conf == sub_cluster_rep:
                    rmsd_val = 0.0
                else:
                    rmsd_val = calculate_rmsd(
                        sub_cluster_rep['final_geometry_atomnos'], sub_cluster_rep['final_geometry_coords'],
                        member_conf['final_geometry_atomnos'], member_conf['final_geometry_coords']
                    )
                sub_cluster_rmsd_listing.append({'filename': member_conf['filename'], 'rmsd_to_rep': rmsd_val})
        else:
            for member_conf in sub_cluster_members:
                sub_cluster_rmsd_listing.append({'filename': member_conf['filename'], 'rmsd_to_rep': None})

        for member_conf in sub_cluster_members:
            member_conf['_second_rmsd_context_listing'] = sub_cluster_rmsd_listing
            member_conf['_second_rmsd_rep_filename'] = sub_cluster_rep['filename']

        final_sub_clusters.append(sub_cluster_members)

    return final_sub_clusters


__all__ = [
    "calculate_rmsd",
    "perform_second_rmsd_clustering",
    "post_process_clusters_with_rmsd",
]
