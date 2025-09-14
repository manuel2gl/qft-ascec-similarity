import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from cclib.io import ccread
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import glob
import re
import subprocess
import shutil
import pickle # Added for caching
from scipy.spatial.transform import Rotation as R # Ensure R is imported like this!


# Constants for Boltzmann distribution
BOLTZMANN_CONSTANT_HARTREE_PER_K = 3.1668114e-6 # Hartree/K (k_B in atomic units)
DEFAULT_TEMPERATURE_K = 298.15 # K (Room temperature)

version = "* Similarity-v01: Jun-2025 *"  # Version of the Similarity script

### Embedded element masses dictionary ###
element_masses = {
    "H": 1.008, "He": 4.0026, "Li": 6.94, "Be": 9.012, "B": 10.81,
    "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Ne": 20.180,
    "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085, "P": 30.974,
    "S": 32.06, "Cl": 35.45, "Ar": 39.948, "K": 39.098, "Ca": 40.078,
    "Sc": 44.956, "Ti": 47.867, "V": 50.942, "Cr": 51.996, "Mn": 54.938,
    "Fe": 55.845, "Ni": 58.693, "Cu": 63.546, "Zn": 65.38, "Ga": 69.723,
    "Ge": 72.630, "As": 74.922, "Se": 78.971, "Br": 79.904, "Kr": 83.798,
    "Rb": 85.468, "Sr": 87.62, "Y": 88.906, "Zr": 91.224, "Nb": 92.906,
    "Mo": 95.96, "Ru": 101.07, "Rh": 102.906, "Pd": 106.42, "Ag": 107.868,
    "Cd": 112.414, "In": 114.818, "Sn": 118.710, "Sb": 121.760, "Te": 127.60,
    "I": 126.904, "Xe": 131.293, "Cs": 132.905, "Ba": 137.327, "La": 138.905,
    "Ce": 140.116, "Pr": 140.908, "Nd": 144.242, "Sm": 150.36, "Eu": 151.964,
    "Gd": 157.25, "Tb": 158.925, "Dy": 162.500, "Ho": 164.930, "Er": 167.259,
    "Tm": 168.934, "Yb": 173.04, "Lu": 174.967, "Hf": 178.49, "Ta": 180.948,
    "W": 183.84, "Re": 186.207, "Os": 190.23, "Ir": 192.217, "Pt": 195.084,
    "Au": 196.967, "Hg": 200.592, "Tl": 204.383, "Pb": 207.2, "Bi": 208.980,
    "Th": 232.038, "Pa": 231.036, "U": 238.028
}

def atomic_number_to_symbol(atomic_number):
    """
    Converts an atomic number to its corresponding element symbol.
    This function is used consistently throughout the script for element symbol retrieval.
    """
    # Corrected and complete periodic table symbols list
    periodic_table_symbols = [
        "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga",
        "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr", "Nb",
        "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
        "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm",
        "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
        "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
        "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa",
        "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md",
        "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg",
        "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og"
    ]
    if 0 <= atomic_number < len(periodic_table_symbols):
        return periodic_table_symbols[atomic_number]
    else:
        # Fallback for unknown atomic numbers
        return str(atomic_number)

def calculate_deviation_percentage(values):
    """Calculates the percentage deviation (max-min / abs(mean)) for a list of numerical values."""
    if not values or len(values) < 2:
        return 0.0 # Or None, depending on desired behavior for single/no values
    
    # Filter out None values before calculating min/max
    numeric_values = [v for v in values if v is not None]
    if not numeric_values:
        return 0.0

    min_val = min(numeric_values)
    max_val = max(numeric_values)

    if min_val == 0.0 and max_val == 0.0:
        return 0.0 # All zeros, no deviation

    if max_val == min_val:
        return 0.0

    mean_val = np.mean(numeric_values)
    if mean_val == 0.0: # Avoid division by zero if mean is zero
        return (max_val - min_val) / abs(mean_val) * 100.0 if max_val != min_val else 0.0
    
    return ((max_val - min_val) / abs(mean_val)) * 100.0


def calculate_rms_deviation(values):
    """Calculates the Root Mean Square Deviation (RMSD) for a list of numerical values."""
    if not values:
        return 0.0
    
    mean_val = np.mean(values)
    squared_deviations = [(val - mean_val)**2 for val in values]
    rmsd = np.sqrt(np.mean(squared_deviations))
    
    return rmsd

def calculate_radius_of_gyration(atomnos, atomcoords):
    """Calculates the radius of gyration for a molecule."""
    try:
        symbols = [atomic_number_to_symbol(n) for n in atomnos]
        masses = np.array([element_masses.get(s, 0.0) for s in symbols])
        
        coords = np.asarray(atomcoords)

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected atomcoords to be (N, 3), but got shape {coords.shape}")

        total_mass = np.sum(masses)
        if total_mass == 0.0:
            return None
        
        center_of_mass = np.sum(coords.T * masses, axis=1) / total_mass
        
        rg_squared = np.sum(masses * np.sum((coords - center_of_mass)**2, axis=1)) / total_mass
        return np.sqrt(rg_squared)
    except Exception as e:
        print(f"Error in calculate_radius_of_gyration: {e}")
        return None

def detect_hydrogen_bonds(atomnos, atomcoords):
    """
    Detects hydrogen bonds based on static distance and angle criteria.
    All potential bonds (based on distance) are recorded in hbond_details,
    but only those meeting the angle criterion (>= 30 degrees) are counted
    towards num_hydrogen_bonds and related statistics.
    """
    try:
        coords = np.asarray(atomcoords)

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected atom coords to be (N, 3), but got shape {coords.shape}")

        # Define potential donor (D) and acceptor (A) atoms by atomic number (N, O, F)
        potential_donor_acceptor_z = {7, 8, 9}
        hydrogen_atom_num = 1 # Atomic number for Hydrogen (H)

        # Static H...A distance criteria
        HB_min_dist_actual = 1.2
        HB_max_dist_actual = 2.7
        
        # Covalent D-H distance search limit
        COVALENT_DH_SEARCH_DIST = 1.5 

        # Angle criterion for actual HB counting
        HB_min_angle_actual = 30.0 # User-specified minimum angle
        HB_max_angle_actual = 180.0 # User-specified maximum angle (for completeness)

        symbols = [atomic_number_to_symbol(n) for n in atomnos]
        atom_labels = [f"{sym}{i+1}" for i, sym in enumerate(symbols)]
        
        all_potential_hbonds_details = [] # Stores all potential H-bonds for review in .dat file
        
        # First Pass: Identify the covalently bonded donor (D) for each hydrogen (H)
        h_covalent_donors = {} # Stores {h_idx: (donor_idx, D-H_distance)}
        for i_h, h_atom_num in enumerate(atomnos):
            if h_atom_num != hydrogen_atom_num:
                continue

            coord_h = coords[i_h]
            
            min_dist_dh = float('inf')
            donor_idx_for_h = -1
            
            for i_d, d_atom_num in enumerate(atomnos):
                if i_d == i_h:
                    continue

                if d_atom_num in potential_donor_acceptor_z:
                    dist_dh = np.linalg.norm(coords[i_d] - coord_h)
                    
                    if dist_dh < min_dist_dh:
                        min_dist_dh = dist_dh
                        donor_idx_for_h = i_d
            
            # FIX: Changed 'donor_idx_for_covalently_bonded_h' to 'donor_idx_for_h'
            if donor_idx_for_h != -1 and min_dist_dh <= COVALENT_DH_SEARCH_DIST:
                h_covalent_donors[i_h] = (donor_idx_for_h, min_dist_dh)

        # Second Pass: Detect all potential H-bonds (distance only) and calculate angles
        for i_h, h_atom_num in enumerate(atomnos):
            if h_atom_num != hydrogen_atom_num:
                continue

            if i_h not in h_covalent_donors:
                continue

            donor_idx, actual_dh_covalent_distance = h_covalent_donors[i_h]
            coord_h = coords[i_h]
            coord_d = coords[donor_idx]

            for i_a, a_atom_num in enumerate(atomnos):
                if i_a == i_h or i_a == donor_idx:
                    continue

                if a_atom_num in potential_donor_acceptor_z:
                    coord_a = coords[i_a]

                    dist_ha = np.linalg.norm(coord_h - coord_a)
                    
                    if HB_min_dist_actual <= dist_ha <= HB_max_dist_actual:
                        vec_h_d = coord_d - coord_h
                        vec_h_a = coord_a - coord_h
                        
                        norm_vec_h_d = np.linalg.norm(vec_h_d)
                        norm_vec_h_a = np.linalg.norm(vec_h_a)

                        if norm_vec_h_d == 0 or norm_vec_h_a == 0:
                            angle_deg = 0.0
                        else:
                            dot_product = np.dot(vec_h_d, vec_h_a)
                            cos_angle = dot_product / (norm_vec_h_d * norm_vec_h_a)
                            cos_angle = np.clip(cos_angle, -1.0, 1.0)
                            angle_rad = np.arccos(cos_angle)
                            angle_deg = np.degrees(angle_rad)

                        # Always add to all_potential_hbonds_details for review
                        all_potential_hbonds_details.append({
                            'donor_atom_label': atom_labels[donor_idx],
                            'hydrogen_atom_label': atom_labels[i_h],
                            'acceptor_atom_label': atom_labels[i_a],
                            'H...A_distance': dist_ha,
                            'D-H...A_angle': angle_deg,
                            'D-H_covalent_distance': actual_dh_covalent_distance
                        })
        
        # Filter for bonds meeting the angle criterion for counting and statistics
        filtered_hbonds_for_stats = [
            b for b in all_potential_hbonds_details 
            if b['D-H...A_angle'] >= HB_min_angle_actual and b['D-H...A_angle'] <= HB_max_angle_actual
        ]

        # Populate extracted_props based on the filtered list for counting and stats
        extracted_props = {
            'num_hydrogen_bonds': len(filtered_hbonds_for_stats),
            'hbond_details': all_potential_hbonds_details, # All potential bonds for review
            'average_hbond_distance': None,
            'min_hbond_distance': None,
            'max_hbond_distance': None,
            'std_hbond_distance': None,
            'average_hbond_angle': None,
            'min_hbond_angle': None,
            'max_hbond_angle': None,
            'std_hbond_angle': None
        }

        if filtered_hbonds_for_stats:
            distances = [bond['H...A_distance'] for bond in filtered_hbonds_for_stats]
            angles = [bond['D-H...A_angle'] for bond in filtered_hbonds_for_stats]
            
            extracted_props['average_hbond_distance'] = np.mean(distances)
            extracted_props['min_hbond_distance'] = np.min(distances)
            extracted_props['max_hbond_distance'] = np.max(distances)
            extracted_props['std_hbond_distance'] = np.std(distances) if len(distances) > 1 else 0.0

            extracted_props['average_hbond_angle'] = np.mean(angles)
            extracted_props['min_hbond_angle'] = np.min(angles)
            extracted_props['max_hbond_angle'] = np.max(angles)
            extracted_props['std_hbond_angle'] = np.std(angles) if len(angles) > 1 else 0.0

        return extracted_props

    except Exception as e:
        print(f"  DEBUG: Error in detect_hydrogen_bonds: {e}")
        return {
            'num_hydrogen_bonds': 0,
            'hbond_details': [],
            'average_hbond_distance': None, 'min_hbond_distance': None, 
            'max_hbond_distance': None, 'std_hbond_distance': None,
            'average_hbond_angle': None, 'min_hbond_angle': None, 
            'max_hbond_angle': None, 'std_hbond_angle': None
        }

def extract_properties_from_logfile(logfile_path):
    data = None # Initialize data to None

    try:
        raw_data = ccread(logfile_path) # Read the raw data, could be ccData or ccCollection
        
        if raw_data:
            # Determine the actual ccData object to work with
            if hasattr(raw_data, 'data') and isinstance(raw_data.data, list) and len(raw_data.data) > 0:
                # This is a ccCollection, extract the last ccData object for optimization results
                data = raw_data.data[-1]
            elif type(raw_data).__name__.startswith('ccData'): # Broaden check to include subclasses like ccData_optdone_bool
                data = raw_data
            else:
                # This branch catches cases where raw_data is not a recognized ccData type or ccCollection
                print(f"  WARNING: Unexpected cclib return type for {os.path.basename(logfile_path)}. Type: {type(raw_data)}. Returning None.")
                return None
        else:
            # ccread returned None
            print(f"  WARNING: cclib returned None for {os.path.basename(logfile_path)}. This means the file could not be parsed. Returning None.")
            return None
        
        # Final explicit check: ensure 'data' is indeed a ccData object before proceeding
        if not type(data).__name__.startswith('ccData'):
            print(f"  ERROR: 'data' is not a ccData object after initial processing for {os.path.basename(logfile_path)}. Actual type: {type(data)}. Returning None.")
            return None

    except Exception as e:
        # This catches any errors during ccread or initial data extraction from ccCollection
        print(f"(CCL_ERROR) Failed to parse or process {os.path.basename(logfile_path)} with cclib: {e}")
        return None

    extracted_props = {
        'filename': os.path.basename(logfile_path),
        'method': "Unknown",
        'functional': "Unknown",
        'basis_set': "Unknown",
        'charge': None,
        'multiplicity': None,
        'num_atoms': 0,
        'final_geometry_atomnos': None,
        'final_geometry_coords': None,
        'final_electronic_energy': None,
        'gibbs_free_energy': None,
        'entropy': None, 
        'homo_energy': None,
        'lumo_energy': None,
        'homo_lumo_gap': None,
        'dipole_moment': None,
        'rotational_constants': None,
        'radius_of_gyration': None,
        'num_imaginary_freqs': None, # Changed default to None
        'first_vib_freq': None,
        'last_vib_freq': None,
        'num_hydrogen_bonds': 0, # Will be set by detect_hydrogen_bonds
        'hbond_details': [],     # Will be set by detect_hydrogen_bonds
        'average_hbond_distance': None,
        'min_hbond_distance': None,
        'max_hbond_distance': None,
        'std_hbond_distance': None,
        'average_hbond_angle': None,
        'min_hbond_angle': None,
        'max_hbond_angle': None,
        'std_hbond_angle': None,
        '_has_freq_calc': False, # New internal flag
        '_initial_cluster_label': None, # To store the integer label of the initial property cluster (from fcluster)
        '_parent_global_cluster_id': None, # New: Stores the 'Y' from "Validated from Cluster Y"
        '_first_rmsd_context_listing': None, # To store details of the original property cluster (RMSD to its rep)
        '_second_rmsd_sub_cluster_id': None, # New label for sub-cluster after 2nd RMSD
        '_second_rmsd_context_listing': None, # RMSD to representative of 2nd-level sub-cluster
        '_second_rmsd_rep_filename': None, # Filename of representative of 2nd-level sub-cluster
        '_rmsd_pass_origin': None # New flag to indicate if cluster came from 1st or 2nd RMSD pass
    }

    try:
        with open(logfile_path, 'r') as f_log:
            lines = f_log.readlines()

        # Determine file type based on extension
        file_extension = os.path.splitext(logfile_path)[1].lower()

        # These lines will now correctly access attributes from a ccData object
        methods_list = data.metadata.get("methods", [])
        extracted_props['method'] = methods_list[0] if methods_list else "Unknown"

        extracted_props['functional'] = data.metadata.get("functional", "Unknown")
        extracted_props['basis_set'] = data.metadata.get("basis_set", "Unknown")

        extracted_props['charge'] = getattr(data, 'charge', None)
        extracted_props['multiplicity'] = getattr(data, 'mult', None)

        if hasattr(data, 'atomnos') and data.atomnos is not None and len(data.atomnos) > 0:
            extracted_props['num_atoms'] = len(data.atomnos)
            extracted_props['final_geometry_atomnos'] = data.atomnos
        else:
            print(f"  WARNING: No atom numbers (atomnos) found for {os.path.basename(logfile_path)}")

        if hasattr(data, 'atomcoords') and data.atomcoords is not None and len(data.atomcoords) > 0:
            coords_candidate = None
            if isinstance(data.atomcoords, list) and len(data.atomcoords) > 0:
                coords_candidate = np.asarray(data.atomcoords[-1])
            elif isinstance(data.atomcoords, np.ndarray) and data.atomcoords.ndim == 3:
                coords_candidate = data.atomcoords[-1]
            elif isinstance(data.atomcoords, np.ndarray) and data.atomcoords.ndim == 2 and data.atomcoords.shape[1] == 3:
                coords_candidate = data.atomcoords
            
            if coords_candidate is not None and coords_candidate.ndim == 2 and coords_candidate.shape[1] == 3:
                extracted_props['final_geometry_coords'] = coords_candidate
            elif coords_candidate is not None:
                print(f"  WARNING: Final geometry candidate has wrong shape for {os.path.basename(logfile_path)}: {coords_candidate.shape}. Skipping geometry-dependent calculations.")
                extracted_props['final_geometry_coords'] = None
            else:
                print(f"  WARNING: atomcoords found but unexpected format for {os.path.basename(logfile_path)}. Expected (N,3), (steps,N,3) or list of (N,3) arrays. Actual shape: {getattr(data.atomcoords, 'shape', 'N/A')}. Skipping geometry-dependent calculations.")
                extracted_props['final_geometry_coords'] = None
        else:
            print(f"  WARNING: No atomcoords found for {os.path.basename(logfile_path)}")

        # --- Conditional ELECTRONIC ENERGY EXTRACTION based on file type ---
        if file_extension == '.out':
            # ORCA .out specific parsing for Final Electronic Energy
            # Iterate lines in reverse to find the last valid energy before OPTIMIZATION RUN DONE
            temp_final_electronic_energy = None
            optimization_done_found_in_reverse = False
            for line in reversed(lines):
                if "*** OPTIMIZATION RUN DONE ***" in line:
                    optimization_done_found_in_reverse = True
                    continue

                if "FINAL SINGLE POINT ENERGY" in line and not optimization_done_found_in_reverse:
                    try:
                        temp_final_electronic_energy = float(line.split()[-1])
                        break
                    except (ValueError, IndexError):
                        pass
                elif "Electronic energy" in line and "..." in line and not optimization_done_found_in_reverse:
                    if temp_final_electronic_energy is None:
                        try:
                            temp_final_electronic_energy = float(line.split()[-2])
                            break
                        except (ValueError, IndexError):
                            pass
            extracted_props['final_electronic_energy'] = temp_final_electronic_energy
        elif file_extension == '.log':
            # Original .log file parsing (SCF Done)
            # Iterate to capture the LAST SCF Done energy
            for line in lines:
                if "SCF Done" in line:
                    parts = line.strip().split()
                    if "=" in parts:
                        idx = parts.index("=")
                        try:
                            extracted_props['final_electronic_energy'] = float(parts[idx + 1])
                        except (ValueError, IndexError):
                            pass


        # --- Conditional GIBBS FREE ENERGY EXTRACTION based on file type ---
        if file_extension == '.out':
            # ORCA .out specific parsing for Gibbs Free Energy
            for line in lines:
                if "Final Gibbs free energy" in line:
                    try:
                        extracted_props['gibbs_free_energy'] = float(line.split()[-2])
                        break
                    except (ValueError, IndexError):
                        pass
        elif file_extension == '.log':
            # Original .log file parsing
            # Iterate to capture the LAST Gibbs Free Energy
            for line in lines:
                if "Sum of electronic and thermal Free Energies=" in line:
                    try:
                        extracted_props['gibbs_free_energy'] = float(line.strip().split()[-1])
                    except ValueError:
                        pass

        # --- Conditional ENTROPY EXTRACTION based on file type ---
        if file_extension == '.out':
            # ORCA .out specific parsing for Entropy
            for line in lines:
                if "Total Entropy" in line:
                    try:
                        extracted_props['entropy'] = float(line.split()[-2])
                        break
                    except (ValueError, IndexError):
                        pass
        elif file_extension == '.log':
            # Gaussian .log specific parsing for Entropy (usually part of thermochemistry)
            # Iterate to capture the LAST Entropy
            for line in lines:
                if "Total Entropy" in line: # Common pattern for Gaussian
                    try:
                        extracted_props['entropy'] = float(line.split()[-2])
                    except (ValueError, IndexError):
                        pass

        # --- Conditional DIPOLE MOMENT EXTRACTION based on file type ---
        if file_extension == '.out':
            # ORCA .out specific parsing (regex)
            dipole_re = re.compile(r"Total Dipole Moment\s*:\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)")
            for line in lines:
                match = dipole_re.search(line)
                if match:
                    try:
                        x = float(match.group(1))
                        y = float(match.group(2))
                        z = float(match.group(3))
                        extracted_props['dipole_moment'] = np.linalg.norm([x, y, z])
                        break
                    except ValueError:
                        pass
        elif file_extension == '.log':
            # Original .log file parsing (X=, Y=, Z=)
            # Iterate to capture the LAST Dipole Moment
            current_dipole = None # Use a temporary variable to hold the latest found value
            for i, line in enumerate(lines):
                if "Dipole moment" in line:
                    for j in range(i+1, min(i+6, len(lines))):
                        if ('X=' in lines[j]) and ('Y=' in lines[j]) and ('Z=' in lines[j]):
                            parts = lines[j].replace('=', ' ').split()
                            try:
                                x_index = parts.index('X') + 1
                                y_index = parts.index('Y') + 1
                                z_index = parts.index('Z') + 1
                                x = float(parts[x_index])
                                y = float(parts[y_index])
                                z = float(parts[z_index])
                                current_dipole = np.linalg.norm([x, y, z])
                            except (ValueError, IndexError):
                                current_dipole = None
                            break # Break from inner loop after finding XYZ
            extracted_props['dipole_moment'] = current_dipole # Assign the last found value
        # --- END Conditional DIPOLE MOMENT EXTRACTION ---


        # Extract HOMO/LUMO energies and gap using cclib ---
        # This part relies on cclib, which generally extracts the final values correctly for Gaussian.
        # Corrected: Changed 'has_a_lumo_energy' back to 'hasattr(data, "moenergies")')
        if hasattr(data, "homos") and data.homos and \
           hasattr(data, "moenergies") and data.moenergies and len(data.moenergies) > 0: 
            try:
                if len(data.homos) > 0 and isinstance(data.moenergies[0], (list, np.ndarray)) and len(data.moenergies[0]) > 0:
                    homo_index = data.homos[0]
                    if homo_index >= 0 and (homo_index + 1) < len(data.moenergies[0]):
                        homo_energy = data.moenergies[0][homo_index]
                        lumo_energy = data.moenergies[0][homo_index + 1]
                        extracted_props['homo_energy'] = homo_energy
                        extracted_props['lumo_energy'] = lumo_energy
                        extracted_props['homo_lumo_gap'] = lumo_energy - homo_energy
                    else:
                        print(f"  WARNING: HOMO/LUMO indices out of bounds or invalid moenergies configuration for {os.path.basename(logfile_path)}")
                else:
                    print(f"  WARNING: 'homos' array or 'moenergies[0]' is empty/invalid for {os.path.basename(logfile_path)}")
            except Exception as e:
                print(f"  ERROR: Problem extracting HOMO/LUMO with cclib for {os.path.basename(logfile_path)}: {e}. Trying custom parse for gap.")
        else:
            print(f"  WARNING: Missing 'homos' or 'moenergies' attributes for {os.path.basename(logfile_path)}")

        # Custom parsing for HOMO-LUMO Gap if cclib fails for .out files (e.g., semiempirical)
        if extracted_props['homo_lumo_gap'] is None and file_extension == '.out':
            homo_lumo_gap_re = re.compile(r":: HOMO-LUMO gap\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*eV\s*::") # Adjusted regex
            for line in lines:
                match = homo_lumo_gap_re.search(line)
                if match:
                    try:
                        extracted_props['homo_lumo_gap'] = float(match.group(1))
                        break
                    except ValueError:
                        pass

        # --- Conditional ROTATIONAL CONSTANTS EXTRACTION based on file type ---
        # First, try cclib for all file types
        # This part also relies on cclib for rotconsts, which should get the final ones.
        if hasattr(data, "rotconsts") and data.rotconsts is not None and len(data.rotconsts) > 0:
            rot_consts_candidate = None
            try:
                if isinstance(data.rotconsts, list) and len(data.rotconsts) > 0:
                    rot_consts_candidate = np.asarray(data.rotconsts[0])
                elif isinstance(data.rotconsts, np.ndarray) and data.rotconsts.ndim > 0:
                    if data.rotconsts.ndim > 1:
                        rot_consts_candidate = data.rotconsts[-1]
                    else:
                        rot_consts_candidate = data.rotconsts
                
                if rot_consts_candidate is not None and rot_consts_candidate.ndim == 1 and len(rot_consts_candidate) == 3:
                    extracted_props['rotational_constants'] = rot_consts_candidate.astype(float)
                else:
                    print(f"  WARNING: cclib found rotational constants but unexpected format for {os.path.basename(logfile_path)}. Expected 1D array of 3 floats. Actual format: {type(data.rotconsts)}, Shape: {getattr(data.rotconsts, 'shape', 'N/A')}. Trying custom parse.")
                    extracted_props['rotational_constants'] = None
            except Exception as e:
                print(f"  ERROR: Problem extracting rotational constants with cclib for {os.path.basename(logfile_path)}: {e}. Trying custom parse.")
                extracted_props['rotational_constants'] = None

        # Fallback to custom string parsing ONLY for .out files if cclib fails or returns unexpected format
        if extracted_props['rotational_constants'] is None and file_extension == '.out':
            rot_const_re = re.compile(r"Rotational constants in cm-1:\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)\s*(-?\d+\.\d+)")
            for line in lines:
                match = rot_const_re.search(line)
                if match:
                    try:
                        extracted_props['rotational_constants'] = np.array([float(match.group(1)),
                                                                             float(match.group(2)),
                                                                             float(match.group(3))])
                        break
                    except ValueError:
                        pass
        # --- END Conditional ROTATIONAL CONSTANTS EXTRACTION ---

        # Calculate Radius of Gyration ---
        if extracted_props['final_geometry_atomnos'] is not None and extracted_props['final_geometry_coords'] is not None:
            if len(extracted_props['final_geometry_atomnos']) > 0 and extracted_props['final_geometry_coords'].shape[0] > 0:
                try:
                    radius_gyr = calculate_radius_of_gyration(extracted_props['final_geometry_atomnos'], extracted_props['final_geometry_coords'])
                    extracted_props['radius_of_gyration'] = radius_gyr
                except Exception as e:
                    print(f"  ERROR: Problem calculating Radius of Gyration for {os.path.basename(logfile_path)}: {e}")
            else:
                print(f"  WARNING: Skipping Radius of Gyration for {os.path.basename(logfile_path)} due to empty atomnos or coords.")

        # Extract Vibrational Frequencies using cclib ---
        # The core check for frequency calculation presence will be here.
        # cclib.vibfreqs will be None or empty if no frequency calculation was found.
        if hasattr(data, "vibfreqs") and data.vibfreqs is not None and len(data.vibfreqs) > 0:
            extracted_props['_has_freq_calc'] = True # Set flag if freq data is present
            try:
                if len(data.vibfreqs) > 0 and isinstance(data.vibfreqs[0], (int, float, np.number)):
                    imag_freqs = [freq for freq in data.vibfreqs if freq < 0]
                    real_freqs = [freq for freq in data.vibfreqs if freq > 0]
                    extracted_props['num_imaginary_freqs'] = len(imag_freqs)
                    if real_freqs:
                        extracted_props['first_vib_freq'] = min(real_freqs)
                        extracted_props['last_vib_freq'] = max(real_freqs)
                else:
                    print(f"  WARNING: Vibrational frequencies found but not valid numeric data for {os.path.basename(logfile_path)}")
            except Exception as e:
                print(f"  ERROR: Problem extracting vibrational frequencies for {os.path.basename(logfile_path)}: {e}")
        else:
            # Explicitly set flag if no freq calc detected
            extracted_props['_has_freq_calc'] = False
            # Ensure these are None if no frequencies are found, which they should be by default
            extracted_props['num_imaginary_freqs'] = None 
            extracted_props['first_vib_freq'] = None
            extracted_props['last_vib_freq'] = None


        # Detect Hydrogen Bonds
        if extracted_props['final_geometry_atomnos'] is not None and extracted_props['final_geometry_coords'] is not None:
            if len(extracted_props['final_geometry_atomnos']) > 0 and extracted_props['final_geometry_coords'].shape[0] > 0:
                try:
                    # Call the updated detect_hydrogen_bonds function
                    hbond_results = detect_hydrogen_bonds(extracted_props['final_geometry_atomnos'], extracted_props['final_geometry_coords'])
                    
                    extracted_props.update(hbond_results) # Update all hbond related keys
                except Exception as e:
                    print(f"  ERROR: Problem detecting hydrogen bonds for {os.path.basename(logfile_path)}: {e}")
            else:
                print(f"  WARNING: Skipping Hydrogen Bond Analysis for {os.path.basename(logfile_path)} due to empty atomnos or coords.")

        return extracted_props

    except Exception as e:
        print(f"  GENERIC_ERROR: Failed to extract properties from {os.path.basename(logfile_path)} (after cclib parse): {e}")
        return None

def calculate_deviation_percentage(values):
    """Calculates the percentage deviation (max-min / abs(mean)) for a list of numerical values."""
    if not values or len(values) < 2:
        return 0.0 # Or None, depending on desired behavior for single/no values
    
    # Filter out None values before calculating min/max
    numeric_values = [v for v in values if v is not None]
    if not numeric_values:
        return 0.0

    min_val = min(numeric_values)
    max_val = max(numeric_values)

    if min_val == 0.0 and max_val == 0.0:
        return 0.0 # All zeros, no deviation

    if max_val == min_val:
        return 0.0

    mean_val = np.mean(numeric_values)
    if mean_val == 0.0: # Avoid division by zero if mean is zero
        return (max_val - min_val) / abs(mean_val) * 100.0 if max_val != min_val else 0.0
    
    return ((max_val - min_val) / abs(mean_val)) * 100.0

# --- START: RMSD functions provided by user ---
def calculate_rmsd(atomnos1, coords1, atomnos2, coords2):
    """
    Calculates the RMSD between two sets of coordinates using Kabsch alignment,
    considering only heavy (non-hydrogen) atoms.

    Args:
        atomnos1 (list or np.array): Atomic numbers for structure 1.
        coords1 (np.array): Nx3 array of atomic coordinates for structure 1.
        atomnos2 (list or np.array): Atomic numbers for structure 2.
        coords2 (np.array): Nx3 array of atomic coordinates for structure 2.

    Returns:
        float: The calculated RMSD value, or None if an error occurs.
    """
    # Filter out hydrogen atoms (atomic number 1)
    heavy_indices1 = [i for i, z in enumerate(atomnos1) if z != 1]
    heavy_coords1 = coords1[heavy_indices1]
    # heavy_atomnos1 = [atomnos1[i] for i in heavy_indices1] # Not strictly needed for RMSD calculation itself

    heavy_indices2 = [i for i, z in enumerate(atomnos2) if z != 1]
    heavy_coords2 = coords2[heavy_indices2]
    # heavy_atomnos2 = [atomnos2[i] for i in heavy_indices2] # Not strictly needed for RMSD calculation itself

    if len(heavy_indices1) == 0 or len(heavy_indices2) == 0:
        return None

    if len(heavy_indices1) != len(heavy_indices2):
        # This check is crucial and implies mismatched heavy atom counts
        # It's better to explicitly check if lengths are different, not just shapes, as shape might match if coords is empty.
        return None

    # Ensure coordinates are numpy arrays and float64
    coords1_filtered = np.asarray(heavy_coords1, dtype=np.float64)
    coords2_filtered = np.asarray(heavy_coords2, dtype=np.float64)

    # Check for empty arrays after filtering, though len() check above should catch
    if coords1_filtered.shape[0] == 0 or coords2_filtered.shape[0] == 0:
        return None

    try:
        # Step 1: Center the coordinates (move to origin)
        center1 = np.mean(coords1_filtered, axis=0)
        centered_coords1 = coords1_filtered - center1

        center2 = np.mean(coords2_filtered, axis=0)
        centered_coords2 = coords2_filtered - center2

        # Step 2: Perform Kabsch alignment to find the optimal rotation
        # R.align_vectors(a, b) finds rotation and RMSD to transform a to align with b.
        # So here, we want to align centered_coords2 (source) to centered_coords1 (target).
        rotation, rmsd_value = R.align_vectors(centered_coords2, centered_coords1)

        # The 'rmsd_value' returned by align_vectors IS the minimized RMSD.
        # You don't need to re-apply the rotation and calculate it again.
        # If you did, it should yield the same 'rmsd_value'.
        # The VMD results strongly suggest that this 'rmsd_value' is the one you want.
        
        return rmsd_value

    except Exception as e:
        print(f"  ERROR during heavy atom RMSD calculation: {e}")
        return None

def post_process_clusters_with_rmsd(initial_clusters, rmsd_validation_threshold):
    """
    Refines property-based clusters by performing an RMSD validation within each.
    Configurations failing the RMSD check are extracted into new single-configuration clusters.

    Args:
        initial_clusters (list): A list of lists/tuples, where each inner list/tuple
                                 represents a property-based cluster and contains
                                 dictionaries of extracted data for each configuration.
        rmsd_validation_threshold (float): The maximum allowed RMSD value (in Angstroms) for
                                 configurations to remain in the same cluster.

    Returns:
        tuple: A tuple containing:
               - list: A new list of validated multi-configuration clusters (members passed RMSD to rep).
               - list: A list of individual outlier configurations (data dicts) that were split out.
    """
    validated_main_clusters = []
    individual_outliers = []

    print(f"  Initiating first pass RMSD validation with threshold: {rmsd_validation_threshold:.3f} Å...")

    for cluster_idx, current_property_cluster in enumerate(initial_clusters):
        if not current_property_cluster:
            continue

        if len(current_property_cluster) == 1:
            # Single member clusters are passed directly to validated_main_clusters
            # Mark them as coming from the first pass
            current_property_cluster[0]['_rmsd_pass_origin'] = 'first_pass_validated'
            validated_main_clusters.append(current_property_cluster)
            continue

        print(f"    Validating initial property cluster {current_property_cluster[0].get('_parent_global_cluster_id', 'N/A')} with {len(current_property_cluster)} configurations...")

        # Select the lowest energy configuration as the representative for this property cluster
        # Fallback to filename for deterministic choice if Gibbs free energy is None
        representative_conf = min(current_property_cluster,
                                  key=lambda x: (x.get('gibbs_free_energy') if x.get('gibbs_free_energy') is not None else float('inf'), x['filename']))

        current_validated_sub_cluster = [representative_conf] # Start new validated cluster with representative
        processed_members_filenames = {representative_conf['filename']}

        # Mark the representative as coming from the first pass
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
                # Store the _parent_global_cluster_id of the representative, which is the original cluster
                conf_member['_parent_global_cluster_id'] = representative_conf['_parent_global_cluster_id']
                conf_member['_rmsd_pass_origin'] = 'second_pass_formed' # Mark as needing second pass treatment
                individual_outliers.append(conf_member) # Collect this as an outlier
                processed_members_filenames.add(conf_member['filename'])
                continue
            
            rmsd_val = calculate_rmsd(
                atomnos_rep, coords_rep,
                atomnos_member, coords_member
            )

            if rmsd_val is not None and rmsd_val <= rmsd_validation_threshold:
                current_validated_sub_cluster.append(conf_member)
                conf_member['_rmsd_pass_origin'] = 'first_pass_validated' # Mark as remaining in first pass
                processed_members_filenames.add(conf_member['filename'])
            else:
                # This configuration is an outlier from the first pass:
                print(f"    {conf_member['filename']} (RMSD={rmsd_val:.3f} Å) is an outlier from {representative_conf['filename']} (Threshold={rmsd_validation_threshold:.3f} Å).")
                # Store the _parent_global_cluster_id of the representative, which is the original cluster
                conf_member['_parent_global_cluster_id'] = representative_conf['_parent_global_cluster_id']
                conf_member['_rmsd_pass_origin'] = 'second_pass_formed' # Mark as needing second pass treatment
                individual_outliers.append(conf_member) # Collect this as an outlier
                processed_members_filenames.add(conf_member['filename'])

        if current_validated_sub_cluster:
             validated_main_clusters.append(current_validated_sub_cluster)

    return validated_main_clusters, individual_outliers

def perform_second_rmsd_clustering(cluster_members_to_refine, rmsd_threshold):
    """
    Performs a second RMSD-based clustering on a group of configurations (typically outliers
    from a previous RMSD pass, or validated clusters that need further refinement).
    
    Args:
        cluster_members_to_refine (list): A list of configuration data dictionaries to re-cluster.
        rmsd_threshold (float): The RMSD threshold for this second clustering step.

    Returns:
        list: A list of new sub-clusters (each a list of data dictionaries).
    """
    if len(cluster_members_to_refine) <= 1:
        # For singletons, ensure the second RMSD context is set even if trivial
        for m in cluster_members_to_refine:
            # If it's a singleton here, its _parent_global_cluster_id should already be set from first pass outlier detection
            m['_second_rmsd_sub_cluster_id'] = m.get('_initial_cluster_label') # Can still inherit initial label
            m['_second_rmsd_context_listing'] = [{'filename': m['filename'], 'rmsd_to_rep': 0.0}]
            m['_second_rmsd_rep_filename'] = m['filename'] # It is its own representative
            m['_rmsd_pass_origin'] = 'second_pass_formed' # Explicitly mark as second pass
        return [[m] for m in cluster_members_to_refine]

    # Calculate all-pairs RMSD distance matrix for members
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
                rmsd = float('inf') # A large value to ensure they don't cluster if RMSD can't be computed
            rmsd_matrix[i, j] = rmsd_matrix[j, i] = rmsd

    # Convert square distance matrix to condensed form for linkage
    condensed_distances = []
    for i in range(num_members):
        for j in range(i + 1, num_members):
            condensed_distances.append(rmsd_matrix[i, j])

    if not condensed_distances:
        # Fallback for no distances (e.g., all infinite, or only one valid pair)
        # Treat each as a singleton, and ensure _second_rmsd_context_listing is set
        for m in cluster_members_to_refine:
            # If it's a singleton here, its _parent_global_cluster_id should already be set from first pass outlier detection
            m['_second_rmsd_sub_cluster_id'] = m.get('_initial_cluster_label') # Or generate new unique ID
            m['_second_rmsd_context_listing'] = [{'filename': m['filename'], 'rmsd_to_rep': 0.0}]
            m['_second_rmsd_rep_filename'] = m['filename']
            m['_rmsd_pass_origin'] = 'second_pass_formed' # Explicitly mark as second pass
        return [[m] for m in cluster_members_to_refine]

    linkage_matrix = linkage(condensed_distances, method='average', metric='euclidean')

    # Perform clustering
    second_cluster_labels = fcluster(linkage_matrix, t=rmsd_threshold, criterion='distance')

    # Organize members into new sub-clusters
    second_level_clusters_data = {}
    for i, label in enumerate(second_cluster_labels):
        cluster_members_to_refine[i]['_second_rmsd_sub_cluster_id'] = label # Set ID
        cluster_members_to_refine[i]['_rmsd_pass_origin'] = 'second_pass_formed' # Mark as second pass
        second_level_clusters_data.setdefault(label, []).append(cluster_members_to_refine[i])

    final_sub_clusters = []
    for label, sub_cluster_members in second_level_clusters_data.items():
        if not sub_cluster_members: continue

        # Select lowest energy member as representative for this second-level cluster
        # Fallback to filename for deterministic choice if Gibbs free energy is None
        sub_cluster_rep = min(sub_cluster_members,
                              key=lambda x: (x.get('gibbs_free_energy') if x.get('gibbs_free_energy') is not None else float('inf'), x['filename']))
        
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
            # Fallback if representative has no geometry
            for member_conf in sub_cluster_members:
                sub_cluster_rmsd_listing.append({'filename': member_conf['filename'], 'rmsd_to_rep': None})

        # Store second RMSD context on each member of this new sub-cluster
        for member_conf in sub_cluster_members:
            member_conf['_second_rmsd_context_listing'] = sub_cluster_rmsd_listing
            member_conf['_second_rmsd_rep_filename'] = sub_cluster_rep['filename']
        
        final_sub_clusters.append(sub_cluster_members)
    
    return final_sub_clusters
# --- END: RMSD functions provided by user ---


def write_cluster_dat_file(dat_file_prefix, cluster_members_data, output_base_dir, rmsd_threshold_value=None, 
                           hbond_count_for_original_cluster=None, weights=None):
    """
    Writes a combined .dat file for all members of a cluster, including comparison and RMSD context sections.
    The `dat_file_prefix` is the full desired name for the .dat file (e.g., 'cluster_12' or 'cluster_12_5').
    `hbond_count_for_original_cluster` is the number of hydrogen bonds for the initial group this cluster came from.
    `weights` is a dictionary mapping feature names (user-friendly) to their weights, used for conditional printing.
    """
    if weights is None:
        weights = {} # Ensure weights is a dict if not provided

    num_configurations = len(cluster_members_data)
    
    dat_output_dir = os.path.join(output_base_dir, "extracted_data")
    os.makedirs(dat_output_dir, exist_ok=True)

    output_filename = os.path.join(dat_output_dir, f"{dat_file_prefix}.dat")

    # Helper to check if a feature should be printed based on its weight
    def should_print_feature(feature_key_in_data, user_friendly_name, weights_dict):
        # Map internal data key to user-friendly name using FEATURE_MAPPING
        # This is a reverse lookup, so iterate through FEATURE_MAPPING
        mapped_user_friendly_name = None
        for u_name, i_key in FEATURE_MAPPING.items():
            # Special handling for rotational constants which are stored as an array
            if user_friendly_name.startswith("rotational_constants") and i_key.startswith("rotational_constants"):
                if user_friendly_name.endswith("_A") and i_key.endswith("_0"):
                    mapped_user_friendly_name = u_name
                    break
                elif user_friendly_name.endswith("_B") and i_key.endswith("_1"):
                    mapped_user_friendly_name = u_name
                    break
                elif user_friendly_name.endswith("_C") and i_key.endswith("_2"):
                    mapped_user_friendly_name = u_name
                    break
            elif i_key == feature_key_in_data:
                mapped_user_friendly_name = u_name
                break
        
        # If not found in mapping, or if it's a non-clustering property (like num_hydrogen_bonds),
        # assume it should be printed unless explicitly weighted to 0.0 by its user-friendly name.
        if mapped_user_friendly_name is None:
            # For properties not directly used in clustering features_for_scaling (like num_hydrogen_bonds, hbond_details)
            # or properties that are just for display (like method, functional, etc.),
            # they should always be printed unless explicitly zero-weighted by their display name.
            if user_friendly_name in weights_dict and weights_dict[user_friendly_name] == 0.0:
                return False
            return True # Default to printing if not a mapped feature or not zero-weighted
        
        # For mapped features, check their weight
        return weights_dict.get(mapped_user_friendly_name, 1.0) != 0.0


    with open(output_filename, 'w', newline='\n') as f:
        # 1. Initial Header Separator
        f.write("=" * 90 + "\n\n")

        rmsd_context_printed = False

        # 2. Initial Clustering RMSD Context Section
        if rmsd_threshold_value is not None and cluster_members_data and '_first_rmsd_context_listing' in cluster_members_data[0] and cluster_members_data[0]['_first_rmsd_context_listing'] is not None:
            initial_rmsd_context = cluster_members_data[0]['_first_rmsd_context_listing']
            f.write("Initial Clustering RMSD Context (Before Refinement):\n")
            f.write("Configurations from the original property cluster:\n")
            for item in initial_rmsd_context:
                rmsd_val_str = f"({item['rmsd_to_rep']:.3f} Å)" if item['rmsd_to_rep'] is not None else "(N/A)"
                f.write(f"    - {item['filename']} {rmsd_val_str}\n")
            f.write("\n")
            
            original_prop_cluster_label = cluster_members_data[0].get('_initial_cluster_label', 'N/A')
            parent_global_cluster_id_for_display = cluster_members_data[0].get('_parent_global_cluster_id', 'N/A')

            hbond_context = f" (H-bonds {hbond_count_for_original_cluster})" if hbond_count_for_original_cluster is not None else ""
            f.write(f"RMSD values are relative to the lowest energy representative of this initial property group")
            f.write("\n\n")
            rmsd_context_printed = True

        # 3. Second RMSD Clustering Context Section
        if rmsd_threshold_value is not None and cluster_members_data and \
           cluster_members_data[0].get('_rmsd_pass_origin') == 'second_pass_formed' and \
           cluster_members_data[0].get('_second_rmsd_context_listing') is not None:
            
            second_rmsd_context = cluster_members_data[0]['_second_rmsd_context_listing']
            second_rmsd_rep_filename = cluster_members_data[0].get('_second_rmsd_rep_filename', 'N/A')
            
            f.write("Second RMSD Clustering Context:\n")
            
            parent_global_cluster_id_for_display = cluster_members_data[0].get('_parent_global_cluster_id', 'N/A')

            if num_configurations == 1:
                f.write(f"This single configuration was either an outlier or remained a singleton after a second RMSD clustering step (threshold: {rmsd_threshold_value:.3f} Å).\n")
                f.write(f"  It originated from original Cluster {parent_global_cluster_id_for_display}.\n")
                
                self_rmsd_info = next((item for item in second_rmsd_context if item['filename'] == cluster_members_data[0]['filename']), None)
                if self_rmsd_info and self_rmsd_info['rmsd_to_rep'] is not None:
                    f.write(f"RMSD to its second-level cluster's representative ({second_rmsd_rep_filename}): {self_rmsd_info['rmsd_to_rep']:.3f} Å\n")
                else:
                    f.write(f"RMSD to its second-level cluster's representative ({second_rmsd_rep_filename}): N/A\n")

            else:
                f.write(f"This cluster was formed by a second RMSD clustering step (threshold: {rmsd_threshold_value:.3f} Å).\n")
                f.write(f"  It originated from original Cluster {parent_global_cluster_id_for_display}.\n")
                f.write(f"Representative for this second-level cluster: {second_rmsd_rep_filename}\n")
                f.write("RMSD values relative to this second-level cluster's representative:\n")
                
                for item in second_rmsd_context:
                    rmsd_val_str = f"({item['rmsd_to_rep']:.3f} Å)" if item['rmsd_to_rep'] is not None else "(N/A)"
                    f.write(f"    - {item['filename']} {rmsd_val_str}\n")
            f.write("\n")
            rmsd_context_printed = True

        # Separator after RMSD context (if any was printed)
        if rmsd_context_printed:
            f.write("=" * 90 + "\n\n")

        # 4. Cluster Summary Header (ALWAYS print this)
        f.write(f"Cluster (represented by: {dat_file_prefix}) ({num_configurations} configurations)\n\n") 
        for mol_data in cluster_members_data:
            f.write(f"    - {mol_data['filename']}\n")
        f.write("\n")

        # 5. Deviation Analysis (ONLY for clusters with >1 configuration)
        if num_configurations > 1:
            f.write("\nDeviation Analysis (Max-Min / |Mean|):\n")
            if should_print_feature('final_electronic_energy', 'electronic_energy', weights) and all(d['final_electronic_energy'] is not None for d in cluster_members_data):
                all_electronic_energies = [d['final_electronic_energy'] for d in cluster_members_data if d['final_electronic_energy'] is not None]
                if all_electronic_energies: f.write(f"  Electronic Energy (Hartree) %Dev: {calculate_deviation_percentage(all_electronic_energies):.2f}%\n")
            
            if should_print_feature('gibbs_free_energy', 'gibbs_free_energy', weights) and all(d['gibbs_free_energy'] is not None for d in cluster_members_data):
                all_gibbs_energies = [d['gibbs_free_energy'] for d in cluster_members_data if d['gibbs_free_energy'] is not None]
                if all_gibbs_energies: f.write(f"  Gibbs Free Energy (Hartree) %Dev: {calculate_deviation_percentage(all_gibbs_energies):.2f}%\n")
            
            if should_print_feature('entropy', 'entropy', weights) and all(d['entropy'] is not None for d in cluster_members_data):
                all_entropy = [d['entropy'] for d in cluster_members_data if d['entropy'] is not None]
                if all_entropy: f.write(f"  Entropy (J/(mol·K) or a.u.): {calculate_deviation_percentage(all_entropy):.2f}%\n")
            
            if should_print_feature('homo_energy', 'homo_energy', weights) and all(d['homo_energy'] is not None for d in cluster_members_data):
                all_homo_energies = [d['homo_energy'] for d in cluster_members_data if d['homo_energy'] is not None]
                if all_homo_energies: f.write(f"  HOMO Energy (eV) %Dev: {calculate_deviation_percentage(all_homo_energies):.2f}%\n")
            
            if should_print_feature('lumo_energy', 'lumo_energy', weights) and all(d['lumo_energy'] is not None for d in cluster_members_data):
                all_lumo_energies = [d['lumo_energy'] for d in cluster_members_data if d['lumo_energy'] is not None]
                if all_lumo_energies: f.write(f"  LUMO Energy (eV) %Dev: {calculate_deviation_percentage(all_lumo_energies):.2f}%\n")
            
            if should_print_feature('homo_lumo_gap', 'homo_lumo_gap', weights) and all(d['homo_lumo_gap'] is not None for d in cluster_members_data):
                all_homo_lumo_gaps = [d['homo_lumo_gap'] for d in cluster_members_data if d['homo_lumo_gap'] is not None]
                if all_homo_lumo_gaps: f.write(f"  HOMO-LUMO Gap (eV) %Dev: {calculate_deviation_percentage(all_homo_lumo_gaps):.2f}%\n")
            
            if should_print_feature('dipole_moment', 'dipole_moment', weights) and all(d['dipole_moment'] is not None for d in cluster_members_data):
                all_dipole_moments = [d['dipole_moment'] for d in cluster_members_data if d['dipole_moment'] is not None]
                if all_dipole_moments: f.write(f"  Dipole Moment (Debye) %Dev: {calculate_deviation_percentage(all_dipole_moments):.2f}%\n")
            
            if should_print_feature('radius_of_gyration', 'radius_of_gyration', weights) and all(d['radius_of_gyration'] is not None for d in cluster_members_data):
                all_rg = [d['radius_of_gyration'] for d in cluster_members_data if d['radius_of_gyration'] is not None]
                if all_rg: f.write(f"  Radius of Gyration (Å) %Dev: {calculate_deviation_percentage(all_rg):.2f}%\n")
            
            # Rotational constants need special handling because they are an array
            all_rot_const_A = [d['rotational_constants'][0] for d in cluster_members_data if d['rotational_constants'] is not None and isinstance(d['rotational_constants'], np.ndarray) and len(d['rotational_constants']) == 3]
            all_rot_const_B = [d['rotational_constants'][1] for d in cluster_members_data if d['rotational_constants'] is not None and isinstance(d['rotational_constants'], np.ndarray) and len(d['rotational_constants']) == 3]
            all_rot_const_C = [d['rotational_constants'][2] for d in cluster_members_data if d['rotational_constants'] is not None and isinstance(d['rotational_constants'], np.ndarray) and len(d['rotational_constants']) == 3]

            if should_print_feature('rotational_constants_0', 'rotational_constants_A', weights) and all_rot_const_A:
                f.write(f"  Rotational Constant A (GHz) %Dev: {calculate_deviation_percentage(all_rot_const_A):.2f}%\n")
            if should_print_feature('rotational_constants_1', 'rotational_constants_B', weights) and all_rot_const_B:
                f.write(f"  Rotational Constant B (GHz) %Dev: {calculate_deviation_percentage(all_rot_const_B):.2f}%\n")
            if should_print_feature('rotational_constants_2', 'rotational_constants_C', weights) and all_rot_const_C:
                f.write(f"  Rotational Constant C (GHz) %Dev: {calculate_deviation_percentage(all_rot_const_C):.2f}%\n")
            
            if should_print_feature('first_vib_freq', 'first_vib_freq', weights) and all(d['first_vib_freq'] is not None for d in cluster_members_data):
                all_first_vib_freqs = [d['first_vib_freq'] for d in cluster_members_data if d['first_vib_freq'] is not None]
                if all_first_vib_freqs: f.write(f"  First Vibrational Frequency (cm^-1) %Dev: {calculate_deviation_percentage(all_first_vib_freqs):.2f}%\n")
            
            if should_print_feature('last_vib_freq', 'last_vib_freq', weights) and all(d['last_vib_freq'] is not None for d in cluster_members_data):
                all_last_vib_freqs = [d['last_vib_freq'] for d in cluster_members_data if d['last_vib_freq'] is not None]
                if all_last_vib_freqs: f.write(f"  Last Vibrational Frequency (cm^-1) %Dev: {calculate_deviation_percentage(all_last_vib_freqs):.2f}%\n")
            
            # num_hydrogen_bonds is not a clustering feature, but a property. It should appear if not zero-weighted.
            if should_print_feature('num_hydrogen_bonds', 'num_hydrogen_bonds', weights) and all(d['num_hydrogen_bonds'] is not None for d in cluster_members_data):
                all_num_hbonds = [d['num_hydrogen_bonds'] for d in cluster_members_data if d['num_hydrogen_bonds'] is not None]
                if all_num_hbonds: f.write(f"  Number of Hydrogen Bonds %Dev: {calculate_deviation_percentage(all_num_hbonds):.2f}%\n")
            
            f.write("\n")
            
        # Separator before the detailed descriptor comparison section
        f.write("=" * 90 + "\n\n")

        # 6. Detailed Descriptors Comparison for each structure
        # This section always prints the descriptors for each member of the cluster.
        f.write("Electronic configuration descriptors:\n")
        for mol_data in cluster_members_data:
            f.write(f"    {mol_data['filename']}:\n")
            if mol_data['final_electronic_energy'] is not None:
                f.write(f"        Final Electronic Energy (Hartree): {mol_data['final_electronic_energy']:.6f}\n")
            if mol_data['gibbs_free_energy'] is not None:
                f.write(f"        Gibbs Free Energy (Hartree): {mol_data['gibbs_free_energy']:.6f}\n")
            if mol_data['homo_energy'] is not None:
                f.write(f"        HOMO Energy (eV): {mol_data['homo_energy']:.6f}\n")
            if mol_data['lumo_energy'] is not None:
                f.write(f"        LUMO Energy (eV): {mol_data['lumo_energy']:.6f}\n")
            if mol_data['homo_lumo_gap'] is not None:
                f.write(f"        HOMO-LUMO Gap (eV): {mol_data['homo_lumo_gap']:.6f}\n")
        f.write("\n")

        f.write("Molecular configuration descriptors:\n")
        for mol_data in cluster_members_data:
            f.write(f"    {mol_data['filename']}:\n")
            if mol_data['dipole_moment'] is not None:
                f.write(f"        Dipole Moment (Debye): {mol_data['dipole_moment']:.6f}\n")
            rc = mol_data['rotational_constants']
            if rc is not None and isinstance(rc, np.ndarray) and rc.ndim == 1 and len(rc) == 3:
                f.write(f"        Rotational Constants (GHz): {rc[0]:.6f}, {rc[1]:.6f}, {rc[2]:.6f}\n")
            if mol_data['radius_of_gyration'] is not None:
                f.write(f"        Radius of Gyration (Å): {mol_data['radius_of_gyration']:.6f}\n")
        f.write("\n")

        f.write("Vibrational frequency summary:\n")
        for mol_data in cluster_members_data:
            f.write(f"    {mol_data['filename']}:\n")
            if mol_data.get('_has_freq_calc', False):
                f.write(f"        Number of imaginary frequencies: {mol_data.get('num_imaginary_freqs', 'N/A')}\n")
                if mol_data['first_vib_freq'] is not None:
                    f.write(f"        First Vibrational Frequency (cm^-1): {mol_data['first_vib_freq']:.2f}\n")
                if mol_data['last_vib_freq'] is not None:
                    f.write(f"        Last Vibrational Frequency (cm^-1): {mol_data['last_vib_freq']:.2f}\n")
            else:
                f.write("        Frequency calculation not performed.\n")
        f.write("\n")

        f.write("Hydrogen bond analysis:\n")
        HB_min_angle_actual_for_display = 30.0 # Define explicitly for display
        f.write(f"Criterion: H...A distance between 1.2 Å and 2.7 Å, with H covalently bonded to a donor (O, N, F).\n")
        f.write(f"  (For counting, D-H...A angle must be >= {HB_min_angle_actual_for_display:.1f}°)\n")
        for mol_data in cluster_members_data:
            f.write(f"    {mol_data['filename']}:\n")
            num_counted_hb = mol_data.get('num_hydrogen_bonds', 0)
            total_potential_hb = len(mol_data.get('hbond_details', []))
            f.write(f"        Number of hydrogen bonds counted (angle >= 30.0°): {num_counted_hb} out of {total_potential_hb} potential bonds.\n")
            
            if mol_data.get('hbond_details'):
                for hbond in mol_data['hbond_details']:
                    angle_note = ""
                    if hbond['D-H...A_angle'] < HB_min_angle_actual_for_display:
                        angle_note = f" (Angle < {HB_min_angle_actual_for_display:.1f}° - Not counted as HB)"
                    f.write(f"            Hydrogen bond: {hbond['donor_atom_label']}-{hbond['hydrogen_atom_label']}...{hbond['acceptor_atom_label']} (Dist: {hbond['H...A_distance']:.3f} Å, D-H: {hbond['D-H_covalent_distance']:.3f} Å, Angle: {hbond['D-H...A_angle']:.2f}°){angle_note}\n")
            else:
                f.write("        No hydrogen bonds detected based on the criterion.\n")
        f.write("\n")

        # Separator before the individual structure details
        f.write("=" * 90 + "\n\n")

        # 7. Individual Structure Details
        # This section now only includes general file info and geometry,
        # avoiding the repetition of descriptor summaries.
        for i, mol_data in enumerate(cluster_members_data):
            if i > 0: # Add shorter separator only before subsequent structures
                f.write("=" * 50 + "\n\n") 

            f.write(f"Processed file: {mol_data['filename']}\n")
            f.write(f"Method: {mol_data.get('method', 'N/A')}\n")
            f.write(f"Functional: {mol_data.get('functional', 'N/A')}\n")
            f.write(f"Basis Set: {mol_data.get('basis_set', 'N/A')}\n")
            f.write(f"Charge: {mol_data.get('charge', 'N/A')}\n")
            f.write(f"Multiplicity: {mol_data.get('multiplicity', 'N/A')}\n")
            f.write(f"Number of atoms: {mol_data.get('num_atoms', 'N/A')}\n")

            # Final Geometry
            if mol_data.get('final_geometry_atomnos') is not None and mol_data.get('final_geometry_coords') is not None:
                f.write("Final Geometry:\n")
                atomnos = mol_data['final_geometry_atomnos']
                atomcoords = mol_data['final_geometry_coords']
                for j in range(len(atomnos)):
                    symbol = atomic_number_to_symbol(atomnos[j])
                    f.write(f"{symbol:<2} {atomcoords[j][0]:10.6f} {atomcoords[j][1]:10.6f} {atomcoords[j][2]:10.6f}\n")
            else:
                f.write("Final Geometry: N/A\n")
            f.write("\n")

            # Removed the repeated descriptor blocks here (Electronic, Molecular, Vibrational, Hydrogen bond)
            # as they are already printed in Section 6 for each member of the cluster.

    print(f"Wrote combined data for Cluster '{dat_file_prefix}' to '{os.path.basename(output_filename)}'")

def write_xyz_file(mol_data, filename):
    """
    Writes atomic coordinates to an XYZ file, including Gibbs Free Energy in the comment line.
    """
    atomnos = mol_data.get('final_geometry_atomnos')
    atomcoords = mol_data.get('final_geometry_coords')
    gibbs_free_energy = mol_data.get('gibbs_free_energy')
    
    if atomnos is None or atomcoords is None or len(atomnos) == 0:
        print(f"  WARNING: Cannot write XYZ for {os.path.basename(filename)}: Missing geometry data.")
        return

    base_name = os.path.splitext(os.path.basename(mol_data['filename']))[0]
    
    gibbs_str = f"{gibbs_free_energy:.6f} hartree" if gibbs_free_energy is not None else "N/A"
    comment_line = f"{base_name} (G = {gibbs_str})" # Updated comment format

    symbols = [atomic_number_to_symbol(n) for n in atomnos]

    with open(filename, 'w', newline='\n') as f:
        f.write(f"{len(atomnos)}\n")
        f.write(f"{comment_line}\n")
        for i in range(len(atomnos)):
            f.write(f"{symbols[i]:<2} {atomcoords[i][0]:10.6f} {atomcoords[i][1]:10.6f} {atomcoords[i][2]:10.6f}\n")

def create_unique_motifs_folder(all_clusters_data, output_base_dir, openbabel_alias="obabel"):
    """
    Creates a motifs folder containing the lowest energy representative structure from each cluster.
    Also creates a combined XYZ file with all representatives and attempts to convert to MOL format.
    
    Args:
        all_clusters_data (list): List of clusters, where each cluster is a list of molecule data dictionaries
        output_base_dir (str): Base output directory where motifs folder will be created
        openbabel_alias (str): Alias for OpenBabel command (default: "obabel")
    """
    if not all_clusters_data:
        print("  No clusters found. Skipping motifs creation.")
        return
    
    # Create motifs directory with number of motifs in the name
    num_motifs = len(all_clusters_data)
    motifs_dir = os.path.join(output_base_dir, f"motifs_{num_motifs:02d}")
    os.makedirs(motifs_dir, exist_ok=True)
    
    print(f"\nCreating motifs folder with {num_motifs} representative structures...")
    print(f"  Output directory: {motifs_dir}")
    
    representatives = []
    
    for cluster_idx, cluster_members in enumerate(all_clusters_data):
        if not cluster_members:
            continue
            
        # Find the lowest energy representative (same logic as used elsewhere)
        representative = min(cluster_members,
                           key=lambda x: (x.get('gibbs_free_energy') if x.get('gibbs_free_energy') is not None else float('inf'), x['filename']))
        
        representatives.append(representative)
        
        # Create individual XYZ file for this representative
        base_name = os.path.splitext(representative['filename'])[0]
        motif_filename = f"motif_{cluster_idx+1:02d}_{base_name}.xyz"
        motif_path = os.path.join(motifs_dir, motif_filename)
        
        write_xyz_file(representative, motif_path)
        
        gibbs_str = f"{representative['gibbs_free_energy']:.6f}" if representative['gibbs_free_energy'] is not None else "N/A"
        print(f"  Motif {cluster_idx+1:02d}: {base_name} (Gibbs Energy: {gibbs_str} Hartree)")
    
    # Create combined XYZ file with all representatives
    combined_xyz_path = os.path.join(motifs_dir, "all_motifs_combined.xyz")
    
    # Sort representatives by Gibbs free energy
    sorted_representatives = sorted(
        representatives,
        key=lambda x: (x.get('gibbs_free_energy') if x.get('gibbs_free_energy') is not None else float('inf'), x['filename'])
    )
    
    with open(combined_xyz_path, "w", newline='\n') as outfile:
        for rep_idx, rep_data in enumerate(sorted_representatives):
            atomnos = rep_data.get('final_geometry_atomnos')
            atomcoords = rep_data.get('final_geometry_coords')
            gibbs_free_energy = rep_data.get('gibbs_free_energy')
            
            if atomnos is None or atomcoords is None or len(atomnos) == 0:
                print(f"    WARNING: Skipping representative {rep_data['filename']} due to missing geometry data.")
                continue
            
            base_name = os.path.splitext(rep_data['filename'])[0]
            gibbs_str = f"{gibbs_free_energy:.6f} hartree" if gibbs_free_energy is not None else "N/A"
            comment_line = f"Motif_{rep_idx+1:02d}_{base_name} (G = {gibbs_str})"
            
            outfile.write(f"{len(atomnos)}\n")
            outfile.write(f"{comment_line}\n")
            for i in range(len(atomnos)):
                symbol = atomic_number_to_symbol(atomnos[i])
                outfile.write(f"{symbol:<2} {atomcoords[i][0]:10.6f} {atomcoords[i][1]:10.6f} {atomcoords[i][2]:10.6f}\n")
    
    print(f"  Created combined XYZ file: {os.path.basename(combined_xyz_path)}")
    
    # Attempt to create MOL file using OpenBabel
    mol_output_path = os.path.join(motifs_dir, "all_motifs_combined.mol")
    openbabel_full_path = shutil.which(openbabel_alias)
    
    if openbabel_full_path:
        try:
            # Use the correct OpenBabel syntax: obabel -i<format> input_file -o<format> -O output_file
            result = subprocess.run([openbabel_alias, "-ixyz", combined_xyz_path, "-omol", "-O", mol_output_path], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"  Successfully created MOL file: {os.path.basename(mol_output_path)}")
            else:
                print(f"  WARNING: OpenBabel conversion to MOL failed. Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"  WARNING: OpenBabel conversion to MOL timed out after 30 seconds.")
        except Exception as e:
            print(f"  WARNING: Error during OpenBabel conversion to MOL: {e}")
    else:
        print(f"  WARNING: OpenBabel ({openbabel_alias}) not found. Skipping MOL conversion.")
        print("  Please ensure OpenBabel is installed and added to your system's PATH.")
    
    print(f"\nMotifs creation complete. Output saved in: {motifs_dir}")


def combine_xyz_files(cluster_members_data, input_dir, output_base_name=None, openbabel_alias="obabel", prefix_template=None, motif_numbers=None):
    """
    Combines relevant .xyz data from cluster members into a single multi-frame .xyz file
    and attempts to convert the resulting file (or the single original .xyz) to a .mol file.
    Each frame in the combined XYZ will include Gibbs Free Energy in its comment line.
    The frames in the combined XYZ will be sorted by Gibbs free energy (lowest to highest).
    
    Args:
        prefix_template (str): Optional template for comment line prefix, e.g., "Motif_{:02d}_" for motifs
        motif_numbers (list): Optional list of motif numbers corresponding to each member (for unique motifs)
    """
    final_xyz_source_path = None # This will be the path to the XYZ file used for MOL conversion
    
    if not cluster_members_data:
        return

    if len(cluster_members_data) == 1:
        # For a single configuration, the XYZ file has already been written by write_xyz_file.
        # We just need to ensure the MOL conversion uses that file and its original name.
        single_mol_data = cluster_members_data[0]
        original_filename_base = os.path.splitext(single_mol_data['filename'])[0]
        final_xyz_source_path = os.path.join(input_dir, f"{original_filename_base}.xyz")
        # The output_base_name for MOL should be the original filename base
        final_output_mol_name_base = original_filename_base
        print(f"  Single configuration found in cluster. Using existing '{os.path.basename(final_xyz_source_path)}' for .mol conversion.")
        
    else:
        # For multiple configurations, create a new combined multi-frame XYZ file.
        if output_base_name is None:
            # Fallback for combined name if not provided (shouldn't happen with current calling logic)
            output_base_name = "combined_cluster"

        full_combined_xyz_path = os.path.join(input_dir, f"{output_base_name}.xyz")
        final_output_mol_name_base = output_base_name # Base name for the .mol file

        # Sort members by Gibbs free energy (lowest to highest), with filename as a tie-breaker
        # Also sort the motif_numbers list in the same order if provided
        if motif_numbers and len(motif_numbers) == len(cluster_members_data):
            # Create pairs of (data, motif_number) and sort by energy
            paired_data = list(zip(cluster_members_data, motif_numbers))
            sorted_pairs = sorted(
                paired_data,
                key=lambda x: (x[0].get('gibbs_free_energy') if x[0].get('gibbs_free_energy') is not None else float('inf'), x[0]['filename'])
            )
            sorted_members_data = [pair[0] for pair in sorted_pairs]
            sorted_motif_numbers = [pair[1] for pair in sorted_pairs]
        else:
            sorted_members_data = sorted(
                cluster_members_data,
                key=lambda x: (x.get('gibbs_free_energy') if x.get('gibbs_free_energy') is not None else float('inf'), x['filename'])
            )
            sorted_motif_numbers = None

        with open(full_combined_xyz_path, "w", newline='\n') as outfile:
            for frame_idx, mol_data in enumerate(sorted_members_data, 1): # Iterate over sorted data
                atomnos = mol_data.get('final_geometry_atomnos')
                atomcoords = mol_data.get('final_geometry_coords')
                gibbs_free_energy = mol_data.get('gibbs_free_energy')

                if atomnos is None or atomcoords is None or len(atomnos) == 0:
                    print(f"    WARNING: Skipping {mol_data['filename']} in combined XYZ due to missing geometry data.")
                    continue
                
                base_name_for_frame = os.path.splitext(mol_data['filename'])[0]
                gibbs_str = f"{gibbs_free_energy:.6f} hartree" if gibbs_free_energy is not None else "N/A"
                
                # Apply prefix template with actual motif number if provided
                if prefix_template and sorted_motif_numbers:
                    motif_num = sorted_motif_numbers[frame_idx - 1]  # frame_idx starts at 1
                    comment_line = f"{prefix_template.format(motif_num)}{base_name_for_frame} (G = {gibbs_str})"
                elif prefix_template:
                    comment_line = f"{prefix_template.format(frame_idx)}{base_name_for_frame} (G = {gibbs_str})"
                else:
                    comment_line = f"{base_name_for_frame} (G = {gibbs_str})"

                outfile.write(f"{len(atomnos)}\n")
                outfile.write(f"{comment_line}\n")
                for i in range(len(atomnos)):
                    symbol = atomic_number_to_symbol(atomnos[i])
                    outfile.write(f"{symbol:<2} {atomcoords[i][0]:10.6f} {atomcoords[i][1]:10.6f} {atomcoords[i][2]:10.6f}\n")
        
        print(f"  Successfully created combined multi-frame .xyz file: '{os.path.basename(full_combined_xyz_path)}'")
        final_xyz_source_path = full_combined_xyz_path

    # Section for Open Babel Integration (always attempts for MOL conversion)
    if final_xyz_source_path:
        mol_output_filename = f"{final_output_mol_name_base}.mol"
        full_mol_output_path = os.path.join(input_dir, mol_output_filename)

        openbabel_full_path = shutil.which(openbabel_alias)
        openbabel_installed = False

        if openbabel_full_path:
            openbabel_installed = True
        else:
            print(f"\n  Open Babel ({openbabel_alias}) command not found or not executable. Skipping .mol conversion.")
            print("  Please ensure Open Babel is installed and added to your system's PATH, or provide the correct alias.")
            print(f"  You can change the alias using the 'openbabel_alias' parameter in the function call, e.g., combine_xyz_files(..., openbabel_alias='obabel').")

        if openbabel_installed:
            try:
                conversion_command = [openbabel_full_path, "-i", "xyz", final_xyz_source_path, "-o", "mol", "-O", full_mol_output_path]
                subprocess.run(conversion_command, check=True, capture_output=True, text=True)

                if os.path.exists(full_mol_output_path):
                    print(f"  Successfully converted '{os.path.basename(final_xyz_source_path)}' to '{os.path.basename(full_mol_output_path)}' using Open Babel.")

            except subprocess.CalledProcessError as e:
                print(f"  Open Babel conversion failed for '{os.path.basename(final_xyz_source_path)}'.")
                print(f"  Error details: {e.stderr.strip()}")
            except Exception as e:
                print(f"  An unexpected error occurred during Open Babel conversion for '{final_xyz_source_path}': {e}")


# New: Feature mapping and weight parsing
FEATURE_MAPPING = {
    "electronic_energy": "final_electronic_energy",
    "gibbs_free_energy": "gibbs_free_energy",
    "entropy": "entropy",
    "homo_energy": "homo_energy",
    "lumo_energy": "lumo_energy",
    "homo_lumo_gap": "homo_lumo_gap",
    "dipole_moment": "dipole_moment",
    "radius_of_gyration": "radius_of_gyration",
    "rotational_constants_A": "rotational_constants_0", # Special handling for array elements
    "rotational_constants_B": "rotational_constants_1",
    "rotational_constants_C": "rotational_constants_2",
    "first_vib_freq": "first_vib_freq",
    "last_vib_freq": "last_vib_freq",
    "average_hbond_distance": "average_hbond_distance",
    "average_hbond_angle": "average_hbond_angle",
    "num_hydrogen_bonds": "num_hydrogen_bonds" # Although not a numerical feature for clustering, good to map
}

def parse_weights_argument(weight_str):
    """
    Parses the --weights argument string into a dictionary of feature_name: weight.
    Example: "(electronic_energy=0.1)(homo_lumo_gap=0.2)"
    """
    weights = {}
    if not weight_str:
        return weights

    # Regex to find (key=value) pairs
    matches = re.findall(r'\(([^=]+)=([\d.]+)\)', weight_str)
    for key, value in matches:
        try:
            weights[key.strip()] = float(value.strip())
        except ValueError:
            print(f"WARNING: Could not parse weight for '{key}={value}'. Skipping this weight.")
    return weights

def parse_abs_tolerance_argument(tolerance_str):
    """
    Parses the --abs-tolerance argument string into a dictionary of feature_name: tolerance.
    Example: "(electronic_energy=1e-5)(dipole_moment=1e-3)"
    """
    tolerances = {}
    if not tolerance_str:
        return tolerances
    
    matches = re.findall(r'\(([^=]+)=([\d\.eE-]+)\)', tolerance_str)
    for key, value in matches:
        try:
            tolerances[key.strip()] = float(value.strip())
        except ValueError:
            print(f"WARNING: Could not parse absolute tolerance for '{key}={value}'. Skipping this tolerance.")
    return tolerances


def create_motif_summary_excluding_hbonds(all_clusters_data, output_base_dir, threshold=1.0, min_std_threshold=1e-6, abs_tolerances=None):
    """
    Creates motif summary by re-clustering cluster representatives excluding hydrogen bond count.
    Groups structures that are identical except for the number of hydrogen bonds.
    Keeps H-bond quality features (distances, angles) but ignores H-bond quantity.
    Creates truly unique motifs in unique_motifs_## directory with combined XYZ/MOL files.
    
    Args:
        all_clusters_data (list): List of clusters from original clustering
        output_base_dir (str): Base output directory
        threshold (float): Clustering threshold for final motif grouping (default: 1.0)
        min_std_threshold (float): Minimum standard deviation threshold
        abs_tolerances (dict): Absolute tolerances for features
    """
    print(f"\n=== Creating Final Motif Summary (Excluding H-bond count) ===")
    
    # Get representatives from each cluster (lowest energy) and track original cluster numbers
    cluster_representatives = []
    cluster_original_indices = []  # Track which original cluster each representative came from
    for cluster_idx, cluster in enumerate(all_clusters_data):
        if cluster:
            representative = min(cluster,
                               key=lambda x: (x.get('gibbs_free_energy') if x.get('gibbs_free_energy') is not None else float('inf'), x['filename']))
            cluster_representatives.append(representative)
            cluster_original_indices.append(cluster_idx + 1)  # +1 for 1-based numbering
    
    if len(cluster_representatives) < 2:
        print("  Not enough cluster representatives for motif analysis. Skipping.")
        return
    
    print(f"  Re-clustering {len(cluster_representatives)} cluster representatives without H-bond count...")
    
    # Define features to exclude (only hydrogen bond count, not quality/geometry features)
    excluded_features = {
        'num_hydrogen_bonds'
    }
    
    # Define features to include for motif clustering
    motif_features = [
        'radius_of_gyration', 'dipole_moment', 'homo_lumo_gap',
        'first_vib_freq', 'last_vib_freq', 'average_hbond_distance', 
        'average_hbond_angle', 'min_hbond_distance', 'max_hbond_distance',
        'std_hbond_distance', 'min_hbond_angle', 'max_hbond_angle', 
        'std_hbond_angle'
    ]
    rotational_constant_subfeatures = [
        'rotational_constants_A', 'rotational_constants_B', 'rotational_constants_C'
    ]
    
    # Check which features are globally available
    globally_missing_features = []
    for feature in motif_features:
        if all(d.get(feature) is None for d in cluster_representatives):
            globally_missing_features.append(feature)
    
    # Check rotational constants
    is_rot_const_globally_missing = True
    for d in cluster_representatives:
        rot_consts = d.get('rotational_constants')
        if rot_consts is not None and isinstance(rot_consts, np.ndarray) and rot_consts.ndim == 1 and len(rot_consts) == 3:
            is_rot_const_globally_missing = False
            break
    
    if is_rot_const_globally_missing:
        globally_missing_features.extend(rotational_constant_subfeatures)
    
    active_features = [f for f in motif_features if f not in globally_missing_features]
    
    if not active_features and is_rot_const_globally_missing:
        print("  WARNING: No valid numerical features available for motif clustering. Skipping.")
        return
    
    # Build feature matrix for clustering
    features_for_scaling = []
    
    for d in cluster_representatives:
        feature_vector = []
        
        for feature_name in active_features:
            internal_key = FEATURE_MAPPING.get(feature_name, feature_name)
            value = d.get(internal_key)
            if value is None:
                value = 0.0
            feature_vector.append(value)
        
        # Add rotational constants if available
        if not is_rot_const_globally_missing:
            rc_temp = d.get('rotational_constants')
            if rc_temp is not None and isinstance(rc_temp, np.ndarray) and len(rc_temp) == 3:
                feature_vector.extend([rc_temp[0], rc_temp[1], rc_temp[2]])
            else:
                feature_vector.extend([0.0, 0.0, 0.0])
        
        features_for_scaling.append(feature_vector)
    
    if not features_for_scaling or not features_for_scaling[0]:
        print("  WARNING: No valid feature vectors for motif clustering. Skipping.")
        return
    
    features_matrix = np.array(features_for_scaling)
    
    # Apply absolute tolerances and scaling
    if abs_tolerances:
        feature_names_for_tolerances = active_features[:]
        if not is_rot_const_globally_missing:
            feature_names_for_tolerances.extend(rotational_constant_subfeatures)
        
        for col_idx, feature_name in enumerate(feature_names_for_tolerances):
            if feature_name in abs_tolerances:
                tolerance = abs_tolerances[feature_name]
                column = features_matrix[:, col_idx]
                if np.max(column) - np.min(column) <= tolerance:
                    features_matrix[:, col_idx] = 0.0
                    print(f"    Zeroed feature '{feature_name}' (max difference: {np.max(column) - np.min(column):.2e} <= tolerance: {tolerance:.2e})")
    
    # Scale features
    scaler = StandardScaler()
    try:
        features_scaled = scaler.fit_transform(features_matrix)
        
        # Check for low std deviation features
        feature_names_for_std_check = active_features[:]
        if not is_rot_const_globally_missing:
            feature_names_for_std_check.extend(rotational_constant_subfeatures)
        
        std_devs = scaler.scale_
        for col_idx, (feature_name, std_dev) in enumerate(zip(feature_names_for_std_check, std_devs)):
            if std_dev < min_std_threshold:
                features_scaled[:, col_idx] = 0.0
                print(f"    Zeroed low-variance feature '{feature_name}' (std dev: {std_dev:.2e} < threshold: {min_std_threshold:.2e})")
        
    except Exception as e:
        print(f"  ERROR in feature scaling for motif clustering: {e}")
        return
    
    # Perform hierarchical clustering
    try:
        linkage_matrix = linkage(features_scaled, method='average', metric='euclidean')
        motif_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')
    except Exception as e:
        print(f"  ERROR in motif clustering: {e}")
        return
    
    # Group representatives by motif labels
    motif_groups = {}
    for i, label in enumerate(motif_labels):
        motif_groups.setdefault(label, []).append(cluster_representatives[i])
    
    print(f"  Found {len(motif_groups)} unique final motifs")
    
    # Create unique motifs directory first
    unique_motifs_dir = os.path.join(output_base_dir, f"unique_motifs_{len(motif_groups):02d}")
    os.makedirs(unique_motifs_dir, exist_ok=True)
    
    # Sort motifs by lowest energy representative (needed for dendrogram labeling)
    sorted_motifs = []
    for motif_id, members in motif_groups.items():
        representative = min(members,
                           key=lambda x: (x.get('gibbs_free_energy') if x.get('gibbs_free_energy') is not None else float('inf'), x['filename']))
        sorted_motifs.append((motif_id, members, representative))
    
    sorted_motifs.sort(key=lambda x: (x[2].get('gibbs_free_energy') if x[2].get('gibbs_free_energy') is not None else float('inf'), x[2]['filename']))
    
    # Create dendrogram for unique motifs clustering
    try:
        import matplotlib.pyplot as plt
        
        # Create labels for dendrogram using original cluster numbers
        # We need to map each cluster representative to its original cluster number
        # First, create a mapping from representative index to original cluster number
        rep_to_original_num = {}
        for i, rep in enumerate(cluster_representatives):
            original_cluster_num = cluster_original_indices[i]
            rep_to_original_num[i] = original_cluster_num
        
        # Create labels based on the original cluster numbers
        motif_labels = [str(rep_to_original_num.get(i, i+1)) for i in range(len(cluster_representatives))]
        
        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, labels=motif_labels, orientation='top', 
                   distance_sort='descending', show_leaf_counts=True)
        plt.title(f'Unique Motifs Dendrogram (Excluding H-bond Count)\nThreshold: {threshold}')
        plt.xlabel('Original Cluster Numbers')
        plt.ylabel('Distance')
        plt.xticks(rotation=0)  # Keep numbers horizontal since they're short
        plt.tight_layout()
        
        # Save dendrogram in the unique motifs directory
        dendrogram_path = os.path.join(unique_motifs_dir, "umotifs_dendrogram.png")
        plt.savefig(dendrogram_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Created unique motifs dendrogram: {os.path.basename(dendrogram_path)}")
        
    except ImportError:
        print("  WARNING: matplotlib not available. Skipping unique motifs dendrogram creation.")
    except Exception as e:
        print(f"  WARNING: Error creating unique motifs dendrogram: {e}")
    
    # Create motif summary file in the unique motifs directory
    summary_filename = os.path.join(unique_motifs_dir, "umotif_summary.txt")
    
    with open(summary_filename, 'w') as f:
        f.write("=" * 60 + "\n\n")
        f.write("Final motif summary - excluding hydrogen bond count\n\n")
        f.write("-" * 60 + "\n\n")
        f.write(f"Original cluster representatives analyzed: {len(cluster_representatives)}\n")
        f.write(f"Number of truly unique motifs: {len(motif_groups)}\n")
        f.write(f"Clustering threshold: {threshold}\n\n")
        f.write("=" * 60 + "\n\n")
        
        # Write motif details
        for motif_idx, (motif_id, members, representative) in enumerate(sorted_motifs, 1):
            f.write(f"Final motif {motif_idx} (Label: {motif_id}) - {len(members)} structure(s)\n")
            f.write(f"Representative: {representative['filename']}\n")
            
            if representative.get('gibbs_free_energy') is not None:
                f.write(f"Representative energy: {representative['gibbs_free_energy']:.6f} Hartree\n")
            else:
                f.write("Representative energy: N/A\n")

            f.write("\nStructures grouped in this final motif:\n")
            
            # Sort members by energy
            sorted_members = sorted(members,
                                  key=lambda x: (x.get('gibbs_free_energy') if x.get('gibbs_free_energy') is not None else float('inf'), x['filename']))
            
            for member in sorted_members:
                energy_str = f"{member['gibbs_free_energy']:.6f}" if member.get('gibbs_free_energy') is not None else "N/A"
                hbond_count = member.get('num_hydrogen_bonds', 'N/A')
                f.write(f"  - {member['filename']} (Energy: {energy_str} Hartree, H-bonds: {hbond_count})\n")
            
            # Show hydrogen bond range if multiple members
            if len(members) > 1:
                hbond_counts = [m.get('num_hydrogen_bonds') for m in members if m.get('num_hydrogen_bonds') is not None]
                if min(hbond_counts) != max(hbond_counts):
                    f.write(f"H-bond range: {min(hbond_counts)} - {max(hbond_counts)}\n")
            
            f.write("\n" + "-" * 60 + "\n\n")
    
    print(f"  Final motif summary written to: {os.path.basename(summary_filename)}")
    
    print(f"  Creating final motif representatives directory: {os.path.basename(unique_motifs_dir)}")
    
    # Create cluster_representatives directory
    cluster_reps_dir = os.path.join(output_base_dir, "cluster_representatives")
    os.makedirs(cluster_reps_dir, exist_ok=True)
    
    # Create a mapping from representative filename to original cluster number
    rep_to_original_cluster = {}
    for i, rep in enumerate(cluster_representatives):
        rep_to_original_cluster[rep['filename']] = cluster_original_indices[i]
    
    # Create individual XYZ files and collect for combined file
    final_representatives = []
    final_motif_numbers = []  # Track the original cluster numbers for each motif
    
    # Process each motif and create cluster subdirectories
    for motif_idx, (motif_id, members, representative) in enumerate(sorted_motifs, 1):
        # Create subdirectory for this motif cluster
        motif_subdir = os.path.join(cluster_reps_dir, f"umotif_{motif_idx:02d}")
        os.makedirs(motif_subdir, exist_ok=True)
        
        # Create XYZ files for all members in this motif (not just representative)
        motif_xyz_files = []
        for member in members:
            base_name = os.path.splitext(member['filename'])[0]
            # Keep original naming convention (no u_motif prefix)
            member_filename = f"motif_{rep_to_original_cluster[member['filename']]:02d}_{base_name}.xyz"
            member_path = os.path.join(motif_subdir, member_filename)
            
            write_xyz_file(member, member_path)
            motif_xyz_files.append(member_path)
        
        # Create combined XYZ and MOL files for this motif cluster
        if motif_xyz_files:
            combined_xyz = os.path.join(motif_subdir, f"umotif_{motif_idx:02d}_combined.xyz")
            combined_mol = os.path.join(motif_subdir, f"umotif_{motif_idx:02d}_combined.mol")
            
            # Create combined XYZ file
            with open(combined_xyz, 'w') as outfile:
                for i, xyz_file in enumerate(motif_xyz_files):
                    with open(xyz_file, 'r') as infile:
                        content = infile.read()
                        outfile.write(content)
                        if i < len(motif_xyz_files) - 1:  # Add separator between structures
                            outfile.write('\n')
            
            # Convert to MOL format using OpenBabel if available
            try:
                subprocess.run(["obabel", combined_xyz, '-O', combined_mol], 
                              check=True, capture_output=True, text=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                # Try alternative obabel names
                for obabel_cmd in ["babel", "openbabel"]:
                    try:
                        subprocess.run([obabel_cmd, combined_xyz, '-O', combined_mol], 
                                      check=True, capture_output=True, text=True)
                        break
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
        
        # Keep track of representatives for the unique_motifs folder
        base_name = os.path.splitext(representative['filename'])[0]
        original_cluster_num = rep_to_original_cluster[representative['filename']]
        motif_filename = f"motif_{original_cluster_num:02d}_{base_name}.xyz"
        motif_path = os.path.join(unique_motifs_dir, motif_filename)
        
        write_xyz_file(representative, motif_path)
        final_representatives.append(representative)
        final_motif_numbers.append(original_cluster_num)  # Store the original cluster number
        
        energy_str = f"{representative['gibbs_free_energy']:.6f}" if representative['gibbs_free_energy'] is not None else "N/A"
        print(f"    Motif {motif_idx:02d}: {len(members)} structure(s), Representative: {base_name} (Energy: {energy_str} Hartree)")
    
    print(f"  Created cluster representatives directory with {len(sorted_motifs)} motif subdirectories")
    
    # Create combined XYZ and MOL files for final motifs (in unique_motifs folder)
    print(f"  Creating combined files for {len(final_representatives)} final motifs...")
    combine_xyz_files(final_representatives, unique_motifs_dir, output_base_name="all_unique_motifs_combined", prefix_template="motif_{:02d}_", motif_numbers=final_motif_numbers)


# Modified to accept rmsd_threshold and output_base_dir
def perform_clustering_and_analysis(input_source, threshold=1.0, file_extension_pattern=None, rmsd_threshold=None, output_base_dir=None, force_reprocess_cache=False, weights=None, is_compare_mode=False, min_std_threshold=1e-6, abs_tolerances=None, motif_threshold=1.0):
    """
    Performs hierarchical clustering and analysis on the extracted molecular properties,
    and saves .dat and .xyz files for each cluster.
    Includes an optional RMSD post-processing step and caching of extracted data.
    Output files will be saved relative to output_base_dir.
    `input_source` can be a folder path (normal mode) or a list of file paths (compare mode).
    `weights` is a dictionary mapping feature names (user-friendly) to their weights.
    `min_std_threshold` (float): Minimum standard deviation for a feature to be scaled.
                                 Features with std dev below this are treated as constant (0.0).
    `abs_tolerances` (dict): Dictionary of feature_name: absolute_tolerance. If the max difference
                             for a feature within a group is less than its tolerance, it's zeroed out.
    """
    if weights is None:
        weights = {} # Ensure weights is a dict even if not provided
    if abs_tolerances is None:
        abs_tolerances = {} # Ensure abs_tolerances is a dict if not provided

    # Ensure output_base_dir is set, default to current working directory if None
    if output_base_dir is None:
        output_base_dir = os.getcwd()

    # NEW: Adjust output_base_dir for comparison mode and handle unique naming
    if is_compare_mode:
        base_comparison_dir_name = "comparison"
        final_comparison_dir = base_comparison_dir_name
        counter = 0
        while os.path.exists(os.path.join(output_base_dir, final_comparison_dir)):
            counter += 1
            final_comparison_dir = f"{base_comparison_dir_name}_{counter}"
        output_base_dir = os.path.join(output_base_dir, final_comparison_dir)
        os.makedirs(output_base_dir, exist_ok=True) # Ensure this new base directory exists
        print(f"  Comparison mode: All outputs will be placed in '{output_base_dir}'")

    # Define a generic cache file path
    cache_file_name = "data_cache.pkl" # Shortened cache file name
    cache_file_path = os.path.join(output_base_dir, cache_file_name)

    all_extracted_data = []
    
    files_to_process = []
    if is_compare_mode:
        files_to_process = input_source # input_source is already the list of files
        # For compare mode, we should probably bypass cache loading/saving, or make it specific to the comparison.
        # For now, let's assume compare mode always re-processes the two files.
        print(f"Starting data extraction for comparison mode from {len(files_to_process)} files...")
        for file_path in sorted(files_to_process):
            print(f"  Extracting from: {os.path.basename(file_path)}...")
            extracted_props = extract_properties_from_logfile(file_path)
            if extracted_props:
                all_extracted_data.append(extracted_props)
    else:
        # Existing cache logic for normal mode
        if os.path.exists(cache_file_path) and not force_reprocess_cache:
            print(f"Attempting to load data from cache: '{os.path.basename(cache_file_path)}'")
            try:
                with open(cache_file_path, 'rb') as f:
                    cached_data = pickle.load(f)
                
                current_files_in_folder = {os.path.basename(f) for f in glob.glob(os.path.join(input_source, file_extension_pattern))}
                retained_cached_data = [d for d in cached_data if d['filename'] in current_files_in_folder]
                
                if len(retained_cached_data) == len(current_files_in_folder):
                    all_extracted_data = retained_cached_data
                    print(f"Data loaded from cache successfully. ({len(all_extracted_data)} entries)")
                else:
                    print("Cache data incomplete or outdated. Re-extracting all files.")
                    all_extracted_data = []
                    if os.path.exists(cache_file_path):
                        os.remove(cache_file_path)

            except Exception as e:
                print(f"Error loading data from cache: {e}. Re-extracting all files.")
                all_extracted_data = []
                if os.path.exists(cache_file_path):
                    os.remove(cache_file_path)

        files_to_process = glob.glob(os.path.join(input_source, file_extension_pattern))
        if not files_to_process:
            print(f"No files matching '{file_extension_pattern}' found in '{input_source}'. Skipping this folder.")
            return

        if not all_extracted_data:
            print(f"Starting data extraction from {len(files_to_process)} files matching '{file_extension_pattern}' in '{input_source}'...")
            for file_path in sorted(files_to_process):
                print(f"  Extracting from: {os.path.basename(file_path)}...")
                extracted_props = extract_properties_from_logfile(file_path)
                if extracted_props:
                    all_extracted_data.append(extracted_props)
            
            if all_extracted_data and not is_compare_mode: # Only save to cache in normal mode
                print(f"Saving extracted data to cache: '{os.path.basename(cache_file_path)}'")
                try:
                    with open(cache_file_path, 'wb') as f:
                        pickle.dump(all_extracted_data, f)
                    print("Data saved to cache successfully.")
                except Exception as e:
                    print(f"  Error saving data to cache: {e}")

    if not all_extracted_data:
        print(f"No data was successfully extracted from files. Skipping clustering.")
        return

    print("\nData extraction complete. Proceeding to clustering.")
    
    clean_data_for_clustering = []
    essential_base_features = ['final_geometry_atomnos', 'final_geometry_coords', 'num_hydrogen_bonds']
    
    for mol_data in all_extracted_data:
        is_essential_missing = False
        missing_essential_info = []
        for f in essential_base_features:
            if mol_data.get(f) is None:
                is_essential_missing = True
                missing_essential_info.append(f"Missing essential feature '{f}'")
        
        if not is_essential_missing:
            clean_data_for_clustering.append(mol_data)
        else:
            print(f"Skipping '{mol_data.get('filename', 'Unknown')}' for clustering due to: {'; '.join(missing_essential_info)}")


    if not clean_data_for_clustering:
        print(f"No complete data entries to cluster after filtering. Exiting clustering step.")
        return

    hbond_groups = {}
    
    # NEW: Handle comparison mode's clustering logic
    if is_compare_mode and len(clean_data_for_clustering) >= 2:
        print("  Comparison mode: Running clustering to generate dendrogram, then forcing a single output cluster.")
        # For comparison, we will treat them as a single group for dendrogram generation
        # and then force them into one output cluster.
        group_data_for_clustering = sorted(clean_data_for_clustering,
                                           key=lambda x: (x.get('gibbs_free_energy') if x.get('gibbs_free_energy') is not None else float('inf'), x['filename']))
        
        # This will be the only "hbond group" processed in comparison mode, effectively.
        hbond_groups = {0: group_data_for_clustering} # Use 0 as a dummy hbond count
    else:
        # Original logic for non-comparison mode
        for item in clean_data_for_clustering:
            hbond_groups.setdefault(item['num_hydrogen_bonds'], []).append(item)

    # Output directory paths
    dendrogram_images_folder = os.path.join(output_base_dir, "dendrogram_images")
    extracted_data_folder = os.path.join(output_base_dir, "extracted_data")
    extracted_clusters_folder = os.path.join(output_base_dir, "extracted_clusters")
    
    os.makedirs(dendrogram_images_folder, exist_ok=True)
    os.makedirs(extracted_data_folder, exist_ok=True)
    os.makedirs(extracted_clusters_folder, exist_ok=True)

    print(f"Dendrogram images will be saved to '{dendrogram_images_folder}'")
    print(f"Extracted data files will be saved to '{extracted_data_folder}'")
    print(f"Extracted cluster XYZ/MOL files will be saved to '{extracted_clusters_folder}'")

    summary_file_content_lines = []
    comparison_specific_summary_lines = [] # New list for comparison-specific details

    total_clusters_outputted = 0
    total_rmsd_outliers_first_pass = 0

    # Helper function to center text within 75 characters
    def center_text(text, width=75):
        return text.center(width)
    
    # Add ASCII art header similar to ASCEC but for Similarity
    summary_file_content_lines.append("=" * 75)
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text("***************************"))
    summary_file_content_lines.append(center_text("* S I M I L A R I T Y *"))
    summary_file_content_lines.append(center_text("***************************"))
    summary_file_content_lines.append("")
    summary_file_content_lines.append("                             √≈≠==≈                                  ")
    summary_file_content_lines.append("   √≈≠==≠≈√   √≈≠==≠≈√         ÷++=                      ≠===≠       ")
    summary_file_content_lines.append("     ÷++÷       ÷++÷           =++=                     ÷×××××=      ")
    summary_file_content_lines.append("     =++=       =++=     ≠===≠ ÷++=      ≠====≠         ÷-÷ ÷-÷      ")
    summary_file_content_lines.append("     =++=       =++=    =××÷=≠=÷++=    ≠÷÷÷==÷÷÷≈      ≠××≠ =××=     ")
    summary_file_content_lines.append("     =++=       =++=   ≠××=    ÷++=   ≠×+×    ×+÷      ÷+×   ×+××    ")
    summary_file_content_lines.append("     =++=       =++=   =+÷     =++=   =+-×÷==÷×-×≠    =×+×÷=÷×+-÷    ")
    summary_file_content_lines.append("     ≠×+÷       ÷+×≠   =+÷     =++=   =+---×××××÷×   ≠××÷==×==÷××≠   ")
    summary_file_content_lines.append("      =××÷     =××=    ≠××=    ÷++÷   ≠×-×           ÷+×       ×+÷   ")
    summary_file_content_lines.append("       ≠=========≠      ≠÷÷÷=≠≠=×+×÷-  ≠======≠≈√  -÷×+×≠     ≠×+×÷- ")
    summary_file_content_lines.append("          ≠===≠           ≠==≠  ≠===≠     ≠===≠    ≈====≈     ≈====≈ ")
    summary_file_content_lines.append("")
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text("Universidad de Antioquia - Medellín - Colombia"))
    summary_file_content_lines.append("")
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text("Clustering Analysis for Quantum Chemistry Calculations"))
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text(version))
    summary_file_content_lines.append("")
    summary_file_content_lines.append("")
    summary_file_content_lines.append(center_text("Química Física Teórica - QFT"))
    summary_file_content_lines.append("")
    summary_file_content_lines.append("")
    summary_file_content_lines.append("=" * 75 + "\n")
    if is_compare_mode:
        summary_file_content_lines.append(f"Comparison Results for: {', '.join([os.path.basename(f) for f in input_source])}")
    else:
        summary_file_content_lines.append(f"Clustering Results for: {os.path.basename(input_source)}")
    
    # Conditional similarity threshold display
    if is_compare_mode:
        summary_file_content_lines.append(f"Similarity threshold (distance): N/A")
    else:
        summary_file_content_lines.append(f"Similarity threshold (distance): {threshold}")
    
    if rmsd_threshold is not None:
        summary_file_content_lines.append(f"RMSD validation threshold: {rmsd_threshold:.3f} Å")
    
    # Moved these to comparison_specific_summary_lines for conditional output
    # if weights:
    #     summary_file_content_lines.append(f"Applied Feature Weights: {weights}")
    # if abs_tolerances:
    #     summary_file_content_lines.append(f"Applied Absolute Tolerances: {abs_tolerances}")

    summary_file_content_lines.append(f"Total configurations processed: {len(clean_data_for_clustering)}")
    summary_file_content_lines.append(f"Total number of final clusters: <TOTAL_CLUSTERS_PLACEHOLDER>")
    if rmsd_threshold is not None:
        summary_file_content_lines.append(f"Total RMSD moved configurations: <TOTAL_RMSD_OUTLIERS_PLACEHOLDER>")
    summary_file_content_lines.append("\n" + "=" * 75 + "\n")


    previous_hbond_group_processed = False 

    # --- Boltzmann Population Calculation (based on initial property clusters) ---
    all_initial_property_clusters = []
    pseudo_global_cluster_id_counter = 1 # This counter is for assigning unique IDs to initial clusters for Boltzmann calc

    for hbond_count, group_data in sorted(hbond_groups.items()):
        if len(group_data) < 2 or not any(d.get(f) is not None for d in group_data for f in ['radius_of_gyration', 'dipole_moment', 'homo_lumo_gap', 'first_vib_freq', 'last_vib_freq', 'average_hbond_distance', 'average_hbond_angle', 'rotational_constants']):
            # For singletons or groups without enough numerical features, treat each as a separate initial cluster
            for single_mol_data in group_data:
                single_mol_data['_initial_cluster_label'] = hbond_count # This is a dummy label for singletons
                single_mol_data['_parent_global_cluster_id'] = pseudo_global_cluster_id_counter # Assign unique ID
                all_initial_property_clusters.append([single_mol_data])
                pseudo_global_cluster_id_counter += 1
        else:
            filenames_base = [os.path.splitext(item['filename'])[0] for item in group_data]
            
            all_potential_numerical_features = [
                'radius_of_gyration', 'dipole_moment', 'homo_lumo_gap',
                'first_vib_freq', 'last_vib_freq', 'average_hbond_distance', 
                'average_hbond_angle'
            ]
            rotational_constant_subfeatures = [
                'rotational_constants_A', 'rotational_constants_B', 'rotational_constants_C'
            ]

            globally_missing_for_group = []
            for feature in all_potential_numerical_features:
                if all(d.get(feature) is None for d in group_data):
                    globally_missing_for_group.append(feature)
            
            is_rot_const_globally_missing_for_group = True
            for d in group_data:
                rot_consts = d.get('rotational_constants')
                if rot_consts is not None and isinstance(rot_consts, np.ndarray) and rot_consts.ndim == 1 and len(rot_consts) == 3:
                    is_rot_const_globally_missing_for_group = False
                    break
            
            if is_rot_const_globally_missing_for_group:
                globally_missing_for_group.extend(rotational_constant_subfeatures)

            active_numerical_features_for_group = [f for f in all_potential_numerical_features if f not in globally_missing_for_group]
            
            features_for_scaling_raw = []
            ordered_feature_names_for_scaling = []

            for d in group_data:
                mol_feature_vector = []
                current_mol_ordered_feature_names = []
                for feature_name_user_friendly in active_numerical_features_for_group:
                    internal_key = FEATURE_MAPPING.get(feature_name_user_friendly, feature_name_user_friendly)
                    if weights.get(feature_name_user_friendly, 1.0) != 0.0:
                        value = d.get(internal_key)
                        if value is None: value = 0.0
                        mol_feature_vector.append(value)
                        current_mol_ordered_feature_names.append(feature_name_user_friendly)
                
                if not is_rot_const_globally_missing_for_group:
                    rc_temp = d.get('rotational_constants')
                    if rc_temp is not None and isinstance(rc_temp, np.ndarray) and len(rc_temp) == 3:
                        if weights.get('rotational_constants_A', 1.0) != 0.0:
                            mol_feature_vector.append(rc_temp[0])
                            current_mol_ordered_feature_names.append('rotational_constants_A')
                        if weights.get('rotational_constants_B', 1.0) != 0.0:
                            mol_feature_vector.append(rc_temp[1])
                            current_mol_ordered_feature_names.append('rotational_constants_B')
                        if weights.get('rotational_constants_C', 1.0) != 0.0:
                            mol_feature_vector.append(rc_temp[2])
                            current_mol_ordered_feature_names.append('rotational_constants_C')
                    else:
                        if weights.get('rotational_constants_A', 1.0) != 0.0:
                            mol_feature_vector.append(0.0)
                            current_mol_ordered_feature_names.append('rotational_constants_A')
                        if weights.get('rotational_constants_B', 1.0) != 0.0:
                            mol_feature_vector.append(0.0)
                            current_mol_ordered_feature_names.append('rotational_constants_B')
                        if weights.get('rotational_constants_C', 1.0) != 0.0:
                            mol_feature_vector.append(0.0)
                            current_mol_ordered_feature_names.append('rotational_constants_C')

                features_for_scaling_raw.append(mol_feature_vector)
                if not ordered_feature_names_for_scaling:
                    ordered_feature_names_for_scaling = current_mol_ordered_feature_names
            
            if not features_for_scaling_raw or all(len(f) == 0 for f in features_for_scaling_raw):
                for single_mol_data in group_data:
                    single_mol_data['_initial_cluster_label'] = hbond_count
                    single_mol_data['_parent_global_cluster_id'] = pseudo_global_cluster_id_counter
                    all_initial_property_clusters.append([single_mol_data])
                    pseudo_global_cluster_id_counter += 1
                continue

            features_for_scaling_raw_np = np.array(features_for_scaling_raw, dtype=float)
            features_scaled = np.zeros_like(features_for_scaling_raw_np)
            for col_idx in range(features_for_scaling_raw_np.shape[1]):
                col_data = features_for_scaling_raw_np[:, col_idx]
                feature_name = ordered_feature_names_for_scaling[col_idx]
                max_abs_diff = np.max(col_data) - np.min(col_data)
                
                if feature_name in abs_tolerances and max_abs_diff < abs_tolerances[feature_name]:
                    features_scaled[:, col_idx] = 0.0
                else:
                    std_dev = np.std(col_data)
                    if std_dev < min_std_threshold:
                        features_scaled[:, col_idx] = 0.0
                    else:
                        scaler = StandardScaler()
                        features_scaled[:, col_idx] = scaler.fit_transform(col_data.reshape(-1, 1)).flatten()

            linkage_matrix = linkage(features_scaled, method='average', metric='euclidean')
            initial_cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')

            initial_clusters_data = {}
            for i, label in enumerate(initial_cluster_labels):
                group_data[i]['_initial_cluster_label'] = label 
                initial_clusters_data.setdefault(label, []).append(group_data[i])
            
            initial_clusters_list_unsorted = list(initial_clusters_data.values())
            
            # Sort these initial clusters by their lowest energy representative
            initial_clusters_list_sorted_by_energy = sorted(
                initial_clusters_list_unsorted,
                key=lambda cluster: (min(m.get('gibbs_free_energy') if m.get('gibbs_free_energy') is not None else float('inf') for m in cluster), 
                                     min(m['filename'] for m in cluster)) 
            )

            for initial_prop_cluster in initial_clusters_list_sorted_by_energy:
                parent_id = pseudo_global_cluster_id_counter 
                for member_conf in initial_prop_cluster:
                    member_conf['_parent_global_cluster_id'] = parent_id 
                all_initial_property_clusters.append(initial_prop_cluster) # Add to the list for Boltzmann calculation
                pseudo_global_cluster_id_counter += 1 

    boltzmann_g1_data = {}
    boltzmann_g_deg_data = {}
    global_min_gibbs_energy = None
    global_min_rep_filename = "N/A"
    global_min_cluster_id = "N/A"

    if all_initial_property_clusters:
        # Find the global minimum Gibbs energy among all representatives
        # Also store the filename and cluster ID of this global minimum representative
        valid_reps_for_emin = []
        for initial_prop_cluster in all_initial_property_clusters:
            # Select the lowest energy member as the representative for this initial property cluster
            rep_conf = min(initial_prop_cluster,
                           key=lambda x: (x.get('gibbs_free_energy') if x.get('gibbs_free_energy') is not None else float('inf'), x['filename']))
            
            if rep_conf.get('gibbs_free_energy') is not None:
                valid_reps_for_emin.append({
                    'energy': rep_conf['gibbs_free_energy'],
                    'filename': rep_conf['filename'],
                    'cluster_id': rep_conf['_parent_global_cluster_id']
                })

        if valid_reps_for_emin:
            global_min_info = min(valid_reps_for_emin, key=lambda x: x['energy'])
            global_min_gibbs_energy = global_min_info['energy']
            global_min_rep_filename = global_min_info['filename']
            global_min_cluster_id = global_min_info['cluster_id']

            sum_factors_g1 = 0.0
            sum_factors_g_deg = 0.0

            # Calculate Boltzmann factors for each initial property cluster
            for initial_prop_cluster in all_initial_property_clusters:
                # Use the representative's energy for the Boltzmann calculation
                rep_conf = min(initial_prop_cluster,
                               key=lambda x: (x.get('gibbs_free_energy') if x.get('gibbs_free_energy') is not None else float('inf'), x['filename']))
                
                if rep_conf.get('gibbs_free_energy') is None:
                    continue # Skip if no valid representative energy

                rep_gibbs_energy = rep_conf['gibbs_free_energy']
                cluster_id = rep_conf['_parent_global_cluster_id']
                cluster_size = len(initial_prop_cluster) # Size of the initial property cluster

                delta_e = rep_gibbs_energy - global_min_gibbs_energy
                
                if BOLTZMANN_CONSTANT_HARTREE_PER_K * DEFAULT_TEMPERATURE_K == 0:
                    factor_g1 = 1.0 if delta_e == 0 else 0.0
                else:
                    factor_g1 = np.exp(-delta_e / (BOLTZMANN_CONSTANT_HARTREE_PER_K * DEFAULT_TEMPERATURE_K))
                
                factor_g_deg = cluster_size * factor_g1
                
                boltzmann_g1_data[cluster_id] = {
                    'energy': rep_gibbs_energy,
                    'filename': rep_conf['filename'],
                    'population': factor_g1,
                    'cluster_size': cluster_size
                }
                boltzmann_g_deg_data[cluster_id] = {
                    'energy': rep_gibbs_energy,
                    'filename': rep_conf['filename'],
                    'population': factor_g_deg,
                    'cluster_size': cluster_size
                }
                
                sum_factors_g1 += factor_g1
                sum_factors_g_deg += factor_g_deg
            
            # Normalize to percentages
            if sum_factors_g1 > 0:
                for cluster_id, data in boltzmann_g1_data.items():
                    data['population'] = (data['population'] / sum_factors_g1) * 100.0
            else:
                for cluster_id in boltzmann_g1_data:
                    boltzmann_g1_data[cluster_id]['population'] = 0.0

            if sum_factors_g_deg > 0:
                for cluster_id, data in boltzmann_g_deg_data.items():
                    data['population'] = (data['population'] / sum_factors_g_deg) * 100.0
            else:
                for cluster_id in boltzmann_g_deg_data:
                    boltzmann_g_deg_data[cluster_id]['population'] = 0.0
    # --- End Boltzmann Population Calculation ---


    # Collect all final clusters for unique motifs creation
    all_final_clusters = []

    # Now iterate through the hbond_groups again to perform the clustering and write files
    # This loop is responsible for generating the actual clusters and writing their files.
    # The Boltzmann data calculated above will be passed to write_cluster_dat_file.
    cluster_global_id_counter = 1 # Reset for final cluster numbering in summary
    for hbond_count, group_data in sorted(hbond_groups.items()):
        
        if previous_hbond_group_processed:
            summary_file_content_lines.append("\n" + "-" * 75 + "\n") 
        
        summary_file_content_lines.append(f"Hydrogen bonds: {hbond_count}\n")
        summary_file_content_lines.append(f"Configurations: {len(group_data)}")

        current_hbond_group_clusters_for_final_output = [] 

        if len(group_data) < 2 or not any(d.get(f) is not None for d in group_data for f in ['radius_of_gyration', 'dipole_moment', 'homo_lumo_gap', 'first_vib_freq', 'last_vib_freq', 'average_hbond_distance', 'average_hbond_angle', 'rotational_constants']):
            print(f"\nSkipping detailed clustering for H-bond group {hbond_count}: Less than 2 configurations or no valid numerical features left after filtering. Treating each as a single-configuration cluster.")
            
            for single_mol_data in group_data:
                single_mol_data['_rmsd_pass_origin'] = 'first_pass_validated' 
                current_hbond_group_clusters_for_final_output.append([single_mol_data]) 

        else: # Proceed with actual clustering for normal mode or if more than 2 files in a group
            filenames_base = [os.path.splitext(item['filename'])[0] for item in group_data]
            
            all_potential_numerical_features = [
                'radius_of_gyration', 'dipole_moment', 'homo_lumo_gap',
                'first_vib_freq', 'last_vib_freq', 'average_hbond_distance', 
                'average_hbond_angle'
            ]
            rotational_constant_subfeatures = [
                'rotational_constants_A', 'rotational_constants_B', 'rotational_constants_C'
            ]

            globally_missing_for_group = []
            for feature in all_potential_numerical_features:
                if all(d.get(feature) is None for d in group_data):
                    globally_missing_for_group.append(feature)
            
            is_rot_const_globally_missing_for_group = True
            for d in group_data:
                rot_consts = d.get('rotational_constants')
                if rot_consts is not None and isinstance(rot_consts, np.ndarray) and rot_consts.ndim == 1 and len(rot_consts) == 3:
                    is_rot_const_globally_missing_for_group = False
                    break
            
            if is_rot_const_globally_missing_for_group:
                globally_missing_for_group.extend(rotational_constant_subfeatures)
                print(f"  Note: Rotational constants are globally missing or invalid for H-bond group {hbond_count}. Excluding them from clustering features.")

            active_numerical_features_for_group = [f for f in all_potential_numerical_features if f not in globally_missing_for_group]
            
            features_for_scaling_raw = [] # Store raw values before conditional scaling
            ordered_feature_names_for_scaling = [] # To keep track of the order for weights

            for d in group_data:
                mol_feature_vector = []
                current_mol_ordered_feature_names = []
                for feature_name_user_friendly in active_numerical_features_for_group:
                    internal_key = FEATURE_MAPPING.get(feature_name_user_friendly, feature_name_user_friendly)
                    # Only add to feature vector if the weight is not 0.0
                    if weights.get(feature_name_user_friendly, 1.0) != 0.0:
                        value = d.get(internal_key)
                        if value is None:
                            value = 0.0 # Default missing numerical values to 0.0 for scaling
                        mol_feature_vector.append(value)
                        current_mol_ordered_feature_names.append(feature_name_user_friendly)
                
                if not is_rot_const_globally_missing_for_group:
                    # Only add rotational constants if their weights are not 0.0
                    rc_temp = d.get('rotational_constants')
                    if rc_temp is not None and isinstance(rc_temp, np.ndarray) and len(rc_temp) == 3:
                        if weights.get('rotational_constants_A', 1.0) != 0.0:
                            val = rc_temp[0]
                            mol_feature_vector.append(val)
                            current_mol_ordered_feature_names.append('rotational_constants_A')
                        if weights.get('rotational_constants_B', 1.0) != 0.0:
                            val = rc_temp[1]
                            mol_feature_vector.append(val)
                            current_mol_ordered_feature_names.append('rotational_constants_B')
                        if weights.get('rotational_constants_C', 1.0) != 0.0:
                            val = rc_temp[2]
                            mol_feature_vector.append(val)
                            current_mol_ordered_feature_names.append('rotational_constants_C')
                    else: # If rotational constants are missing, add zeros if their weights are not 0.0
                        # Ensure we add placeholders if the feature is not zero-weighted, even if data is missing
                        if weights.get('rotational_constants_A', 1.0) != 0.0:
                            val = 0.0
                            mol_feature_vector.append(val)
                            current_mol_ordered_feature_names.append('rotational_constants_A')
                        if weights.get('rotational_constants_B', 1.0) != 0.0:
                            val = 0.0
                            mol_feature_vector.append(val)
                            current_mol_ordered_feature_names.append('rotational_constants_B')
                        if weights.get('rotational_constants_C', 1.0) != 0.0:
                            val = 0.0
                            mol_feature_vector.append(val)
                            current_mol_ordered_feature_names.append('rotational_constants_C')

                features_for_scaling_raw.append(mol_feature_vector)
                if not ordered_feature_names_for_scaling:
                    ordered_feature_names_for_scaling = current_mol_ordered_feature_names
            
            # Check if there are any features left to scale after applying weights
            if not features_for_scaling_raw or all(len(f) == 0 for f in features_for_scaling_raw):
                print(f"  WARNING: No numerical features left for clustering after applying weights. Treating each as a single-configuration cluster.")
                for single_mol_data in group_data:
                    single_mol_data['_rmsd_pass_origin'] = 'first_pass_validated' 
                    current_hbond_group_clusters_for_final_output.append([single_mol_data]) 
                continue # Skip to next hbond group

            # Convert to numpy array for easier column-wise operations
            features_for_scaling_raw_np = np.array(features_for_scaling_raw, dtype=float)
            
            # Apply conditional scaling with absolute tolerance
            features_scaled = np.zeros_like(features_for_scaling_raw_np)
            for col_idx in range(features_for_scaling_raw_np.shape[1]):
                col_data = features_for_scaling_raw_np[:, col_idx]
                feature_name = ordered_feature_names_for_scaling[col_idx]
                
                # Check for absolute tolerance first
                max_abs_diff = np.max(col_data) - np.min(col_data)
                
                if feature_name in abs_tolerances and max_abs_diff < abs_tolerances[feature_name]:
                    # If max difference is below absolute tolerance, treat as constant (0.0)
                    features_scaled[:, col_idx] = 0.0
                    print(f"  NOTE: Feature '{feature_name}' (column {col_idx}) has max abs diff {max_abs_diff:.2e} < abs tolerance {abs_tolerances[feature_name]:.2e}. Treating as constant (0.0).")
                else:
                    # Fallback to min_std_threshold if no specific abs_tolerance or abs_tolerance not met
                    std_dev = np.std(col_data)
                    if std_dev < min_std_threshold:
                        features_scaled[:, col_idx] = 0.0
                        print(f"  NOTE: Feature '{feature_name}' (column {col_idx}) has std dev {std_dev:.2e} < min std threshold {min_std_threshold:.2e}. Treating as constant (0.0).")
                    else:
                        scaler = StandardScaler()
                        features_scaled[:, col_idx] = scaler.fit_transform(col_data.reshape(-1, 1)).flatten()
                        print(f"  NOTE: Feature '{feature_name}' (column {col_idx}) scaled normally.")

            linkage_matrix = linkage(features_scaled, method='average', metric='euclidean')

            # Dendrogram generation - now always runs if there's data to cluster
            plt.figure(figsize=(12, 8))
            
            # Check if all distances are effectively zero and add a small offset for visualization
            if np.all(linkage_matrix[:, 2] == 0.0):
                linkage_matrix[:, 2] += 1e-12 # Add a tiny offset for visualization

            dendrogram(linkage_matrix, labels=filenames_base, leaf_rotation=90, leaf_font_size=8)
            
            dendrogram_title_suffix = "Comparison" if is_compare_mode else f"H-bonds = {hbond_count}"
            plt.title(f"Hierarchical Clustering Dendrogram ({dendrogram_title_suffix})")
            
            plt.ylabel("Euclidean Distance")
            plt.ylim(bottom=0) # Ensure y-axis starts at 0 or above
            
            # --- MODIFIED DENDROGRAM FILENAME LOGIC ---
            if is_compare_mode:
                dendrogram_filename = os.path.join(dendrogram_images_folder, f"dendrogram.png")
            else:
                dendrogram_filename = os.path.join(dendrogram_images_folder, f"dendrogram_H{hbond_count}.png")
            # --- END MODIFIED DENDROGRAM FILENAME LOGIC ---

            plt.tight_layout()
            plt.savefig(dendrogram_filename)
            plt.close()
            print(f"Dendrogram saved as '{os.path.basename(dendrogram_filename)}'")

            if is_compare_mode and len(group_data) >= 2:
                # For comparison mode, after generating the dendrogram,
                # we force the two files into a single output cluster.
                # The 'threshold' and 'fcluster' are not used to determine the output clusters here.
                print("  Comparison mode: Overriding clustering to output a single combined cluster.")
                
                # Manually set cluster properties for the forced cluster
                for i, member in enumerate(group_data): # group_data is already sorted by energy
                    member['_initial_cluster_label'] = 1 # Always label as 1 for comparison
                    member['_parent_global_cluster_id'] = 1 # Always label as 1 for comparison
                    member['_rmsd_pass_origin'] = 'first_pass_validated' # Mark as validated
                    member['_second_rmsd_sub_cluster_id'] = None
                    member['_second_rmsd_context_listing'] = None
                    member['_second_rmsd_rep_filename'] = None

                # Populate _first_rmsd_context_listing for the forced cluster
                prop_rep_conf = group_data[0] # Lowest energy is the representative
                first_rmsd_listing = []
                for member_conf in group_data:
                    if member_conf == prop_rep_conf:
                        rmsd_val = 0.0
                    elif prop_rep_conf.get('final_geometry_coords') is not None and prop_rep_conf.get('final_geometry_atomnos') is not None and \
                         member_conf.get('final_geometry_coords') is not None and \
                         member_conf.get('final_geometry_atomnos') is not None:
                        
                        rmsd_val = calculate_rmsd(
                            prop_rep_conf['final_geometry_atomnos'], prop_rep_conf['final_geometry_coords'], # Corrected key here
                            member_conf['final_geometry_atomnos'], member_conf['final_geometry_coords']
                        )
                    else:
                        rmsd_val = None
                    first_rmsd_listing.append({'filename': member_conf['filename'], 'rmsd_to_rep': rmsd_val})
                
                for member_conf in group_data:
                    member_conf['_first_rmsd_context_listing'] = first_rmsd_listing

                current_hbond_group_clusters_for_final_output.append(group_data) # Add the single combined cluster
            else:
                # Normal clustering logic for non-comparison mode or >2 files in a group
                initial_cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')

                initial_clusters_data = {}
                for i, label in enumerate(initial_cluster_labels):
                    group_data[i]['_initial_cluster_label'] = label 
                    # The _parent_global_cluster_id should already be set for these from the Boltzmann calculation setup
                    initial_clusters_data.setdefault(label, []).append(group_data[i])
                
                initial_clusters_list_unsorted = list(initial_clusters_data.values())

                initial_clusters_list_sorted_by_energy = sorted(
                    initial_clusters_list_unsorted,
                    key=lambda cluster: (min(m.get('gibbs_free_energy') if m.get('gibbs_free_energy') is not None else float('inf') for m in cluster), 
                                         min(m['filename'] for m in cluster)) 
                )

                # Assign _parent_global_cluster_id if not already assigned (e.g., for singletons or groups without features)
                # This part is now handled in the Boltzmann calculation setup, ensuring IDs are unique and consistent.
                # We just need to ensure the _parent_global_cluster_id is present for all members here.
                for initial_prop_cluster in initial_clusters_list_sorted_by_energy:
                    if initial_prop_cluster and initial_prop_cluster[0].get('_parent_global_cluster_id') is None:
                        # This case should ideally not happen if the Boltzmann setup is done correctly
                        # but as a fallback, assign a new ID.
                        parent_id = pseudo_global_cluster_id_counter 
                        for member_conf in initial_prop_cluster:
                            member_conf['_parent_global_cluster_id'] = parent_id 
                        pseudo_global_cluster_id_counter += 1 


                if rmsd_threshold is not None:
                    print(f"  Performing first RMSD validation for H-bond group {hbond_count}...")
                    
                    validated_main_clusters, individual_outliers_from_first_pass = \
                        post_process_clusters_with_rmsd(initial_clusters_list_sorted_by_energy, rmsd_threshold)
                    
                    current_hbond_group_clusters_for_final_output.extend(validated_main_clusters)
                    total_rmsd_outliers_first_pass += len(individual_outliers_from_first_pass)

                    if individual_outliers_from_first_pass:
                        print(f"    Attempting second RMSD clustering on {len(individual_outliers_from_first_pass)} outliers from first pass (H-bonds {hbond_count})...")
                        
                        outliers_grouped_by_parent_global_cluster = {}
                        for outlier_conf in individual_outliers_from_first_pass:
                            parent_global_id = outlier_conf.get('_parent_global_cluster_id')
                            if parent_global_id is not None:
                                outliers_grouped_by_parent_global_cluster.setdefault(parent_global_id, []).append(outlier_conf)

                        for parent_global_id_for_outlier_group, outlier_group in outliers_grouped_by_parent_global_cluster.items():
                            if len(outlier_group) > 1:
                                print(f"      Re-clustering {len(outlier_group)} outliers from original Cluster {parent_global_id_for_outlier_group}...")
                                second_level_clusters = perform_second_rmsd_clustering(outlier_group, rmsd_threshold)
                                current_hbond_group_clusters_for_final_output.extend(second_level_clusters)
                            else:
                                single_member_processed = perform_second_rmsd_clustering(outlier_group, rmsd_threshold)
                                current_hbond_group_clusters_for_final_output.extend(single_member_processed)
                else:
                    for cluster in initial_clusters_list_sorted_by_energy:
                        for member in cluster:
                            member['_rmsd_pass_origin'] = 'first_pass_validated'
                    current_hbond_group_clusters_for_final_output.extend(initial_clusters_list_sorted_by_energy)

        current_hbond_group_clusters_for_final_output.sort(key=lambda cluster: (min(m.get('gibbs_free_energy') if m.get('gibbs_free_energy') is not None else float('inf') for m in cluster),
                                                                                  min(m['filename'] for m in cluster))) 

        # Add clusters from this hydrogen bond group to the global collection
        all_final_clusters.extend(current_hbond_group_clusters_for_final_output)

        summary_file_content_lines.append(f"Number of clusters: {len(current_hbond_group_clusters_for_final_output)}\n\n")

        for members_data in current_hbond_group_clusters_for_final_output:
            current_global_cluster_id = cluster_global_id_counter 

            summary_line_prefix = f"Cluster {current_global_cluster_id} ({len(members_data)} configurations)"

            if rmsd_threshold is not None and members_data[0].get('_rmsd_pass_origin') == 'second_pass_formed':
                parent_global_cluster_id_for_tag = members_data[0].get('_parent_global_cluster_id')

                if len(members_data) == 1:
                    summary_line_prefix += f" | RMSD Validated from Cluster {parent_global_cluster_id_for_tag}"
                else: 
                    summary_line_prefix += f" | RMSD Validated from Cluster {parent_global_cluster_id_for_tag}"

            summary_file_content_lines.append(summary_line_prefix + ":")
            summary_file_content_lines.append("Files:")
            for m_data in members_data:
                gibbs_str = f"{m_data['gibbs_free_energy']:.6f}" if m_data['gibbs_free_energy'] is not None else "N/A"
                summary_file_content_lines.append(f"  - {m_data['filename']} (Gibbs Energy: {gibbs_str} Hartree)")
            summary_file_content_lines.append("\n")

            print(f"\n{summary_line_prefix}:")
            for m_data in members_data:
                print(f"  - {m_data['filename']}")

            cluster_name_prefix = "" 
            num_configurations_in_cluster = len(members_data)

            if num_configurations_in_cluster == 1:
                cluster_name_prefix = f"cluster_{current_global_cluster_id}"
            else:
                cluster_name_prefix = f"cluster_{current_global_cluster_id}_{num_configurations_in_cluster}"

            write_cluster_dat_file(cluster_name_prefix, members_data, output_base_dir, rmsd_threshold, 
                                   hbond_count_for_original_cluster=hbond_count, weights=weights)

            cluster_xyz_subfolder = os.path.join(extracted_clusters_folder, cluster_name_prefix)
            os.makedirs(cluster_xyz_subfolder, exist_ok=True)
            print(f"  Saving .xyz files to '{cluster_xyz_subfolder}'")

            for m_data in members_data:
                xyz_filename = os.path.join(cluster_xyz_subfolder, os.path.splitext(m_data['filename'])[0] + ".xyz")
                write_xyz_file(m_data, xyz_filename) 
            
            combine_xyz_files(members_data, cluster_xyz_subfolder, output_base_name=cluster_name_prefix)

            total_clusters_outputted += 1 
            cluster_global_id_counter += 1

        previous_hbond_group_processed = True 


    for i, line in enumerate(summary_file_content_lines):
        if "<TOTAL_CLUSTERS_PLACEHOLDER>" in line:
            summary_file_content_lines[i] = line.replace("<TOTAL_CLUSTERS_PLACEHOLDER>", str(total_clusters_outputted))
        if "<TOTAL_RMSD_OUTLIERS_PLACEHOLDER>" in line:
            summary_file_content_lines[i] = line.replace("<TOTAL_RMSD_OUTLIERS_PLACEHOLDER>", str(total_rmsd_outliers_first_pass))

    # Add comparison-specific details at the very end if in comparison mode
    if is_compare_mode:
        comparison_specific_summary_lines.append("\n" + "=" * 75 + "\n")
        comparison_specific_summary_lines.append("Comparison Parameters:\n")
        if weights:
            comparison_specific_summary_lines.append("  Applied Feature Weights:")
            for key, value in weights.items():
                comparison_specific_summary_lines.append(f"    - {key}: {value}")
        if abs_tolerances:
            comparison_specific_summary_lines.append("  Applied Absolute Tolerances:")
            for key, value in abs_tolerances.items():
                # Format the float to a fixed number of decimal places to avoid scientific notation
                # Adjust precision as needed, e.g., for 1e-5 use 5 decimal places, for 0.5 use 1 decimal place
                # A general approach is to find the number of decimal places or use a high fixed number.
                # For simplicity and to cover common cases, I'll use a fixed high precision like 7.
                formatted_value = f"{value:.7f}".rstrip('0').rstrip('.') if '.' in f"{value:.7f}" else f"{int(value)}"
                comparison_specific_summary_lines.append(f"    - {key}: {formatted_value}")
        summary_file_content_lines.extend(comparison_specific_summary_lines)

    # --- Append Boltzmann Population Analysis to clustering_summary.txt ---
    if global_min_gibbs_energy is not None:
        summary_file_content_lines.append("\n" + "=" * 90 + "\n\n")
        summary_file_content_lines.append("Boltzmann Population Analysis:\n")
        summary_file_content_lines.append(f"  Reference: {os.path.splitext(global_min_rep_filename)[0]} from cluster_{global_min_cluster_id}\n")
        summary_file_content_lines.append(f"  Reference Energy (Emin): {global_min_gibbs_energy:.6f} Hartree\n")
        summary_file_content_lines.append(f"  Temperature (T): {DEFAULT_TEMPERATURE_K:.2f} K\n\n")
        
        summary_file_content_lines.append("Population by Energy Minimum (assuming non-degeneracy, gi = 1):\n")
        # Sort by population percentage descending for better readability
        sorted_g1_data = sorted(boltzmann_g1_data.items(), key=lambda item: item[1]['population'], reverse=True)
        for cluster_id, data in sorted_g1_data:
            summary_file_content_lines.append(f"    cluster_{cluster_id}")
            summary_file_content_lines.append(f"    Lowest Energy ({os.path.splitext(data['filename'])[0]}): {data['energy']:.6f} Hartree")
            summary_file_content_lines.append(f"    Population: {data['population']:.2f} %\n")

        summary_file_content_lines.append("\n" + "-" * 75 + "\n\n")

        summary_file_content_lines.append("Population by Ensemble Size (assuming gi = number of found configurations):\n")
        # Sort by population percentage descending
        sorted_g_deg_data = sorted(boltzmann_g_deg_data.items(), key=lambda item: item[1]['population'], reverse=True)
        for cluster_id, data in sorted_g_deg_data:
            summary_file_content_lines.append(f"    cluster_{cluster_id}")
            summary_file_content_lines.append(f"    Lowest Energy ({os.path.splitext(data['filename'])[0]}): {data['energy']:.6f} Hartree")
            summary_file_content_lines.append(f"    Number of structures = {data['cluster_size']}")
            summary_file_content_lines.append(f"    Population: {data['population']:.2f} %\n")

        summary_file_content_lines.append("\n" + "=" * 90 + "\n")
    # --- End Boltzmann Population Analysis ---

    # Create motifs folder with representative structures from each cluster
    create_unique_motifs_folder(all_final_clusters, output_base_dir)

    summary_file = os.path.join(output_base_dir, "clustering_summary.txt")
    with open(summary_file, "w", newline='\n') as f:
        f.write("\n".join(summary_file_content_lines))

    print(f"\nClustering summary saved to '{os.path.basename(summary_file)}' in '{output_base_dir}'")

### Main block ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process quantum chemistry log files for clustering and analysis.")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Similarity threshold for clustering (e.g., 0.999 for very similar).")
    parser.add_argument("--rmsd", type=float, nargs='?', const=1.0, default=None, 
                        help="Enable RMSD post-processing and set the RMSD validation threshold in Angstroms (default: 1.0 Å if flag is present without a value). If not provided at all, RMSD validation is skipped.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Specify the base directory for all output folders (dendrograms, extracted data, clusters). Defaults to the current working directory.")
    parser.add_argument("--reprocess-files", action="store_true",
                        help="Force re-extraction of data from log files, ignoring any existing cache.")
    parser.add_argument("--compare", nargs='+', 
                        help="Compare multiple specific log/out files (e.g., --compare file1.log file2.log file3.log). Minimum 2 files required.")
    parser.add_argument("--weights", type=str, default="", 
                        help="Specify feature weights as a string, e.g., '(electronic_energy=0.1)(homo_lumo_gap=0.2)'.")
    parser.add_argument("--min-std-threshold", type=float, default=1e-6, 
                        help="Minimum standard deviation for a feature to be scaled. Features with std dev below this are treated as constant (0.0). Default is 1e-6.")
    parser.add_argument("--abs-tolerance", type=str, default="",
                        help="Specify absolute tolerances for features as a string, e.g., '(electronic_energy=1e-5)(dipole_moment=1e-3)'. Features with max difference below tolerance are zeroed out for scaling.")
    parser.add_argument("--motif", type=float, default=None,
                        help="Threshold for final motif clustering (excluding H-bond count). If not specified, uses the same value as --threshold.")


    args = parser.parse_args()
    clustering_threshold = args.threshold
    rmsd_validation_threshold = args.rmsd 
    output_directory = args.output_dir
    force_reprocess_cache = args.reprocess_files
    weights_dict = parse_weights_argument(args.weights)
    min_std_threshold_val = args.min_std_threshold 
    abs_tolerances_dict = parse_abs_tolerance_argument(args.abs_tolerance)
    motif_threshold = args.motif if args.motif is not None else clustering_threshold

    # Set default absolute tolerances if not provided via command line
    if not abs_tolerances_dict:
        abs_tolerances_dict = {
            "electronic_energy": 5e-6,  # Tighter, was 1e-6
            "gibbs_free_energy": 5e-6,  # Tighter, was 1e-6
            "homo_energy": 3e-4,        # Increased from 1e-4
            "lumo_energy": 2e-4,        # Keep as is
            "homo_lumo_gap": 3e-4,      # Increased from 1e-4
            "dipole_moment": 1.5e-3,    # Keep as is
            "radius_of_gyration": 1.5e-4, # Keep as is
            "rotational_constants_A": 7e-5, # Keep as is
            "rotational_constants_B": 3.5e-4, # Keep as is
            "rotational_constants_C": 3e-4, # Keep as is
            "first_vib_freq": 1e-2,     # Keep as is
            "last_vib_freq": 0.3,      # Keep as is
            "average_hbond_distance": 1e-3, # Keep as is
            "average_hbond_angle": 1e-2   # Keep as is
        }

    current_dir = os.getcwd()
    
    if args.compare:
        if len(args.compare) < 2:
            print("Error: --compare requires at least 2 files.")
            exit(1)
        
        compare_files = args.compare
        
        # Check that all files exist
        for file_path in compare_files:
            if not os.path.exists(file_path):
                print(f"Error: File not found: {file_path}")
                exit(1)
        
        # Determine file extensions and check compatibility
        extensions = [os.path.splitext(f)[1].lower() for f in compare_files]
        unique_extensions = set(extensions)
        
        if len(unique_extensions) > 1:
            print(f"Warning: Comparing files with different extensions ({', '.join(unique_extensions)}). Proceeding, but ensure they are compatible.")
        
        # Use the extension of the first file for pattern
        file_extension_pattern_for_compare = extensions[0] if extensions[0] in ['.log', '.out'] else None
        if not file_extension_pattern_for_compare:
            print("Error: Provided files do not have .log or .out extensions.")
            exit(1)

        file_names = [os.path.basename(f) for f in compare_files]
        print(f"\n--- Comparing {len(compare_files)} files: {', '.join(file_names)} ---\n")
        perform_clustering_and_analysis(
            input_source=compare_files,
            threshold=clustering_threshold,
            file_extension_pattern=file_extension_pattern_for_compare, # Pass for consistency, though not used for glob
            rmsd_threshold=rmsd_validation_threshold,
            output_base_dir=output_directory,
            force_reprocess_cache=True, # Always reprocess for comparison
            weights=weights_dict,
            is_compare_mode=True,
            min_std_threshold=min_std_threshold_val,
            abs_tolerances=abs_tolerances_dict, # Pass the new argument
            motif_threshold=motif_threshold
        )
        print(f"\n--- Finished comparing {len(compare_files)} files: {', '.join(file_names)} ---\n")

    else: # Normal mode (folder processing)
        all_potential_folders = [current_dir] + [d for d in glob.glob(os.path.join(current_dir, '*')) if os.path.isdir(d)]
        
        folders_with_log_files = []
        folders_with_out_files = []

        for folder in all_potential_folders:
            has_log = bool(glob.glob(os.path.join(folder, "*.log")))
            has_out = bool(glob.glob(os.path.join(folder, "*.out")))
            
            if has_log:
                folders_with_log_files.append(folder)
            if has_out:
                folders_with_out_files.append(folder)

        all_valid_folders_to_display = sorted(list(set(folders_with_log_files + folders_with_out_files)))

        if not all_valid_folders_to_display:
            print("No subdirectories containing .log or .out files found, or files are organized directly in the current directory.")
            exit(0)

        print("\nFound the following folder(s) containing quantum chemistry log/out files:\n")
        for i, folder in enumerate(all_valid_folders_to_display):
            display_name = os.path.basename(folder)
            if folder == current_dir:
                display_name = "./"
            
            folder_types_present = []
            if folder in folders_with_log_files: folder_types_present.append(".log")
            if folder in folders_with_out_files: folder_types_present.append(".out")
            
            print(f"  [{i+1}] {display_name} (Contains: {', '.join(folder_types_present)})")

        selected_folders = []
        while True:
            choice = input("\nEnter the number of the folder to process, or type 'a' to process all: ").strip().lower()
            
            if choice == 'a':
                selected_folders = all_valid_folders_to_display
                break
            try:
                folder_index = int(choice) - 1
                if 0 <= folder_index < len(all_valid_folders_to_display):
                    selected_folders = [all_valid_folders_to_display[folder_index]]
                    break
                else:
                    print("\nInvalid number. Please enter a valid number from the list.")
            except ValueError:
                print("\nInvalid input. Please enter a number or 'a'.")

        selected_set_has_log = False
        selected_set_has_out = False
        for folder_path in selected_folders:
            if folder_path in folders_with_log_files:
                selected_set_has_log = True
            if folder_path in folders_with_out_files:
                selected_set_has_out = True
            if selected_set_has_log and selected_set_has_out:
                break

        file_extension_pattern = None 
        if selected_set_has_log and selected_set_has_out:
            while file_extension_pattern is None:
                type_choice = input("\nBoth .log and .out files are present in the selected folder(s).\nWhich file type would you like to process?\n  [1] .log files\n  [2] .out files\n  Enter your choice (1 or 2): ").strip()
                if type_choice == '1':
                    file_extension_pattern = "*.log"
                elif type_choice == '2':
                    file_extension_pattern = "*.out"
                else:
                    print("Invalid choice. Please enter '1' or '2'.")
        elif selected_set_has_log:
            file_extension_pattern = "*.log" 
            print("\nOnly .log files found in the selected folder(s). Processing .log files.")
        elif selected_set_has_out:
            file_extension_pattern = "*.out" 
            print("\nOnly .out files found in the selected folder(s). Processing .out files.")
        else:
            print("\nNo .log or .out files found in the selected folder(s) that match available types. Exiting.")
            exit(0)

        print(f"\nProcessing {len(selected_folders)} folder(s) for files matching '{file_extension_pattern}'...")
        for folder_path in selected_folders:
            display_name = os.path.basename(folder_path)
            if folder_path == current_dir:
                display_name = "./"
            print(f"\n--- Processing folder: {display_name} ---\n")

            perform_clustering_and_analysis(folder_path, clustering_threshold, file_extension_pattern, rmsd_validation_threshold, output_directory, force_reprocess_cache, weights_dict, is_compare_mode=False, min_std_threshold=min_std_threshold_val, abs_tolerances=abs_tolerances_dict, motif_threshold=motif_threshold)

            print(f"\n--- Finished processing folder: {display_name} ---\n")

    print("\nAll selected molecular analyses complete!\n")
