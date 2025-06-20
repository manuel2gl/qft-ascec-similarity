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
    periodic_table_symbols = [
        "X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Ni", "Cu", "Zn", "Ga",
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
        return (max_val - min_val) * 100.0 if max_val != min_val else 0.0
    
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
    Detects hydrogen bonds based on static distance criteria (1.2 - 2.7 A)
    and calculates the D-H...A angle and D-H covalent distance for useful information.
    The angle is calculated but does not filter the detected bonds.
    """
    try:
        coords = np.asarray(atomcoords)

        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError(f"Expected atom coords to be (N, 3), but got shape {coords.shape}")

        # Define potential donor (D) and acceptor (A) atoms by atomic number (N, O, F)
        potential_donor_acceptor_z = {7, 8, 9}
        hydrogen_atom_num = 1 # Atomic number for Hydrogen (H)

        # Static H...A distance criteria for hydrogen bond detection
        HB_min_dist_actual = 1.2
        HB_max_dist_actual = 2.7
        
        # A loose upper limit for initial search to find the *covalent* D-H donor (static, non-filtering)
        COVALENT_DH_SEARCH_DIST = 1.5 

        symbols = [atomic_number_to_symbol(n) for n in atomnos] # Assuming atomic_number_to_symbol is defined
        atom_labels = [f"{sym}{i+1}" for i, sym in enumerate(symbols)]
        hbonds = []

        # First Pass: Identify the covalently bonded donor (D) for each hydrogen (H)
        # This information is needed for D-H...A angle calculation and for output.
        # This pass does NOT filter H-bonds, it just finds the covalent partner.
        h_covalent_donors = {} # Stores {h_idx: (donor_idx, D-H_distance)}
        for i_h, h_atom_num in enumerate(atomnos):
            if h_atom_num != hydrogen_atom_num:
                continue

            coord_h = coords[i_h]
            
            min_dist_dh = float('inf')
            donor_idx_for_h = -1
            
            for i_d, d_atom_num in enumerate(atomnos):
                if i_d == i_h:
                    continue # Skip self

                if d_atom_num in potential_donor_acceptor_z: # Check if it's a potential donor atom
                    dist_dh = np.linalg.norm(coords[i_d] - coord_h)
                    
                    if dist_dh < min_dist_dh: # Look for the *nearest* potential donor
                        min_dist_dh = dist_dh
                        donor_idx_for_h = i_d
            
            # If a plausible covalent donor was found, store it
            if donor_idx_for_h != -1 and min_dist_dh <= COVALENT_DH_SEARCH_DIST:
                h_covalent_donors[i_h] = (donor_idx_for_h, min_dist_dh)

        # Second Pass: Detect H-bonds using static distance criteria
        for i_h, h_atom_num in enumerate(atomnos):
            if h_atom_num != hydrogen_atom_num:
                continue # Only process Hydrogen atoms

            # Skip if this hydrogen does not have a detected covalent donor
            if i_h not in h_covalent_donors:
                continue

            donor_idx, actual_dh_covalent_distance = h_covalent_donors[i_h]
            coord_h = coords[i_h]
            coord_d = coords[donor_idx]

            # Look for potential acceptor atoms (A)
            for i_a, a_atom_num in enumerate(atomnos):
                if i_a == i_h or i_a == donor_idx:
                    continue # Skip H and D themselves

                if a_atom_num in potential_donor_acceptor_z: # Check if it's a potential acceptor atom
                    coord_a = coords[i_a]

                    # Check H...A distance using static criteria
                    dist_ha = np.linalg.norm(coord_h - coord_a)
                    
                    if HB_min_dist_actual <= dist_ha <= HB_max_dist_actual:
                        # Calculate D-H...A angle (linearity criterion)
                        vec_h_d = coord_d - coord_h
                        vec_h_a = coord_a - coord_h
                        
                        norm_vec_h_d = np.linalg.norm(vec_h_d)
                        norm_vec_h_a = np.linalg.norm(vec_h_a)

                        # Handle potential division by zero
                        if norm_vec_h_d == 0 or norm_vec_h_a == 0:
                            angle_deg = 0.0 # Or some other indicator like NaN
                        else:
                            dot_product = np.dot(vec_h_d, vec_h_a)
                            cos_angle = dot_product / (norm_vec_h_d * norm_vec_h_a)
                            cos_angle = np.clip(cos_angle, -1.0, 1.0) # Clip to avoid floating point errors
                            angle_rad = np.arccos(cos_angle)
                            angle_deg = np.degrees(angle_rad)

                        # Add the bond if it meets distance criteria, regardless of angle
                        hbonds.append({
                            'donor_atom_label': atom_labels[donor_idx],
                            'hydrogen_atom_label': atom_labels[i_h],
                            'acceptor_atom_label': atom_labels[i_a],
                            'H...A_distance': dist_ha,
                            'D-H...A_angle': angle_deg,
                            'D-H_covalent_distance': actual_dh_covalent_distance
                        })
        return hbonds

    except Exception as e:
        print(f"  DEBUG: Error in detect_hydrogen_bonds: {e}")
        return []

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
        'num_hydrogen_bonds': 0,
        'hbond_details': [],
        'average_hbond_distance': None,
        'min_hbond_distance': None,
        'max_hbond_distance': None,
        'std_hbond_distance': None,
        'average_hbond_angle': None,
        'min_hbond_angle': None,
        'max_hbond_angle': None,
        'std_hbond_angle': None,
        '_has_freq_calc': False # New internal flag
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
                    # Once we encounter the 'done' message going backwards,
                    # any energies *before* it are relevant. We just need the *last* one.
                    # We continue to let the loop find the last relevant energy before this.
                    continue

                if "FINAL SINGLE POINT ENERGY" in line and not optimization_done_found_in_reverse:
                    try:
                        temp_final_electronic_energy = float(line.split()[-1])
                        break # Found the last valid one before the optimization done message
                    except (ValueError, IndexError):
                        pass
                elif "Electronic energy" in line and "..." in line and not optimization_done_found_in_reverse:
                    if temp_final_electronic_energy is None: # Only consider if FINAL SINGLE POINT ENERGY wasn't found yet
                        try:
                            temp_final_electronic_energy = float(line.split()[-2])
                            break # Found the last valid one before the optimization done message
                        except (ValueError, IndexError):
                            pass
            extracted_props['final_electronic_energy'] = temp_final_electronic_energy
        elif file_extension == '.log':
            # Original .log file parsing (SCF Done)
            for line in lines:
                if "SCF Done" in line:
                    parts = line.strip().split()
                    if "=" in parts:
                        idx = parts.index("=")
                        try:
                            extracted_props['final_electronic_energy'] = float(parts[idx + 1])
                        except (ValueError, IndexError):
                            pass
                        break # Stop after finding the first 'SCF Done' energy, typically the final one


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
            for line in lines:
                if "Sum of electronic and thermal Free Energies=" in line:
                    try:
                        extracted_props['gibbs_free_energy'] = float(line.strip().split()[-1])
                    except ValueError:
                        pass
                    break # Stop after finding the first Free Energy

        # --- Conditional ENTROPY EXTRACTION based on file type ---
        if file_extension == '.out':
            # ORCA .out specific parsing for Entropy
            for line in lines:
                if "Total Entropy" in line:
                    try:
                        # Assuming format like "Total Entropy                     ...        92.052840 Eh"
                        extracted_props['entropy'] = float(line.split()[-2])
                        break
                    except (ValueError, IndexError):
                        pass
        elif file_extension == '.log':
            # Gaussian .log specific parsing for Entropy (usually part of thermochemistry)
            for line in lines:
                if "Total Entropy" in line: # Common pattern for Gaussian
                    try:
                        # Extract the numeric value assuming it's the second to last element
                        extracted_props['entropy'] = float(line.split()[-2])
                        break
                    except (ValueError, IndexError):
                        pass

        # --- Conditional DIPOLE MOMENT EXTRACTION based on file type ---
        if file_extension == '.out':
            # ORCA .out specific parsing (regex)
            # This regex covers the previous ORCA output for dipole moment.
            # If semiempirical output is different, this will need another regex.
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
                                extracted_props['dipole_moment'] = np.linalg.norm([x, y, z])
                            except (ValueError, IndexError):
                                extracted_props['dipole_moment'] = None
                            break
                    if extracted_props['dipole_moment'] is not None:
                        break
        # --- END Conditional DIPOLE MOMENT EXTRACTION ---


        # Extract HOMO/LUMO energies and gap using cclib ---
        # First, try cclib. If it fails, custom parse the HOMO-LUMO gap from ORCA .out summary.
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
                        print(f"  WARNING: HOMO/LUMO indices out of bounds or invalid moenergies conformation for {os.path.basename(logfile_path)}")
                else:
                    print(f"  WARNING: 'homos' array or 'moenergies[0]' is empty/invalid for {os.path.basename(logfile_path)}")
            except Exception as e:
                print(f"  ERROR: Problem extracting HOMO/LUMO with cclib for {os.path.basename(logfile_path)}: {e}. Trying custom parse for gap.")
        else:
            print(f"  WARNING: Missing 'homos' or 'moenergies' attributes for {os.path.basename(logfile_path)}")

        # Custom parsing for HOMO-LUMO Gap if cclib fails for .out files (e.g., semiempirical)
        if extracted_props['homo_lumo_gap'] is None and file_extension == '.out':
            homo_lumo_gap_re = re.compile(r":: HOMO-LUMO gap\s*(-?\d+\.\d+)\s*eV\s*::")
            for line in lines:
                match = homo_lumo_gap_re.search(line)
                if match:
                    try:
                        extracted_props['homo_lumo_gap'] = float(match.group(1))
                        # If gap is found this way, homo/lumo energies might still be None
                        # or would need separate parsing if available
                        break
                    except ValueError:
                        pass

        # --- Conditional ROTATIONAL CONSTANTS EXTRACTION based on file type ---
        # First, try cclib for all file types
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
                    hbonds = detect_hydrogen_bonds(extracted_props['final_geometry_atomnos'], extracted_props['final_geometry_coords'])
                    extracted_props['num_hydrogen_bonds'] = len(hbonds)
                    extracted_props['hbond_details'] = hbonds

                    # Initialize all H-bond related stats to None (as default if no bonds found)
                    avg_hbond_dist, min_hbond_dist, max_hbond_dist, std_hbond_dist = [None]*4
                    avg_hbond_angle, min_hbond_angle, max_hbond_angle, std_hbond_angle = [None]*4

                    if hbonds: # Only calculate if hydrogen bonds were found
                        distances = [bond['H...A_distance'] for bond in hbonds]
                        angles = [bond['D-H...A_angle'] for bond in hbonds]
                        
                        avg_hbond_dist = np.mean(distances)
                        min_hbond_dist = np.min(distances)
                        max_hbond_dist = np.max(distances)
                        # np.std needs at least 2 elements, otherwise it's 0.0 for a single bond
                        std_hbond_dist = np.std(distances) if len(distances) > 1 else 0.0

                        avg_hbond_angle = np.mean(angles)
                        min_hbond_angle = np.min(angles)
                        max_hbond_angle = np.max(angles)
                        std_hbond_angle = np.std(angles) if len(angles) > 1 else 0.0

                    # Store all calculated H-bond stats in extracted_props
                    extracted_props['average_hbond_distance'] = avg_hbond_dist
                    extracted_props['min_hbond_distance'] = min_hbond_dist
                    extracted_props['max_hbond_distance'] = max_hbond_dist
                    extracted_props['std_hbond_distance'] = std_hbond_dist

                    extracted_props['average_hbond_angle'] = avg_hbond_angle
                    extracted_props['min_hbond_angle'] = min_hbond_angle
                    extracted_props['max_hbond_angle'] = max_hbond_angle
                    extracted_props['std_hbond_angle'] = std_hbond_angle

                except Exception as e:
                    print(f"  ERROR: Problem detecting hydrogen bonds for {os.path.basename(logfile_path)}: {e}")
            else:
                print(f"  WARNING: Skipping Hydrogen Bond Analysis for {os.path.basename(logfile_path)} due to empty atomnos or coords.")

        return extracted_props

    except Exception as e:
        print(f"  GENERIC_ERROR: Failed to extract properties from {os.path.basename(logfile_path)} (after cclib parse): {e}")
        return None

def calculate_deviation_percentage(values):
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

    # Use max_val for denominator to avoid division by zero and handle cases where min_val is zero
    # If all values are the same and non-zero, deviation is 0.
    if max_val == min_val:
        return 0.0

    # Calculate percentage deviation relative to the minimum value or mean.
    # Using min_val as reference. If min_val is 0, this might be problematic.
    # A robust approach could be relative to mean or max_val if min_val can be 0.
    # For energies, values are often negative, so min_val might be the most negative.
    # Let's use the absolute range relative to the absolute average, or max-min / min if min != 0.
    
    # For properties like energy, a common way to express spread is (max - min) / abs(min) or (max - min) / abs(mean).
    # Let's use (max - min) / abs(mean) for a more general percentage deviation.
    # If all values are positive, this is just range / mean.
    # If values are negative, abs(mean) makes sense.
    
    mean_val = np.mean(numeric_values)
    if mean_val == 0.0: # Avoid division by zero if mean is zero
        return (max_val - min_val) * 100.0 if max_val != min_val else 0.0
    
    return ((max_val - min_val) / abs(mean_val)) * 100.0


def write_cluster_dat_file(cluster_id, cluster_members_data, output_base_dir):
    """
    Writes a combined .dat file for all members of a cluster, including a comparison section.
    """
    num_conformations = len(cluster_members_data)
    
    cluster_file_name_prefix = f"cluster_{cluster_id}"
    if num_conformations > 1:
        cluster_file_name_prefix += f"_{num_conformations}"

    dat_output_dir = os.path.join(output_base_dir, "extracted_data")
    os.makedirs(dat_output_dir, exist_ok=True)

    output_filename = os.path.join(dat_output_dir, f"{cluster_file_name_prefix}.dat")

    with open(output_filename, 'w', newline='\n') as f:
        # --- Start Comparison Section (ONLY for clusters with >1 conformation) ---
        if num_conformations > 1:
            f.write("=" * 50 + "\n\n")
            f.write(f"Cluster {cluster_id} ({num_conformations} conformations)\n\n")
            
            for mol_data in cluster_members_data:
                f.write(f"    - {mol_data['filename']}\n")
            f.write("\n")

            # Gather all values for each descriptor to calculate deviations
            all_electronic_energies = [d['final_electronic_energy'] for d in cluster_members_data if d['final_electronic_energy'] is not None]
            all_gibbs_energies = [d['gibbs_free_energy'] for d in cluster_members_data if d['gibbs_free_energy'] is not None]
            all_entropy = [d['entropy'] for d in cluster_members_data if d['entropy'] is not None]
            all_homo_energies = [d['homo_energy'] for d in cluster_members_data if d['homo_energy'] is not None]
            all_lumo_energies = [d['lumo_energy'] for d in cluster_members_data if d['lumo_energy'] is not None]
            all_homo_lumo_gaps = [d['homo_lumo_gap'] for d in cluster_members_data if d['homo_lumo_gap'] is not None]
            all_dipole_moments = [d['dipole_moment'] for d in cluster_members_data if d['dipole_moment'] is not None]
            all_rg = [d['radius_of_gyration'] for d in cluster_members_data if d['radius_of_gyration'] is not None]
            
            all_rot_const_A = [d['rotational_constants'][0] for d in cluster_members_data if d['rotational_constants'] is not None and isinstance(d['rotational_constants'], np.ndarray) and len(d['rotational_constants']) == 3]
            all_rot_const_B = [d['rotational_constants'][1] for d in cluster_members_data if d['rotational_constants'] is not None and isinstance(d['rotational_constants'], np.ndarray) and len(d['rotational_constants']) == 3]
            all_rot_const_C = [d['rotational_constants'][2] for d in cluster_members_data if d['rotational_constants'] is not None and isinstance(d['rotational_constants'], np.ndarray) and len(d['rotational_constants']) == 3]

            all_first_vib_freqs = [d['first_vib_freq'] for d in cluster_members_data if d['first_vib_freq'] is not None]
            all_last_vib_freqs = [d['last_vib_freq'] for d in cluster_members_data if d['last_vib_freq'] is not None]
            all_num_hbonds = [d['num_hydrogen_bonds'] for d in cluster_members_data if d['num_hydrogen_bonds'] is not None]

            # Individual values for comparison
            # These sections will print a header and then individual values for each molecule
            # They already handle `None` values by simply not printing that specific line
            f.write("Electronic conformation descriptors:\n")
            for mol_data in cluster_members_data:
                f.write(f"    {mol_data['filename']}:\n")
                if mol_data['final_electronic_energy'] is not None:
                    f.write(f"        Final Electronic Energy (Hartree): {mol_data['final_electronic_energy']:.6f}\n")
                if mol_data['gibbs_free_energy'] is not None:
                    f.write(f"        Gibbs Free Energy (Hartree): {mol_data['gibbs_free_energy']:.6f}\n")
                if mol_data['entropy'] is not None:
                    f.write(f"        Entropy (J/(mol·K) or a.u.): {mol_data['entropy']:.6f}\n")
                if mol_data['homo_energy'] is not None:
                    f.write(f"        HOMO Energy (eV): {mol_data['homo_energy']:.6f}\n")
                if mol_data['lumo_energy'] is not None:
                    f.write(f"        LUMO Energy (eV): {mol_data['lumo_energy']:.6f}\n")
                if mol_data['homo_lumo_gap'] is not None:
                    f.write(f"        HOMO-LUMO Gap (eV): {mol_data['homo_lumo_gap']:.6f}\n")
            f.write("\n")

            f.write("Molecular conformation descriptors:\n")
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

            # This section prints if _has_freq_calc is True for at least one member
            if any(d.get('_has_freq_calc', False) for d in cluster_members_data):
                f.write("Vibrational frequency summary:\n")
                for mol_data in cluster_members_data:
                    f.write(f"    {mol_data['filename']}:\n")
                    if mol_data.get('_has_freq_calc', False): # Only print if freq calc was done
                        f.write(f"        Number of imaginary frequencies: {mol_data.get('num_imaginary_freqs', 'N/A')}\n")
                        if mol_data['first_vib_freq'] is not None:
                            f.write(f"        First Vibrational Frequency (cm^-1): {mol_data['first_vib_freq']:.2f}\n")
                        if mol_data['last_vib_freq'] is not None:
                            f.write(f"        Last Vibrational Frequency (cm^-1): {mol_data['last_vib_freq']:.2f}\n")
                    else:
                        f.write("        (Frequency calculation not performed)\n")
                f.write("\n")
            
            # This section prints if at least one molecule has H-bond details
            if any(d.get('num_hydrogen_bonds', 0) > 0 for d in cluster_members_data):
                f.write("Hydrogen bond analysis:\n")
                f.write("Criterion: H...A distance between 1.2 Å and 2.7 Å, with H covalently bonded to a donor (O, N, F).\n")
                for mol_data in cluster_members_data:
                    f.write(f"    {mol_data['filename']}:\n")
                    if mol_data.get('num_hydrogen_bonds', 0) > 0:
                        f.write(f"        Number of hydrogen bonds found: {mol_data['num_hydrogen_bonds']}\n")
                        if mol_data.get('hbond_details'):
                            for hbond in mol_data['hbond_details']:
                                f.write(f"            HB: {hbond['donor_atom_label']}-{hbond['hydrogen_atom_label']}...{hbond['acceptor_atom_label']} (Dist: {hbond['H...A_distance']:.3f} Å, D-H: {hbond['D-H_covalent_distance']:.3f} Å, Angle: {hbond['D-H...A_angle']:.2f}°)\n")
                    else:
                        f.write("        No hydrogen bonds detected based on the criterion.\n")
                f.write("\n")


            # Total Deviations Section
            f.write("Total Deviations (relative to mean):\n")
            if all_electronic_energies:
                f.write(f"    Final Electronic Energy: {calculate_deviation_percentage(all_electronic_energies):.2f} %\n")
            if all_gibbs_energies:
                f.write(f"    Gibbs Free Energy: {calculate_deviation_percentage(all_gibbs_energies):.2f} %\n")
            if all_entropy:
                f.write(f"    Entropy: {calculate_deviation_percentage(all_entropy):.2f} %\n")
            if all_homo_energies:
                f.write(f"    HOMO Energy: {calculate_deviation_percentage(all_homo_energies):.2f} %\n")
            if all_lumo_energies:
                f.write(f"    LUMO Energy: {calculate_deviation_percentage(all_lumo_energies):.2f} %\n")
            if all_homo_lumo_gaps:
                f.write(f"    HOMO-LUMO Gap: {calculate_deviation_percentage(all_homo_lumo_gaps):.2f} %\n")
            if all_dipole_moments:
                f.write(f"    Dipole Moment: {calculate_deviation_percentage(all_dipole_moments):.2f} %\n")
            if all_rot_const_A and all_rot_const_B and all_rot_const_C:
                f.write(f"    Rotational Constants (A): {calculate_deviation_percentage(all_rot_const_A):.2f} %\n")
                f.write(f"    Rotational Constants (B): {calculate_deviation_percentage(all_rot_const_B):.2f} %\n")
                f.write(f"    Rotational Constants (C): {calculate_deviation_percentage(all_rot_const_C):.2f} %\n")
            if all_rg:
                f.write(f"    Radius of Gyration: {calculate_deviation_percentage(all_rg):.2f} %\n")
            if all_first_vib_freqs:
                f.write(f"    First Vibrational Frequency: {calculate_deviation_percentage(all_first_vib_freqs):.2f} %\n")
            if all_last_vib_freqs:
                f.write(f"    Last Vibrational Frequency: {calculate_deviation_percentage(all_last_vib_freqs):.2f} %\n")
            if all_num_hbonds:
                if len(set(all_num_hbonds)) > 1:
                    f.write(f"    Number of Hydrogen Bonds: {calculate_deviation_percentage(all_num_hbonds):.2f} %\n")
                else:
                    f.write(f"    Number of Hydrogen Bonds: {all_num_hbonds[0]} (All conformations have same number)\n")
            f.write("\n")
            f.write("=" * 50 + "\n\n")
        # --- End Comparison Section ---

        # --- Original Per-conformation Info Section ---
        for i, mol_data in enumerate(cluster_members_data):
            if i > 0:
                f.write("\n" + "=" * 50 + "\n\n") # Separator between conformations

            f.write(f"Processed file: {mol_data['filename']}\n")
            f.write(f"Method: {mol_data['method']}\n")
            f.write(f"Functional: {mol_data['functional']}\n")
            f.write(f"Basis Set: {mol_data['basis_set']}\n")
            f.write(f"Charge: {mol_data['charge']}\n")
            f.write(f"Multiplicity: {mol_data['multiplicity']}\n")
            f.write(f"Number of atoms: {mol_data['num_atoms']}\n")
            f.write("Final Geometry:\n")
            if mol_data['final_geometry_atomnos'] is not None and mol_data['final_geometry_coords'] is not None and len(mol_data['final_geometry_atomnos']) > 0:
                for atom_num, coord in zip(mol_data['final_geometry_atomnos'], mol_data['final_geometry_coords']):
                    symbol = atomic_number_to_symbol(atom_num)
                    f.write(f"{symbol:2s}  {coord[0]:10.6f}  {coord[1]:10.6f}  {coord[2]:10.6f}\n")
            else:
                f.write("Geometry data not available.\n")

            # --- Electronic conformation descriptors ---
            electronic_descriptors_present = (
                mol_data.get('final_electronic_energy') is not None or
                mol_data.get('gibbs_free_energy') is not None or
                mol_data.get('entropy') is not None or # Check for entropy
                mol_data.get('homo_energy') is not None or
                mol_data.get('lumo_energy') is not None or
                mol_data.get('homo_lumo_gap') is not None
            )

            if electronic_descriptors_present:
                f.write("\nElectronic conformation descriptors:\n")
                if mol_data.get('final_electronic_energy') is not None:
                    f.write(f"Final Electronic Energy (Hartree): {mol_data['final_electronic_energy']:.6f}\n")
                if mol_data.get('gibbs_free_energy') is not None:
                    f.write(f"Gibbs Free Energy (Hartree): {mol_data['gibbs_free_energy']:.6f}\n")
                if mol_data.get('entropy') is not None:
                    f.write(f"Entropy (J/(mol·K) or a.u.): {mol_data['entropy']:.6f}\n")
                if mol_data.get('homo_energy') is not None:
                    f.write(f"HOMO Energy (eV): {mol_data['homo_energy']:.6f}\n")
                if mol_data.get('lumo_energy') is not None:
                    f.write(f"LUMO Energy (eV): {mol_data['lumo_energy']:.6f}\n")
                if mol_data.get('homo_lumo_gap') is not None:
                    f.write(f"HOMO-LUMO Gap (eV): {mol_data['homo_lumo_gap']:.6f}\n")

            # --- Molecular conformation descriptors ---
            molecular_descriptors_present = (
                mol_data.get('dipole_moment') is not None or
                (mol_data.get('rotational_constants') is not None and isinstance(mol_data['rotational_constants'], np.ndarray) and len(mol_data['rotational_constants']) == 3) or
                mol_data.get('radius_of_gyration') is not None
            )

            if molecular_descriptors_present:
                f.write("\nMolecular conformation descriptors:\n")
                if mol_data.get('dipole_moment') is not None:
                    f.write(f"Dipole Moment (Debye): {mol_data['dipole_moment']:.6f}\n")
                
                rc = mol_data.get('rotational_constants')
                if rc is not None and isinstance(rc, np.ndarray) and rc.ndim == 1 and len(rc) == 3:
                    f.write(f"Rotational Constants (GHz): {rc[0]:.6f}, {rc[1]:.6f}, {rc[2]:.6f}\n")
                
                if mol_data.get('radius_of_gyration') is not None:
                    f.write(f"Radius of Gyration (Å): {mol_data['radius_of_gyration']:.6f}\n")
            
            # --- Vibrational frequency summary ---
            # ONLY print this section if a frequency calculation was performed (_has_freq_calc is True)
            if mol_data.get('_has_freq_calc', False):
                f.write("\nVibrational frequency summary:\n")
                # num_imaginary_freqs will be an int (0 or more) if _has_freq_calc is True
                f.write(f"Number of imaginary frequencies: {mol_data.get('num_imaginary_freqs', 'N/A')}\n")
                if mol_data.get('first_vib_freq') is not None:
                    f.write(f"First Vibrational Frequency (cm^-1): {mol_data['first_vib_freq']:.2f}\n")
                if mol_data.get('last_vib_freq') is not None:
                    f.write(f"Last Vibrational Frequency (cm^-1): {mol_data['last_vib_freq']:.2f}\n")
            
            # --- Hydrogen bond analysis ---
            # ONLY print this section if hydrogen bonds are present
            if mol_data.get('num_hydrogen_bonds', 0) > 0:
                f.write("\nHydrogen bond analysis:\n")
                f.write("Criterion: H...A distance between 1.2 Å and 2.7 Å, with H covalently bonded to a donor (O, N, F).\n")
                f.write(f"Number of hydrogen bonds found: {mol_data['num_hydrogen_bonds']}\n")
                if mol_data.get('hbond_details'):
                    for hbond in mol_data['hbond_details']:
                        f.write(f"Hydrogen bond: {hbond['donor_atom_label']}-{hbond['hydrogen_atom_label']}...{hbond['acceptor_atom_label']}  Distance: {hbond['H...A_distance']:.3f} Å, D-H: {hbond['D-H_covalent_distance']:.3f} Å, Angle: {hbond['D-H...A_angle']:.2f}°\n")

    print(f"Wrote combined data for Cluster {cluster_id} to '{os.path.basename(output_filename)}'")

def write_xyz_file(atomnos, atomcoords, filename):
    """
    Writes atomic coordinates to an XYZ file.
    The second line will be the base filename (without extension).
    """
    if atomnos is None or atomcoords is None or len(atomnos) == 0:
        print(f"  WARNING: Cannot write XYZ for {os.path.basename(filename)}: Missing geometry data.")
        return

    base_name = os.path.splitext(os.path.basename(filename))[0]

    symbols = [atomic_number_to_symbol(n) for n in atomnos]

    with open(filename, 'w', newline='\n') as f:
        f.write(f"{len(atomnos)}\n")
        f.write(f"{base_name}\n")
        for i in range(len(atomnos)):
            f.write(f"{symbols[i]:<2} {atomcoords[i][0]:10.6f} {atomcoords[i][1]:10.6f} {atomcoords[i][2]:10.6f}\n")

# New functions for combining XYZ files
def get_sort_key(filename):
    """
    Extracts the first number from the filename for sorting.
    Used to sort XYZ files in numerical order.
    """
    match = re.search(r'\d+', filename)
    return int(match.group(0)) if match else 0

def combine_xyz_files(input_dir, output_filename="combined_cluster.xyz", exclude_pattern="_trj.xyz", openbabel_alias="obabel"):
    """
    Combines all relevant .xyz files (excluding trajectory files and the
    combined output file itself) found in the current directory and its
    subdirectories into a single .xyz file, processing them in order based
    on the first configuration number found in the filename.
    Optionally converts the combined .xyz to a .mol file using Open Babel.

    Args:
        input_dir (str): The directory to search for .xyz files.
        output_filename (str): The name of the output .xyz file.
                               Defaults to "combined_cluster.xyz".
        exclude_pattern (str): The pattern to look for before the .xyz
                                 extension to exclude the file.
                                 Defaults to "_trj.xyz".
        openbabel_alias (str): The command-line alias for Open Babel (e.g., "babel", "obabel").
                               Defaults to "babel".
    """
    all_xyz_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            filepath = os.path.join(root, file)
            if filepath.endswith(".xyz") and exclude_pattern not in file and file != output_filename:
                all_xyz_files.append(filepath)

    if not all_xyz_files:
        return

    sorted_xyz_files = sorted(all_xyz_files, key=lambda x: get_sort_key(os.path.basename(x)))

    full_output_path = os.path.join(input_dir, output_filename)
    with open(full_output_path, "w", newline='\n') as outfile:
        for xyz_file in sorted_xyz_files:
            with open(xyz_file, "r") as infile:
                lines = infile.readlines()
                outfile.writelines(lines)

    print(f"  Successfully combined {len(sorted_xyz_files)} .xyz files into: '{os.path.basename(full_output_path)}'")

    # Section for Open Babel Integration
    mol_output_filename = os.path.splitext(output_filename)[0] + ".mol"
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
            conversion_command = [openbabel_full_path, "-i", "xyz", full_output_path, "-o", "mol", "-O", full_mol_output_path]
            # We use capture_output=True and check=True to handle errors gracefully,
            # but we don't print stdout/stderr unless there's an error.
            subprocess.run(conversion_command, check=True, capture_output=True, text=True)

            if os.path.exists(full_mol_output_path):
                print(f"  Successfully converted '{os.path.basename(full_output_path)}' to '{os.path.basename(full_mol_output_path)}' using Open Babel.")

        except subprocess.CalledProcessError as e:
            print(f"  Open Babel conversion failed for '{os.path.basename(full_output_path)}'.")
            print(f"  Error details: {e.stderr.strip()}")
        except Exception as e:
            print(f"  An unexpected error occurred during Open Babel conversion for '{os.path.basename(full_output_path)}': {e}")

def perform_clustering_and_analysis(input_folder, threshold=1.0, file_extension_pattern="*.log"):
    """
    Performs hierarchical clustering and analysis on the extracted molecular properties,
    and saves .dat and .xyz files for each cluster.
    """
    # Use the provided file_extension_pattern for glob.glob
    files_to_process = glob.glob(os.path.join(input_folder, file_extension_pattern))
    
    # Check if any files matching the pattern were found
    if not files_to_process:
        print(f"No files matching '{file_extension_pattern}' found in '{input_folder}'. Skipping this folder.")
        return

    print(f"Starting data extraction from {len(files_to_process)} files matching '{file_extension_pattern}' in '{input_folder}'...")
    all_extracted_data = []
    for file_path in sorted(files_to_process):
        print(f"  Extracting from: {os.path.basename(file_path)}...")
        # Use extract_properties_from_logfile which uses ccread
        extracted_props = extract_properties_from_logfile(file_path)
        if extracted_props:
            all_extracted_data.append(extracted_props)

    if not all_extracted_data:
        print(f"No data was successfully extracted from files matching '{file_extension_pattern}' in '{input_folder}'. Skipping clustering for this folder.")
        return

    print("\nData extraction complete. Proceeding to clustering.")
    
    clean_data_for_clustering = []
    # Essential features for basic processing and grouping. If these are missing, the file is skipped.
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
        print(f"No complete data entries to cluster after filtering in '{input_folder}'. Exiting clustering step for this folder.")
        return

    hbond_groups = {}
    for item in clean_data_for_clustering:
        hbond_groups.setdefault(item['num_hydrogen_bonds'], []).append(item)

    total_conformations_processed = len(all_extracted_data)
    total_clusters_count = 0
    
    summary_lines = []
    summary_lines.append("=" * 75 + "\n")
    summary_lines.append(f"Clustering Summary for folder: {os.path.basename(input_folder)}")
    summary_lines.append(f"Similarity threshold (distance): {threshold}")
    summary_lines.append(f"Number of conformations processed: {total_conformations_processed}")
    summary_lines.append("\n" + "=" * 75 + "\n")

    dendrogram_images_folder = os.path.join(input_folder, "dendrogram_images")
    os.makedirs(dendrogram_images_folder, exist_ok=True)
    print(f"Dendrogram images will be saved to '{dendrogram_images_folder}'")

    extracted_data_folder = os.path.join(input_folder, "extracted_data")
    extracted_clusters_folder = os.path.join(input_folder, "extracted_clusters")
    os.makedirs(extracted_data_folder, exist_ok=True)
    os.makedirs(extracted_clusters_folder, exist_ok=True)

    cluster_global_id_counter = 1

    prev_hbond_count = None

    for hbond_count, group_data in sorted(hbond_groups.items()):
        
        if prev_hbond_count is not None:
            summary_lines.append("\n" + "=" * 75 + "\n")

        summary_lines.append(f"Hydrogen bonds: {hbond_count}\n")
        
        # Define all potential numerical features to consider for dynamic selection
        all_potential_numerical_features = [
            'radius_of_gyration',
            'dipole_moment',
            'homo_lumo_gap',
            'first_vib_freq',
            'last_vib_freq',
            'average_hbond_distance', 
            'average_hbond_angle'
        ]
        # Rotational constants are treated separately due to their conformation
        rotational_constant_subfeatures = [
            'rotational_constants_0', 'rotational_constants_1', 'rotational_constants_2'
        ]

        # Determine which features are globally missing (missing for ALL members in this group)
        globally_missing_for_group = []
        for feature in all_potential_numerical_features:
            if all(d.get(feature) is None for d in group_data):
                globally_missing_for_group.append(feature)
        
        # Check rotational constants separately for global missingness
        is_rot_const_globally_missing_for_group = True
        for d in group_data:
            rot_consts = d.get('rotational_constants')
            # Check if rotational_constants exist and have the expected format for at least one molecule
            if rot_consts is not None and isinstance(rot_consts, np.ndarray) and rot_consts.ndim == 1 and len(rot_consts) == 3:
                is_rot_const_globally_missing_for_group = False
                break
        
        if is_rot_const_globally_missing_for_group:
            globally_missing_for_group.extend(rotational_constant_subfeatures)
            print(f"  Note: Rotational constants are globally missing or invalid for H-bond group {hbond_count}. Excluding them from clustering features.")

        # Construct the list of active features for this group's clustering
        # These are features that are NOT globally missing for this H-bond group
        active_numerical_features_for_group = [f for f in all_potential_numerical_features if f not in globally_missing_for_group]
        
        # Prepare data for scaling using only active features for this group
        features_for_scaling = []
        for d in group_data:
            mol_feature_vector = []
            for feature_name in active_numerical_features_for_group:
                # Use the value if present, else 0.0 (imputation for individual missing values)
                mol_feature_vector.append(d.get(feature_name) if d.get(feature_name) is not None else 0.0)
            
            if not is_rot_const_globally_missing_for_group:
                rot_consts = d.get('rotational_constants')
                if rot_consts is not None and isinstance(rot_consts, np.ndarray) and rot_consts.ndim == 1 and len(rot_consts) == 3:
                    mol_feature_vector.extend(rot_consts.tolist())
                else:
                    # If not globally missing but individually missing/invalid, use 0.0 for all three components
                    mol_feature_vector.extend([0.0, 0.0, 0.0])
            
            features_for_scaling.append(mol_feature_vector)
        
    # Check if there are enough conformations and features to perform meaningful clustering
        if len(group_data) < 2 or not features_for_scaling or len(features_for_scaling[0]) == 0:
            print(f"\nSkipping detailed clustering for H-bond group {hbond_count}: Less than 2 conformations or no valid numerical features left after filtering. Treating each as a single-conformation cluster.")
            summary_lines.append(f"Conformations: {len(group_data)}")
            summary_lines.append(f"Number of clusters: {len(group_data)}\n\n")
            for single_mol_data in group_data:
                summary_lines.append(f"Cluster {cluster_global_id_counter} (1 conformation):")
                summary_lines.append("Files:")
                gibbs_str = f"{single_mol_data['gibbs_free_energy']:.6f}" if single_mol_data['gibbs_free_energy'] is not None else "N/A"
                summary_lines.append(f"  - {single_mol_data['filename']} (Gibbs Energy: {gibbs_str} Hartree)")
                summary_lines.append("\n") 

                print(f"\nCluster {cluster_global_id_counter} (1 conformation):")
                print(f"  - {single_mol_data['filename']}")

                write_cluster_dat_file(cluster_global_id_counter, [single_mol_data], input_folder)

                cluster_xyz_subfolder_name = f"cluster_{cluster_global_id_counter}"
                cluster_xyz_subfolder = os.path.join(extracted_clusters_folder, cluster_xyz_subfolder_name)
                os.makedirs(cluster_xyz_subfolder, exist_ok=True)
                print(f"  Saving .xyz files to '{cluster_xyz_subfolder}'")

                xyz_filename = os.path.join(cluster_xyz_subfolder, os.path.splitext(single_mol_data['filename'])[0] + ".xyz")
                write_xyz_file(single_mol_data['final_geometry_atomnos'], single_mol_data['final_geometry_coords'], xyz_filename)
                
                total_clusters_count += 1
                cluster_global_id_counter += 1
            
            prev_hbond_count = hbond_count
            continue # Skip to next hbond group
        
        # End of dynamic feature selection and fallback handling

        filenames_base = [os.path.splitext(item['filename'])[0] for item in group_data]
        
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_for_scaling)

        linkage_matrix = linkage(features_scaled, method='average', metric='euclidean')

        plt.figure(figsize=(12, 8))
        dendrogram(linkage_matrix, labels=filenames_base, leaf_rotation=90, leaf_font_size=8)
        plt.title(f"Hierarchical Clustering Dendrogram (H-bonds = {hbond_count}) - {os.path.basename(input_folder)}")
        plt.ylabel("Euclidean Distance")
        plt.tight_layout()
        dendrogram_filename = os.path.join(dendrogram_images_folder, f"dendrogram_H{hbond_count}.png")
        plt.savefig(dendrogram_filename)
        plt.close()
        print(f"Dendrogram saved as '{os.path.basename(dendrogram_filename)}'")

        cluster_labels = fcluster(linkage_matrix, t=threshold, criterion='distance')

        clusters_data = {}
        for i, label in enumerate(cluster_labels):
            clusters_data.setdefault(label, []).append(group_data[i])

        # Print and append Conformations and Number of clusters for this H-bond group
        summary_lines.append(f"Conformations: {len(group_data)}")
        summary_lines.append(f"Number of clusters: {len(clusters_data)}\n\n")

        sorted_cluster_labels = sorted(clusters_data.keys())

        for cluster_id_local in sorted_cluster_labels:
            members_data = clusters_data[cluster_id_local]
            
            gibbs_energies_in_cluster = [m['gibbs_free_energy'] for m in members_data if m['gibbs_free_energy'] is not None]
            deviation_gibbs = calculate_deviation_percentage(gibbs_energies_in_cluster)
            
            summary_lines.append(f"Cluster {cluster_global_id_counter} ({len(members_data)} conformations):")
            summary_lines.append("Files:")
            for m_data in members_data:
                gibbs_str = f"{m_data['gibbs_free_energy']:.6f}" if m_data['gibbs_free_energy'] is not None else "N/A"
                summary_lines.append(f"  - {m_data['filename']} (Gibbs Energy: {gibbs_str} Hartree)")
            
            summary_lines.append("\n")

            print(f"\nCluster {cluster_global_id_counter} ({len(members_data)} conformations):")
            for m_data in members_data:
                print(f"  - {m_data['filename']}")

            write_cluster_dat_file(cluster_global_id_counter, members_data, input_folder)

            cluster_xyz_subfolder_name = f"cluster_{cluster_global_id_counter}"
            if len(members_data) > 1:
                cluster_xyz_subfolder_name += f"_{len(members_data)}"
            
            cluster_xyz_subfolder = os.path.join(extracted_clusters_folder, cluster_xyz_subfolder_name)
            os.makedirs(cluster_xyz_subfolder, exist_ok=True)
            print(f"  Saving .xyz files to '{cluster_xyz_subfolder}'")

            xyz_files_for_combining = []
            for m_data in members_data:
                xyz_filename = os.path.join(cluster_xyz_subfolder, os.path.splitext(m_data['filename'])[0] + ".xyz")
                write_xyz_file(m_data['final_geometry_atomnos'], m_data['final_geometry_coords'], xyz_filename)
                xyz_files_for_combining.append(xyz_filename)

            if len(members_data) > 1:
                combine_xyz_files(cluster_xyz_subfolder, output_filename=f"combined_cluster_{cluster_global_id_counter}_{len(members_data)}.xyz")
            
            total_clusters_count += 1
            cluster_global_id_counter += 1
        
        prev_hbond_count = hbond_count

    final_summary_output = []
    initial_info_added = False
    for line in summary_lines:
        if not initial_info_added and "Number of conformations processed:" in line:
            final_summary_output.append(line)
            final_summary_output.append(f"Total number of clusters: {total_clusters_count}")
            initial_info_added = True
        else:
            final_summary_output.append(line)

    summary_file = os.path.join(input_folder, "clustering_summary.txt")
    with open(summary_file, "w", newline='\n') as f:
        f.write("\n".join(final_summary_output))

    print(f"\nClustering summary saved to '{os.path.basename(summary_file)}'")

### Main block ###
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process quantum chemistry log files for clustering and analysis.")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Similarity threshold for clustering (e.g., 0.999 for very similar).")
    args = parser.parse_args()
    clustering_threshold = args.threshold

    current_dir = os.getcwd()
    
    # Get all subdirectories and the current directory to scan for log/out files
    all_potential_folders = [current_dir] + [d for d in glob.glob(os.path.join(current_dir, '*')) if os.path.isdir(d)]
    
    # Categorize folders based on the file types they contain
    folders_with_log_files = []
    folders_with_out_files = []

    for folder in all_potential_folders:
        has_log = bool(glob.glob(os.path.join(folder, "*.log")))
        has_out = bool(glob.glob(os.path.join(folder, "*.out")))
        
        if has_log:
            folders_with_log_files.append(folder)
        if has_out:
            folders_with_out_files.append(folder)

    # Create a unique list of all folders that contain either .log or .out files
    all_valid_folders_to_display = sorted(list(set(folders_with_log_files + folders_with_out_files)))

    if not all_valid_folders_to_display:
        print("No subdirectories containing .log or .out files found, or files are organized directly in the current directory.")
        exit(0)

    print("\nFound the following folder(s) containing quantum chemistry log/out files:\n")
    for i, folder in enumerate(all_valid_folders_to_display):
        display_name = os.path.basename(folder)
        if folder == current_dir:
            display_name = "./"
        
        # Indicate what kind of files are in each folder for the user
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

    # Determine if both .log and .out types are present across the *selected* folders in aggregate
    selected_set_has_log = False
    selected_set_has_out = False
    for folder_path in selected_folders:
        if folder_path in folders_with_log_files:
            selected_set_has_log = True
        if folder_path in folders_with_out_files:
            selected_set_has_out = True
        if selected_set_has_log and selected_set_has_out: # Optimization: if both found, no need to check further
            break

    # This section determines which files to process based on user choice/detection
    file_extension_to_process_pattern = None
    if selected_set_has_log and selected_set_has_out:
        # Prompt user if both types are found across the selected folders
        while file_extension_to_process_pattern is None:
            type_choice = input("\nBoth .log and .out files are present in the selected folder(s).\nWhich file type would you like to process?\n  [1] .log files\n  [2] .out files\n  Enter your choice (1 or 2): ").strip()
            if type_choice == '1':
                file_extension_to_process_pattern = "*.log"
            elif type_choice == '2':
                file_extension_to_process_pattern = "*.out"
            else:
                print("Invalid choice. Please enter '1' or '2'.")
    elif selected_set_has_log:
        file_extension_to_process_pattern = "*.log"
        print("\nOnly .log files found in the selected folder(s). Processing .log files.")
    elif selected_set_has_out:
        file_extension_to_process_pattern = "*.out"
        print("\nOnly .out files found in the selected folder(s). Processing .out files.")
    else:
        print("\nNo .log or .out files found in the selected folder(s) that match available types. Exiting.")
        exit(0)

    print(f"\nProcessing {len(selected_folders)} folder(s) for files matching '{file_extension_to_process_pattern}'...")
    for folder_path in selected_folders:
        display_name = os.path.basename(folder_path)
        if folder_path == current_dir:
            display_name = "./"
        print(f"\n--- Processing folder: {display_name} ---\n")

        # Pass the chosen file extension pattern to the function
        perform_clustering_and_analysis(folder_path, clustering_threshold, file_extension_to_process_pattern)

        print(f"\n--- Finished processing folder: {display_name} ---\n")

    print("\nAll selected molecular analyses complete!\n")