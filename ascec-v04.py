# ascec-v04
#!/usr/bin/env python3

import argparse
from collections import Counter
import dataclasses
import math
import numpy as np
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple 

# Global Constants
MAX_ATOM = 1000 # Increase this if your systems are larger than 1000 atoms
MAX_MOLE = 100  # Increase this if you have more than 100 molecules
B2 = 3.166811563e-6   # Boltzmann constant in Hartree/K

# Constants for overlap prevention
OVERLAP_SCALE_FACTOR = 0.7 # Factor to make overlap check slightly more lenient (e.g., allow partial overlap)
MAX_OVERLAP_PLACEMENT_ATTEMPTS = 10000 # Max attempts to place a single molecule without significant overlap

# Set this to True to create a SEPARATE COPY of the XYZ file
# (e.g., mtobox_seed.xyz) which will include 8 dummy 'X' atoms for
# visualizing the box in programs like GaussView.
# If False, only the original XYZ file (mto_seed.xyz) will be generated.
CREATE_BOX_XYZ_COPY = True 
#CREATE_BOX_XYZ_COPY = False 

# Symbol for dummy atoms used to mark box corners.
# WARNING: 'X' is not a standard element.
DUMMY_ATOM_SYMBOL = "X"

# Atomic radii - Used in some initial configurations for overlap prevention
# These are typical Covalent Radii (single bond, in Angstroms).
# If your Fortran Ratom array uses different values or a specific parameterized set,
# please ensure consistency with that source.
R_ATOM = {
    # Period 1
    1: 0.31,   # H (Hydrogen) - Your previous was 0.373
    # 2: He (Helium) - Noble gases generally do not have covalent radii.
    #    If present in your system and requiring steric interaction, Van der Waals radii would be used for them.

    # Period 2
    3: 1.34,   # Li (Lithium)
    4: 1.02,   # Be (Beryllium)
    5: 0.82,   # B (Boron)
    6: 0.77,   # C (Carbon) - Your previous was 0.772
    7: 0.75,   # N (Nitrogen) - Your previous was 0.710
    8: 0.73,   # O (Oxygen) - Your previous was 0.604
    9: 0.71,   # F (Fluorine)
    # 10: Ne (Neon) - No covalent radius

    # Period 3
    11: 1.54,  # Na (Sodium)
    12: 1.30,  # Mg (Magnesium)
    13: 1.18,  # Al (Aluminum)
    14: 1.11,  # Si (Silicon)
    15: 1.07,  # P (Phosphorus)
    16: 1.05,  # S (Sulfur)
    17: 1.02,  # Cl (Chlorine)
    # 18: Ar (Argon) - No covalent radius

    # Period 4
    19: 1.96,  # K (Potassium)
    20: 1.76,  # Ca (Calcium)
    # Transition metals: Covalent radii can vary significantly based on oxidation state and coordination.
    # The following are typical single bond values.
    30: 1.22,  # Zn (Zinc)
    31: 1.22,  # Ga (Gallium)
    32: 1.20,  # Ge (Germanium)
    33: 1.21,  # As (Arsenic)
    34: 1.20,  # Se (Selenium)
    35: 1.20,  # Br (Bromine)
    # 36: Kr (Krypton) - No covalent radius

    # Period 5
    53: 1.39,  # I (Iodine)
    # 54: Xe (Xenon) - No covalent radius
}

# Element Symbol to Atomic Number Mapping
# This dictionary will be used to convert element symbols from the input to atomic numbers.
ELEMENT_SYMBOLS = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109, "Ds": 110,
    "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114, "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118
}

# Atomic Weights for elements (Global Constant)
ATOMIC_WEIGHTS = {
    1: 1.008,   2: 4.0026,   3: 6.94,    4: 9.012,   5: 10.81,
    6: 12.011,  7: 14.007,   8: 15.999,  9: 18.998, 10: 20.180,
    11: 22.990, 12: 24.305,  13: 26.982, 14: 28.085, 15: 30.974,
    16: 32.06,  17: 35.45,   18: 39.948, 19: 39.098, 20: 40.078,
    21: 44.956, 22: 47.867,  23: 50.942, 24: 51.996, 25: 54.938,
    26: 55.845, 27: 58.933,  28: 58.693, 29: 63.546, 30: 65.38,
    31: 69.723, 32: 72.630,  33: 74.922, 34: 78.971, 35: 79.904,
    36: 83.798, 37: 85.468,  38: 87.62,  39: 88.906, 40: 91.224,
    41: 92.906, 42: 95.96,   43: 98.0,   44: 101.07, 45: 102.906,
    46: 106.42, 47: 107.868, 48: 112.414, 49: 114.818, 50: 118.710,
    51: 121.760, 52: 127.60, 53: 126.904, 54: 131.293, 55: 132.905,
    56: 137.327, 57: 138.905, 58: 140.116, 59: 140.908, 60: 144.242,
    61: 145.0,  62: 150.36,  63: 151.964, 64: 157.25, 65: 158.925,
    66: 162.500, 67: 164.930, 68: 167.259, 69: 168.934, 70: 173.04,
    71: 174.967, 72: 178.49,  73: 180.948, 74: 183.84, 75: 186.207,
    76: 190.23,  77: 192.217, 78: 195.084, 79: 196.967, 80: 200.592,
    81: 204.383, 82: 207.2,   83: 208.980, 84: 209.0,  85: 210.0,
    86: 222.0,   87: 223.0,   88: 226.0,  89: 227.0,  90: 232.038,
    91: 231.036, 92: 238.028, 93: 237.0,  94: 244.0,  95: 243.0,
    96: 247.0,   97: 247.0,   98: 251.0,  99: 252.0, 100: 257.0,
    101: 258.0, 102: 259.0, 103: 262.0, 104: 261.0, 105: 262.0,
    106: 266.0, 107: 264.0, 108: 269.0, 109: 268.0, 110: 271.0,
    111: 272.0, 112: 277.0, 113: 286.0, 114: 289.0, 115: 289.0,
    116: 293.0, 117: 294.0, 118: 294.0
}

# Electronegativity for elements
ELECTRONEGATIVITY_VALUES = {
    'H': 2.20, 'He': 0.0, # Noble gases typically not involved in bonding, using 0 for sorting
    'Li': 0.98, 'Be': 1.57, 'B': 2.04, 'C': 2.55, 'N': 3.04, 'O': 3.44, 'F': 3.98,
    'Ne': 0.0,
    'Na': 0.93, 'Mg': 1.31, 'Al': 1.61, 'Si': 1.90, 'P': 2.19, 'S': 2.58, 'Cl': 3.16,
    'Ar': 0.0,
    'K': 0.82, 'Ca': 1.00, 'Sc': 1.36, 'Ti': 1.54, 'V': 1.63, 'Cr': 1.66, 'Mn': 1.55,
    'Fe': 1.83, 'Co': 1.88, 'Ni': 1.91, 'Cu': 1.90, 'Zn': 1.65, 'Ga': 1.81, 'Ge': 2.01,
    'As': 2.18, 'Se': 2.55, 'Br': 2.96, 'Kr': 0.0,
    'Rb': 0.82, 'Sr': 0.95, 'Y': 1.22, 'Zr': 1.33, 'Nb': 1.6, 'Mo': 2.16, 'Tc': 1.9,
    'Ru': 2.2, 'Rh': 2.28, 'Pd': 2.20, 'Ag': 1.93, 'Cd': 1.69, 'In': 1.78, 'Sn': 1.96,
    'Sb': 2.05, 'Te': 2.1, 'I': 2.66, 'Xe': 0.0,
    'Cs': 0.79, 'Ba': 0.89, 'La': 1.1, 'Ce': 1.12, 'Pr': 1.13, 'Nd': 1.14, 'Pm': 1.13,
    'Sm': 1.17, 'Eu': 1.17, 'Gd': 1.2, 'Tb': 1.1, 'Dy': 1.22, 'Ho': 1.23, 'Er': 1.24,
    'Tm': 1.25, 'Yb': 1.1, 'Lu': 1.27, 'Hf': 1.3, 'Ta': 1.5, 'W': 2.36, 'Re': 1.9,
    'Os': 2.2, 'Ir': 2.2, 'Pt': 2.28, 'Au': 2.54, 'Hg': 2.00, 'Tl': 1.62, 'Pb': 1.87,
    'Bi': 2.02, 'Po': 2.0, 'At': 2.2, 'Rn': 0.0,
    'Fr': 0.7, 'Ra': 0.9, 'Ac': 1.1, 'Th': 1.3, 'Pa': 1.5, 'U': 1.38, 'Np': 1.36,
    'Pu': 1.28, 'Am': 1.13, 'Cm': 1.28, 'Bk': 1.3, 'Cf': 1.3, 'Es': 1.3, 'Fm': 1.3,
    'Md': 1.3, 'No': 1.3, 'Lr': 1.3, 'Rf': 1.3, 'Db': 1.3, 'Sg': 1.3, 'Bh': 1.3,
    'Hs': 1.3, 'Mt': 1.3, 'Ds': 1.3, 'Rg': 1.3, 'Cn': 1.3, 'Nh': 1.3, 'Fl': 1.3,
    'Mc': 1.3, 'Lv': 1.3, 'Ts': 1.3, 'Og': 1.3
}

# 2. Define SystemState Class
class SystemState:
    def __init__(self):
        # --- Configuration parameters (typically read from input file) ---
        self.random_generate_config: int = 0      # 0: Annealing; 1: Random configurations
        self.num_random_configs: int = 0          # Used if random_generate_config is 1
        self.cube_length: float = 0.0             # Simulation Cube Length (Angstroms)
        self.quenching_routine: int = 0           # 1: Linear, 2: Geometrical
        self.linear_temp_init: float = 0.0
        self.linear_temp_decrement: float = 0.0
        self.linear_num_steps: int = 0
        self.geom_temp_init: float = 0.0
        self.geom_temp_factor: float = 0.0
        self.geom_num_steps: int = 0
        self.max_cycle: int = 0                   # Maximum Monte Carlo Cycles per Temperature
        self.max_displacement_a: float = 0.0      # Maximum Displacement of each mass center (ds)
        self.max_rotation_angle_rad: float = 0.0  # Maximum Rotation angle in radians (dphi)
        self.ia: int = 0                          # QM program type (1: Gaussian, 2: ORCA, etc.)
        self.alias: str = ""                      # Program alias/executable name (e.g., "g09")
        self.hamiltonian: Optional[str] = None    # (e.g., "pm3", "hf")
        self.basis_set: Optional[str] = None      # (e.g., "6-31G*", "STO-3G")
        self.charge: int = 0                      # (iQ)
        self.spin_multiplicity: int = 0           # (iS2)
        self.qm_memory: Optional[str] = None      # memory
        self.qm_nproc: Optional[int] = None       # nprocs
        self.qm_additional_keywords: str = ""     # if necessary
        self.num_molecules: int = 0               # (nmo) from input file
        self.output_dir: str = "."                # Directory for output files
        self.ivalE: int = 0                       # Evaluate Energy flag (0: No energy evaluation, just mto movements)
        self.mto: int = 0                         # Number of random configurations if ivalE=0 (Fortran's nrandom)
        self.qm_program_name: Optional[str] = None # Will store "gaussian", "orca", etc., derived from alias
        self.molecules_to_add = []                # Initialize as an empty list
        self.all_molecule_definitions = []        # Also good to initialize this if not already
        self.openbabel_alias = "obabel"
        
        # Add this line to assign the global R_ATOM to the state object
        self.R_ATOM = R_ATOM 
        
        # Also ensure these are defined, if not already:
        self.OVERLAP_SCALE_FACTOR = 0.7
        self.MAX_OVERLAP_PLACEMENT_ATTEMPTS_PER_MOLECULE = 500000 


        # --- Random number generator state (internal to ran0_method) ---
        # This MUST be an integer. It will be updated by ran0_method.
        self.random_seed: int = -1
        self.IY: int = 0                          # Internal state for ran0_method
        self.IVEC: List[int] = [0] * 32           # Internal state for ran0_method (NTAB=32 from ran0_method)

        # --- Simulation State Variables (updated during simulation) ---
        self.nstep: int = 0                        # Total energy evaluation steps (Fortran's cycle)
        self.icount: int = 0                       # Counter for steps at current temperature (Fortran's icount)
        self.iboltz: int = 0                       # Count of configurations accepted by Boltzmann (Fortran's iboltz)
        self.current_energy: float = 0.0          # Ep (current energy of the system)
        self.lowest_energy: float = 1e30          # oldE (stores the lowest energy found, initialize to large value)
        self.current_temp: float = 0.0            # Ti (current annealing temperature)
        self.maxstep: int = 0                     # Current maximum steps at a temperature (derived from max_cycle initially)
        self.natom: int = 0                       # Total number of atoms, calculated from molecule definitions
        self.nmo: int = 0                         # Number of molecules (often kept in sync with num_molecules)
        self.jdum: str = ""                       # Timestamp/label for output files (can be populated from alias or datetime)
        self.xbox: float = 0.0                    # Length of the simulation cube (synced with cube_length)

        # --- Molecular and Atomic Data ---
        self.rp: np.ndarray = np.empty((0, 3), dtype=np.float64) # Current atomic coordinates
        self.rf: np.ndarray = np.empty((0, 3), dtype=np.float64) # Proposed atomic coordinates
        self.rcm: np.ndarray = np.empty((MAX_MOLE, 3), dtype=np.float64) # Center of mass for each molecule
        self.imolec: List[int] = []               # Indices defining molecules (which atoms belong to which molecule)
        self.iznu: List[int] = []                 # Atomic numbers for each atom in the system
        self.all_molecule_definitions: List['MoleculeData'] = [] # Stores parsed molecule data from input

        # Element Maps
        # These are preferred for storing element data.
        self.atomic_number_to_symbol: dict = {}
        self.atomic_number_to_weight: dict = {}

        # These Fortran-style arrays might be redundant if the dicts above are the primary source.
        # Keep them only if your code explicitly relies on indexed access that these provide.
        self.sym: List[str] = [''] * 120          # From Fortran's symelem (symbol for atomic number index)
        self.wt: List[float] = [0.0] * 120        # From Fortran's wtelem (weight for atomic number index)
        
        # --- Corrected: Now these are proper attributes ---
        self.qm_call_count: int = 0 # NEW: Counter for QM calculation calls
        self.initial_qm_retries: int = 40 # Number of retries for initial QM calculation, default to 20

    def ran0_method(self) -> float:
        IA = 16807
        IM = 2147483647 # 2^31 - 1
        AM = 1.0 / IM
        IQ = 127773
        IR = 2836
        NTAB = 32
        NDIV = 1 + (IM - 1) // NTAB
        EPS = 1.2e-7
        RNMX = 1.0 - EPS
        MASK = 123459876 # A common MASK from NR

        if self.random_seed == 0:
            self.random_seed = 1

        if self.random_seed < 0:
            self.random_seed = -self.random_seed
            for j in range(NTAB + 7):
                k = self.random_seed // IQ
                self.random_seed = IA * (self.random_seed - k * IQ) - IR * k
                if self.random_seed < 0:
                    self.random_seed += IM
                if j < NTAB:
                    self.IVEC[j] = self.random_seed
            self.IY = self.IVEC[0]

        k = self.random_seed // IQ
        self.random_seed = IA * (self.random_seed - k * IQ) - IR * k
        if self.random_seed < 0:
            self.random_seed += IM
        j = self.random_seed // NDIV

        self.IY = self.IVEC[j] # Use j as index directly (from NR)
        self.IVEC[j] = self.random_seed

        if self.IY > RNMX * IM: # Ensure value is within bounds
            return RNMX
        else:
            return float(self.IY) * AM


def get_molecular_formula_string(atom_symbols_list: List[str]) -> str:
    """
    Generates a simple molecular formula string (e.g., H2O, CH4)
    from a list of atom symbols. Elements are sorted by a common chemical convention
    (C, then H, then others by increasing electronegativity).
    """
    if not atom_symbols_list:
        return ""

    counts = Counter(atom_symbols_list)

    def sort_key_for_formula(symbol):
        # Prioritize Carbon and Hydrogen based on common chemical formula conventions
        if symbol == 'C':
            return (-2, 0) # Carbon gets the highest priority (lowest tuple value)
        if symbol == 'H':
            return (-1, 0) # Hydrogen gets the second highest priority

        # For all other elements, sort by electronegativity (ascending)
        # If an element is not in our dictionary, assign a very high value (float('inf'))
        # so it sorts to the end of the list.
        electronegativity = ELECTRONEGATIVITY_VALUES.get(symbol, float('inf'))
        return (0, electronegativity) # Others get lower priority, then sorted by EN

    sorted_elements = sorted(counts.keys(), key=sort_key_for_formula)

    formula_parts = []
    for symbol in sorted_elements:
        count = counts[symbol]
        formula_parts.append(symbol)
        if count > 1:
            formula_parts.append(str(count))
    return "".join(formula_parts)

def convert_xyz_to_mol(xyz_filepath: str, openbabel_alias: str = "obabel"):
    """
    Converts a single .xyz file to a .mol file using Open Babel.
    Prints messages indicating success or failure to sys.stderr.
    """
    mol_filepath = os.path.splitext(xyz_filepath)[0] + ".mol"

    openbabel_full_path = shutil.which(openbabel_alias)

    if not openbabel_full_path:
        # Redirect these messages to sys.stderr
        print(f"\n  Open Babel ({openbabel_alias}) command not found or not executable. Skipping .mol conversion for '{os.path.basename(xyz_filepath)}'.", file=sys.stderr)
        print("  Please ensure Open Babel is installed and added to your system's PATH, or provide the correct alias.", file=sys.stderr)
        return False

    try:
        conversion_command = [openbabel_full_path, "-i", "xyz", xyz_filepath, "-o", "mol", "-O", mol_filepath]
        subprocess.run(conversion_command, check=True, capture_output=True, text=True)

        if os.path.exists(mol_filepath):
            # Redirect this success message to sys.stderr
            return True
        else:
            # Redirect this failure message to sys.stderr
            print(f"  Open Babel conversion failed to create '{os.path.basename(mol_filepath)}' for '{os.path.basename(xyz_filepath)}'.", file=sys.stderr)
            return False

    except subprocess.CalledProcessError as e:
        # Redirect error messages to sys.stderr
        print(f"  Open Babel conversion failed for '{os.path.basename(xyz_filepath)}'.", file=sys.stderr)
        print(f"  Error details: {e.stderr.strip()}", file=sys.stderr)
        return False
    except Exception as e:
        # Redirect error messages to sys.stderr
        print(f"  An unexpected error occurred during Open Babel conversion for '{os.path.basename(xyz_filepath)}': {e}", file=sys.stderr)
        return False

# 2. Output write
def write_simulation_summary(state: SystemState, output_file_handle, xyz_output_file_display_name: str):
    """
    Writes a summary of the simulation parameters to the main out file.
    """
    # Construct the elemental composition string
    element_composition_lines = []
    if hasattr(state, 'element_types') and state.element_types:
        for atomic_num, count in state.element_types:
            symbol = state.atomic_number_to_symbol.get(atomic_num, f"Unk({atomic_num})")
            # Using 10 for symbol width, 10 for count to match desired formatting
            element_composition_lines.append(f"   {symbol:<3} {count:>3}")
    else:
        element_composition_lines.append("  (Elemental composition not available)")

    # Construct the molecular composition string
    molecular_composition_lines = []
    if hasattr(state, 'all_molecule_definitions') and state.all_molecule_definitions:
        if hasattr(state, 'molecules_to_add') and state.molecules_to_add:
            for i, mol_def_idx in enumerate(state.molecules_to_add):
                if mol_def_idx < len(state.all_molecule_definitions):
                    mol_def = state.all_molecule_definitions[mol_def_idx]
                    
                    # Reconstruct atom symbols in the correct order for display
                    atom_symbols_list = []
                    for atom_data in mol_def.atoms_coords:
                        atomic_num = atom_data[0]
                        atom_symbols_list.append(state.atomic_number_to_symbol.get(atomic_num, ''))
                    
                    molecular_formula = get_molecular_formula_string(atom_symbols_list)
                    
                    # Ensure label is not empty to avoid index error
                    molecule_instance_name = f"{mol_def.label[0].lower()}{i+1}" if mol_def.label else f"mol{i+1}"
                    
                    # Ensure consistent padding for the instance name (e.g., w1, w2, w10)
                    # Use a fixed width or dynamic padding. Let's aim for a similar look to your example.
                    # We can pad to 5 characters, then add some spacing to align formula.
                    molecular_composition_lines.append(f"  {molecule_instance_name:<5} {molecular_formula}")
                else:
                    molecular_composition_lines.append(f"  mol{i+1:<5} (Definition not found for index {mol_def_idx})")
        else:
            molecular_composition_lines.append("  (No molecules specified for addition or 'molecules_to_add' is empty/missing)")
    else:
        molecular_composition_lines.append("  (Molecular definitions not available)")


    # Determine the message for energy evaluation
    energy_eval_message = ""
    output_config_message = ""
    
    if state.random_generate_config == 0: # Random configuration generation mode
        energy_eval_message = "** Energy will not be evaluated **"
        output_config_message = f"Will produce {state.num_random_configs:>2} random configurations"
    elif state.random_generate_config == 1: # Annealing mode
        energy_eval_message = "** Energy will be evaluated **"
        if state.quenching_routine == 1:
            output_config_message = f"Starting annealing with {state.linear_num_steps} linear quenching steps."
        elif state.quenching_routine == 2:
            output_config_message = f"Starting annealing with {state.geom_num_steps} geometric quenching steps."

    # Write to the file handle
    print("=" * 60 + "\n", file=output_file_handle)
    print("Annealing Simulado Con Energía Cuántica (ASCEC)", file=output_file_handle)
    print("ASCEC-V04: Jun-2025", file=output_file_handle)
    print("\nElemental composition of the system:", file=output_file_handle)
    for line in element_composition_lines:
        print(line, file=output_file_handle)
    print(f"There are a total of {state.natom:>2} nuclei", file=output_file_handle)
    print(f"\nCube's length = {state.cube_length:>6.5f} A", file=output_file_handle)

    print("\nNumber of molecules:", state.num_molecules, file=output_file_handle)
    print("\nMolecular composition", file=output_file_handle)
    for line in molecular_composition_lines:
        print(line, file=output_file_handle)

    print(f"\nMaximum displacement of each mass center = {state.max_displacement_a:>6.5f} A", file=output_file_handle)
    print(f"Maximum rotation angle = {state.max_rotation_angle_rad:>6.5f} radians", file=output_file_handle)
    print(f"\nSeed = {state.random_seed:>6}", file=output_file_handle)

    print(f"\n{energy_eval_message}", file=output_file_handle)
    print(output_config_message, file=output_file_handle)

    if state.random_generate_config == 0: # Only print this line for random config mode
         print(f"\nCoordinates stored in {xyz_output_file_display_name}", file=output_file_handle)

    print("\n" + "=" * 60, file=output_file_handle)
    print("\n", file=output_file_handle) # Add an extra newline for separation

# 3. Translated symelem.f (initialize_element_symbols)
def initialize_element_symbols(state: SystemState):
    """
    Initializes a dictionary mapping atomic numbers to element symbols
    and populates the Fortran-style list state.sym.
    """
    element_map = {
        1: 'H',   2: 'He',  3: 'Li',  4: 'Be',  5: 'B',
        6: 'C',   7: 'N',   8: 'O',   9: 'F',  10: 'Ne',
        11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P',
        16: 'S',  17: 'Cl', 18: 'Ar', 19: 'K',  20: 'Ca',
        21: 'Sc', 22: 'Ti', 23: 'V',  24: 'Cr', 25: 'Mn',
        26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn',
        31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br',
        36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y',  40: 'Zr',
        41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh',
        46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn',
        51: 'Sb', 52: 'Te', 53: 'I',  54: 'Xe', 55: 'Cs',
        56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd',
        61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb',
        66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb',
        71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W',  75: 'Re',
        76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg',
        81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At',
        86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th',
        91: 'Pa', 92: 'U',  93: 'Np', 94: 'Pu', 95: 'Am',
        96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm',
        101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Db',
        106: 'Sg', 107: 'Bh', 108: 'Hs', 109: 'Mt', 110: 'Ds',
        111: 'Rg', 112: 'Cn', 113: 'Nh', 114: 'Fl', 115: 'Mc',
        116: 'Lv', 117: 'Ts', 118: 'Og'
    }
    state.atomic_number_to_symbol = element_map

    for z, symbol in element_map.items():
        if z < len(state.sym):
            state.sym[z] = symbol.strip() # Strip spaces if only one character (e.g., "H ")

# 4. Initialize element weights
def initialize_element_weights(state: SystemState):
    """
    Initializes a dictionary mapping atomic numbers to their atomic weights
    and populates the list state.wt.
    """
    # Populate the dictionary for Pythonic lookup
    # We use ATOMIC_WEIGHTS (your global constant) directly
    state.atomic_number_to_weight = ATOMIC_WEIGHTS.copy()

    # Then, populate the wt list for Fortran-like indexing
    for atomic_num, weight in ATOMIC_WEIGHTS.items():
        # Ensure the atomic number is a valid index within the state.wt list
        if 0 < atomic_num < len(state.wt):
            state.wt[atomic_num] = weight

# 6. Calculate_mass_center
def calculate_mass_center(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Calculates the center of mass for a set of atoms."""
    total_mass = np.sum(masses)
    if total_mass == 0: # Avoid division by zero
        # This might indicate an issue with mass assignment or an empty molecule
        # For a robust system, you might want to log this or raise a more specific error
        return np.zeros(3) # Return origin or handle error appropriately
    
    # Ensure masses are correctly broadcast for element-wise multiplication
    # masses[:, np.newaxis] reshapes (N,) to (N,1) for multiplication with (N,3) coords
    return np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass

# 7. Translated draw_molekel.f (draw_molekel)
def draw_molekel(natom: int, r_coords: np.ndarray, cycle: int, anneal_flag: int, energy: float, state: SystemState):
    """
    Fortran subroutine draw_molekel: Writes output files for visualization.
    This is a conceptual translation. Actual implementation needs to match Fortran's output format.
    """
    output_filename = ""
    if anneal_flag == 0: # Random generation mode (mto_ files)
        output_filename = f"mto_{state.idum}_{cycle:04d}.xyz"
    else: # Annealing mode (result_ files)
        output_filename = f"result_{state.idum}_{cycle:04d}.xyz"

    # Prepend output_dir
    full_output_path = os.path.join(state.output_dir, output_filename)

    with open(full_output_path, 'w') as f:
        f.write(f"{natom}\n")
        f.write(f"Energy: {energy:.6f}\n") # Example: adding energy to comment line
        for i in range(natom):
            atomic_num = state.iznu[i] if hasattr(state, 'iznu') and len(state.iznu) > i else 0
            symbol = state.sym[atomic_num - 1] if atomic_num > 0 else "X"
            f.write(f"{symbol} {r_coords[i,0]:10.6f} {r_coords[i,1]:10.6f} {r_coords[i,2]:10.6f}\n")


# 8. Config_molecules
def config_molecules(natom: int, nmo: int, r_coords: np.ndarray, state: SystemState):
    """
    Fortran subroutine config: Configures initial molecular positions and orientations.
    Modifies r_coords in place. Applies PBC.
    Now includes overlap prevention using R_ATOM during initial random configuration.
    """
    # This function is called by initialize_molecular_coords_in_box for initial setup.
    # The overlap prevention logic applies when generating random configurations for either
    # the random_generate_config == 0 (annealing initial config) or
    # random_generate_config == 1 (just generating random configs with no QM).

    current_atom_idx = 0
    final_rp_coords = np.zeros_like(r_coords) # Temporary array to build the configuration

    # Ensure all_molecule_definitions is populated before proceeding
    if not state.all_molecule_definitions:
        print("Error: No molecule definitions found for configuration generation. Cannot proceed.", file=sys.stderr)
        return

    # Prepare a list to store atom data with their final absolute positions as they are placed
    # This will be used for overlap checking: [(atom_idx, atomic_num, x, y, z), ...]
    placed_atoms_data = []

    for i, mol_def in enumerate(state.all_molecule_definitions):
        # Loop to find a non-overlapping placement for the current molecule
        overlap_found = True
        attempts = 0
        while overlap_found and attempts < MAX_OVERLAP_PLACEMENT_ATTEMPTS:
            attempts += 1
            
            # Generate a random translation for the molecule's center
            # Coordinates are -xbox/2 to +xbox/2
            translation = np.random.uniform(-state.xbox/2, state.xbox/2, size=3)

            # Generate three random Euler angles (yaw, pitch, roll) in radians
            # covering the full 3D rotation space.
            alpha = np.random.uniform(0, 2 * np.pi) # Rotation around Z-axis
            beta = np.random.uniform(0, 2 * np.pi)  # Rotation around Y-axis
            gamma = np.random.uniform(0, 2 * np.pi) # Rotation around X-axis

            Rx = np.array([
                [1, 0, 0],
                [0, np.cos(gamma), -np.sin(gamma)],
                [0, np.sin(gamma), np.cos(gamma)]
            ])

            Ry = np.array([
                [np.cos(beta), 0, np.sin(beta)],
                [0, 1, 0],
                [-np.sin(beta), 0, np.cos(beta)]
            ])

            Rz = np.array([
                [np.cos(alpha), -np.sin(alpha), 0],
                [np.sin(alpha), np.cos(alpha), 0],
                [0, 0, 1]
            ])

            # Combine rotations. The order Rz @ Ry @ Rx means applying Rx first, then Ry, then Rz.
            rotation_matrix = Rz @ Ry @ Rx

            # Calculate proposed atom positions for the current molecule
            proposed_mol_atoms = [] # (atomic_num, abs_coords_array, relative_index_within_molecule)
            # Corrected line: Use enumerate to get atom_rel_idx
            for atom_rel_idx, atom_data_in_mol in enumerate(mol_def.atoms_coords):
                atomic_num, x_rel, y_rel, z_rel = atom_data_in_mol
                relative_coords_vector = np.array([x_rel, y_rel, z_rel])
                rotated_coords = np.dot(rotation_matrix, relative_coords_vector)
                abs_coords = rotated_coords + translation
                # Corrected line: Append all three values
                proposed_mol_atoms.append((atomic_num, abs_coords, atom_rel_idx))

            # Check for overlaps with already placed atoms
            overlap_found = False
            for prop_atom_num, prop_coords, prop_atom_rel_idx in proposed_mol_atoms:
                for placed_atom_data in placed_atoms_data:
                    placed_atom_abs_idx = placed_atom_data[0]
                    placed_atom_num = placed_atom_data[1] 
                    placed_coords = np.array(placed_atom_data[2:]) 

                    distance = np.linalg.norm(prop_coords - placed_coords)

                    radius1 = state.R_ATOM.get(prop_atom_num, 0.5)
                    radius2 = state.R_ATOM.get(placed_atom_num, 0.5)

                    min_distance_allowed = (radius1 + radius2) * state.OVERLAP_SCALE_FACTOR

                    # Check for overlap. Add a small epsilon to avoid false positives for identical atoms at same spot
                    if distance < min_distance_allowed and distance > 1e-4:
                        overlap_found = True
                        attempts += 1 # Increment attempts here, as Fortran's icount increments on *each* detected overlap leading to a retry.

                        if attempts > 100000: # Threshold for dynamic ds adjustment
                            print(f"\n** Warning **", file=sys.stderr)
                            print(f"More than {attempts-1} movements.", file=sys.stderr)
                            # Fortran message references molecule and atom numbers
                            print(f"Molecule {i+1} keeps violating the space of molecule {state.get_molecule_label_from_atom_index(placed_atom_abs_idx)}.", file=sys.stderr)
                            print(f"Involved atoms: {state.atomic_number_to_symbol.get(prop_atom_num, 'Unk')} (relative index: {prop_atom_rel_idx}) and {state.atomic_number_to_symbol.get(placed_atom_num, 'Unk')} (absolute index: {placed_atom_abs_idx})", file=sys.stderr)
                            
                            print(f"Maximum displacement parameter will be enlarged by 20%", file=sys.stderr)
                            state.max_displacement_a *= 1.2
                            if state.max_displacement_a > state.xbox:
                                state.max_displacement_a = state.xbox
                                print(f"Maximum value for ds = xbox reached ({state.xbox:.3f})", file=sys.stderr)
                            print(f"New value for ds = {state.max_displacement_a:.3f}", file=sys.stderr)
                            print(f"*****************\n", file=sys.stderr)
                            attempts = 0 # Reset attempts after adjusting ds, to allow new tries with larger steps
                        
                        # print(f"  Overlap detected between {state.atomic_number_to_symbol.get(prop_atom_num, 'Unk')} and {state.atomic_number_to_symbol.get(placed_atom_num, 'Unk')} (dist: {distance:.3f} < min: {min_distance_allowed:.3f})", file=sys.stderr)
                        break # Found an overlap for this proposed molecule, break inner loop over placed atoms
                if overlap_found:
                    break # If overlap found for any atom in proposed molecule, restart placement for this molecule
            
        if overlap_found: # If attempts exhausted and overlap still found
            print(f"Warning: Could not find non-overlapping placement for molecule {mol_def.label} after {state.MAX_OVERLAP_PLACEMENT_ATTEMPTS_PER_MOLECULE} attempts. Placing it anyway, may cause QM errors.", file=sys.stderr)

        # If a non-overlapping position is found (or attempts exhausted), add to final_rp_coords
        for atom_data_in_mol in proposed_mol_atoms:
            # CORRECTED LINE: Unpack all three elements
            atomic_num, abs_coords, prop_atom_rel_idx = atom_data_in_mol
            final_rp_coords[current_atom_idx, :] = abs_coords
            # Add to placed_atoms_data for future overlap checks
            # Store (original_idx, atomic_num, x, y, z) for easy access
            placed_atoms_data.append((current_atom_idx, atomic_num, abs_coords[0], abs_coords[1], abs_coords[2]))
            state.iznu[current_atom_idx] = atomic_num # Ensure iznu is populated
            current_atom_idx += 1
    
    # Assign the built configuration to state.rp
    state.rp[:] = final_rp_coords
    
    # PERIODIC BOUNDARY CONDITIONS (PBC) - Keep these commented out as per Fortran behavior
    # half_xbox = state.xbox / 2.0
    # state.rp[:] = state.rp - state.xbox * np.floor((state.rp + half_xbox) / state.xbox)

# 9. Define MoleculeData Class for input parsing
class MoleculeData:
    """
    A data structure to hold information for a single molecule parsed from input.
    """
    def __init__(self, label: str, num_atoms: int, atoms_coords: List[Tuple[int, float, float, float]]):
        self.label = label
        self.num_atoms = num_atoms
        self.atoms_coords = atoms_coords

# 10. Initial configuration
def initialize_molecular_coords_in_box(state: SystemState) -> Tuple[np.ndarray, List[int]]:
    """
    Initializes the coordinates (state.rp) for all molecules by placing them
    superimposed, ensuring they are fully contained and centered within the
    simulation box, and assigning their atomic numbers (state.iznu).

    Args:
        state: The SystemState object containing molecular definitions,
               num_molecules, and box dimensions.

    Returns:
        A tuple containing:
        - np.ndarray: The initialized coordinates (rp) for all atoms.
        - List[int]: The list of atomic numbers (iznu) for all atoms.
    """
    # Validate that molecular definitions are available
    if not hasattr(state, 'coords_per_molecule') or state.coords_per_molecule is None or \
       not hasattr(state, 'atomic_numbers_per_molecule') or not state.atomic_numbers_per_molecule or \
       state.natom_per_molecule == 0:
        print("Error: Molecular definitions (coords_per_molecule, atomic_numbers_per_molecule) not loaded from input file.", file=sys.stderr)
        print("Please ensure the 'Molecule Definition' section in your input file is correct and parsed by read_input_file.", file=sys.stderr)
        raise ValueError("Cannot initialize coordinates: Molecular definition is missing or invalid.")

    # Calculate total number of atoms
    total_atoms = state.num_molecules * state.natom_per_molecule
    state.natom = total_atoms # Update state.natom to reflect the total count

    rp = np.zeros((total_atoms, 3), dtype=np.float64)
    iznu = [0] * total_atoms

    template_coords = state.coords_per_molecule
    template_atomic_numbers = state.atomic_numbers_per_molecule

    # Calculate the bounding box of the *template molecule*
    template_min_coords = np.min(template_coords, axis=0)
    template_max_coords = np.max(template_coords, axis=0)
    
    # Calculate the center of the template molecule's bounding box
    template_geometric_center = (template_min_coords + template_max_coords) / 2.0

    # Calculate the desired center for all molecules in the simulation box
    box_center = state.cube_length / 2.0 * np.ones(3) # e.g., [cube_len/2, cube_len/2, cube_len/2]

    # Calculate the overall shift needed to move the template molecule's center to the box center
    # This ensures the *initial* configuration of the superimposed molecules is centered and contained.
    initial_centering_offset = box_center - template_geometric_center

    for i in range(state.num_molecules):
        start_idx = i * state.natom_per_molecule
        end_idx = start_idx + state.natom_per_molecule

        # Apply the template coordinates and the calculated centering offset
        # This will place all superimposed molecules in the center of the simulation box.
        rp[start_idx:end_idx, :] = template_coords + initial_centering_offset

        # Assign atomic numbers
        iznu[start_idx:end_idx] = template_atomic_numbers
    
    return rp, iznu

# 11. read_input_file
def read_input_file(state: SystemState, source) -> List[MoleculeData]:
    """
    Reads and parses the input. Can read from a file path or a file-like object (like sys.stdin).
    Populates SystemState and returns a list of MoleculeData objects.
    This version uses a two-phase parsing approach for robustness and correctly handles '*' separators.
    """
    molecule_definitions: List[MoleculeData] = []
    
    lines = []
    if isinstance(source, str):
        state.input_file_path = source
        with open(source, 'r') as f_obj:
            lines = f_obj.readlines()
    else:
        state.input_file_path = "stdin_input.in" # Placeholder name for stdin
        lines = source.readlines()

    def clean_line(line: str) -> str:
        """Removes comments and leading/trailing whitespace from a line, including non-breaking spaces."""
        if '#' in line:
            line = line.split('#')[0]
        if '!' in line:
            line = line.split('!')[0]
        line = line.replace('\xa0', ' ')
        return line.strip()

    lines_iterator = iter(lines) # Use an iterator to cleanly step through lines

    # PHASE 1: Read Fixed Configuration Parameters (Lines 1-11)
    config_lines_parsed = 0 # Counter for successfully parsed config lines

    while config_lines_parsed < 12: # Expecting 12 configuration lines
        try:
            raw_line = next(lines_iterator)
        except StopIteration:
            raise EOFError(f"Unexpected end of input while reading configuration parameters. Expected 11 lines, but found only {config_lines_parsed}.")
        
        line_num = lines.index(raw_line) + 1 # For error messages
        line = clean_line(raw_line)

        if not line: # Skip empty or comment-only lines in config section
            continue

        parts = line.split()
        if not parts: # Should not happen if 'line' is not empty, but for safety
            continue

        # Parsing config lines based on count
        if config_lines_parsed == 0: # Line 1: Simulation Mode & Number of Configurations
            state.random_generate_config = int(parts[0])
            state.num_random_configs = int(parts[1])
            
            if state.random_generate_config == 0: # Random Configuration Generation Mode
                state.ivalE = 0 # Do NOT evaluate energy for simple random config generation
                state.mto = state.num_random_configs # Use num_random_configs as mto for output count
            elif state.random_generate_config == 1: # Annealing Mode
                state.ivalE = 1 # Always evaluate energy in annealing mode
                state.mto = 0   # Not applicable for annealing, or can be 1 for initial config

        elif config_lines_parsed == 1: # Line 2: Simulation Cube Length
            state.cube_length = float(parts[0])
            state.xbox = state.cube_length

        elif config_lines_parsed == 2: # Line 3: Annealing Quenching Routine
            state.quenching_routine = int(parts[0])

        elif config_lines_parsed == 3: # Line 4: Linear Quenching Parameters
            state.linear_temp_init = float(parts[0])
            state.linear_temp_decrement = float(parts[1])
            state.linear_num_steps = int(parts[2])

        elif config_lines_parsed == 4: # Line 5: Geometric Quenching Parameters
            state.geom_temp_init = float(parts[0])
            state.geom_temp_factor = float(parts[1])
            state.geom_num_steps = int(parts[2])

        elif config_lines_parsed == 5: # Line 6: Maximum Monte Carlo Cycles per T
            state.max_cycle = int(parts[0])
            state.maxstep = state.max_cycle

        elif config_lines_parsed == 6: # Line 7: Maximum Displacement & Rotation
            state.max_displacement_a = float(parts[0])
            state.max_rotation_angle_rad = float(parts[1])
            state.ds = state.max_displacement_a
            state.dphi = state.max_rotation_angle_rad

        elif config_lines_parsed == 7: # Line 8: QM Program Index & Alias (e.g., "1 g09")
            state.ia = int(parts[0])      # Store program index if needed elsewhere
            state.qm_program = parts[1]   # Store "g09" directly in qm_program
            state.alias = parts[1]        # Keep alias for output filenames
            # Removed QM_PROGRAM_DETAILS lookup as calculate_energy uses state.qm_program directly
            state.jdum = state.alias      # Keep jdum if used for legacy reasons

        elif config_lines_parsed == 8: # Line 9: Hamiltonian & Basis Set (e.g., "pm3 zdo")
            state.qm_method = parts[0]      # Store "pm3" in qm_method
            state.qm_basis_set = parts[1]   # Store "zdo" in qm_basis_set

        elif config_lines_parsed == 9:      # Line 10: nprocs & maxmemory
            state.qm_nproc = int(parts[0])   # Store 8 in qm_nproc
            state.qm_memory = parts[1]       # Store (as string) in qm_memory

        elif config_lines_parsed == 10:     # Line 11: Charge & Spin Multiplicity
            state.charge = int(parts[0])
            state.multiplicity = int(parts[1]) # Use 'multiplicity' as per calculate_energy

        elif config_lines_parsed == 11:     # Line 12: Number of Molecules
            state.num_molecules = int(parts[0])
            state.nmo = state.num_molecules
        else:
            # This 'else' should only be hit if there are more than 11 non-comment/blank lines
            # before the molecule definitions start.
            print(f"Warning: Unexpected configuration line at index {config_lines_parsed}. Line: {line}", file=sys.stderr)
            # You might want to raise an error here if strict adherence to 11 config lines is required.
            # For now, we'll continue and increment.

        config_lines_parsed += 1 # Increment only after a *valid* config line (non-empty, non-comment) is processed

    # PHASE 2: Read Molecule Definitions
    reading_molecule = False
    current_molecule_num_atoms_expected = 0
    current_molecule_label = ""
    current_molecule_atoms: List[Tuple[int, float, float, float]] = []
    atoms_read_in_current_molecule = 0

    for raw_line in lines_iterator: # Continue iterating from where config parsing left off
        line_num = lines.index(raw_line) + 1 # For error messages
        line = clean_line(raw_line)

        if not line:
            continue

        parts = line.split()

        if parts[0] == "*":
            if reading_molecule: # Found '*' and was reading a molecule -> this '*' closes the previous molecule
                if current_molecule_num_atoms_expected == atoms_read_in_current_molecule:
                    molecule_definitions.append(
                        MoleculeData(current_molecule_label, current_molecule_num_atoms_expected, current_molecule_atoms)
                    )
                else:
                    raise ValueError(f"Error parsing molecule block near line {line_num}: Expected {current_molecule_num_atoms_expected} atoms but read {atoms_read_in_current_molecule} for molecule {current_molecule_label}.")
                
                # Reset temp vars for the molecule that was just processed
                current_molecule_num_atoms_expected = 0
                current_molecule_label = ""
                current_molecule_atoms = []
                atoms_read_in_current_molecule = 0
                
                # After saving, this same '*' immediately implies the start of the next molecule.
                # So, 'reading_molecule' remains True, and we expect the next line to be num_atoms.
                reading_molecule = True # Keep in reading_molecule mode for the next molecule
                continue # Skip to next line, expecting num_atoms for new molecule

            else: # Found '*' and was NOT reading a molecule -> this must be the very first '*' opening the first molecule
                reading_molecule = True
                continue # Skip to next line, expecting num_atoms for the first molecule

        elif reading_molecule: # We are inside a molecule block (after an opening '*' and before its closing '*')
            if current_molecule_num_atoms_expected == 0: # Expecting number of atoms
                try:
                    current_molecule_num_atoms_expected = int(parts[0])
                except ValueError:
                    raise ValueError(f"Error parsing molecule block near line {line_num}: Expected number of atoms, got '{parts[0]}'.")
            elif not current_molecule_label: # Expecting molecule label
                current_molecule_label = parts[0]
            else: # Expecting atom coordinates
                if atoms_read_in_current_molecule < current_molecule_num_atoms_expected:
                    if len(parts) < 4:
                        raise ValueError(f"Error parsing atom coordinates near line {line_num}: Expected element symbol and 3 coordinates, got '{line}'.")
                    
                    symbol = parts[0]
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    
                    atomic_num = ELEMENT_SYMBOLS.get(symbol)
                    if atomic_num is None:
                        raise ValueError(f"Error: Unknown element symbol '{symbol}' near line {line_num}.")
                    
                    current_molecule_atoms.append((atomic_num, x, y, z))
                    atoms_read_in_current_molecule += 1
                else:
                    raise ValueError(f"Error parsing molecule block near line {line_num}: Read more atoms than expected for molecule {current_molecule_label}. Missing '*' delimiter?")
        else: # This block handles lines after config params but before the first '*' or between molecules and not a '*'
            # These should ideally be blank lines or comments. We can just ignore them.
            print(f"Warning: Unexpected content '{raw_line.strip()}' found outside of defined blocks near line {line_num}.", file=sys.stderr)
            continue # Simply skip any lines that are not part of config or molecule definitions
            
    # Post-loop check for unclosed molecule (e.g., missing final '*' at end of file)
    # Only append if a molecule was actually being defined (i.e., had expected atoms)
    if reading_molecule and current_molecule_num_atoms_expected > 0: # <-- CRUCIAL CHANGE HERE
        if current_molecule_num_atoms_expected == atoms_read_in_current_molecule:
            molecule_definitions.append(
                MoleculeData(current_molecule_label, current_molecule_num_atoms_expected, current_molecule_atoms)
            )
        else:
            raise ValueError(f"Error: Last molecule block not properly closed or incomplete for molecule {current_molecule_label}.")

    # Final Validation and SystemState Population
    # Validate that the number of molecules read matches the expected number
    if len(molecule_definitions) != state.num_molecules:
        raise ValueError(f"Error: Number of molecules defined in input ({len(molecule_definitions)}) does not match expected ({state.num_molecules}).")

    state.all_molecule_definitions = molecule_definitions
    # This assumes all 'num_molecules' are instances of the first defined molecule.
    # This aligns with the 'w1', 'w2', etc. output desired by the user.
    if state.num_molecules > 0 and molecule_definitions:
        # Create a list of indices, pointing to the first molecule definition (index 0)
        # repeated 'state.num_molecules' times.
        state.molecules_to_add = [0] * state.num_molecules
    else:
        # Ensure 'molecules_to_add' is always initialized, even if no molecules are expected.
        state.molecules_to_add = []

    state.natom = sum(mol.num_atoms for mol in molecule_definitions)
    
    # to correctly populate the template molecule data
    if molecule_definitions: # Ensure there's at least one molecule defined
        first_molecule_data = molecule_definitions[0]
        state.natom_per_molecule = first_molecule_data.num_atoms
        state.molecule_label = first_molecule_data.label # Store label of the template molecule

        # Convert to numpy array for coords_per_molecule
        coords_list = [[atom[1], atom[2], atom[3]] for atom in first_molecule_data.atoms_coords]
        state.coords_per_molecule = np.array(coords_list, dtype=np.float64)

        # Store atomic numbers
        state.atomic_numbers_per_molecule = [atom[0] for atom in first_molecule_data.atoms_coords]
    else:
        raise ValueError("No molecule definitions found in the input file.")

    # Ensure iznu is a list of ints, as per SystemState definition
    state.iznu = [0] * state.natom 
    
    unique_elements = {}
    atom_idx = 0
    for mol_data in molecule_definitions:
        for atomic_num, _, _, _ in mol_data.atoms_coords:
            state.iznu[atom_idx] = atomic_num
            unique_elements[atomic_num] = unique_elements.get(atomic_num, 0) + 1
            atom_idx += 1

    state.num_elements_defined = len(unique_elements) # ielem
    state.element_types = [(z, unique_elements[z]) for z in sorted(list(unique_elements.keys()))]
    state.iz_types = [z for z, _ in state.element_types] # Populate iz_types from element_types
    state.nnu_types = [count for _, count in state.element_types] # Populate nnu_types from element_types

    # Populate atomic_number_to_symbol and atomic_number_to_weight maps 
    ELEMENT_SYMBOLS_REVERSE = {v: k for k, v in ELEMENT_SYMBOLS.items()}

    for atomic_num in unique_elements.keys(): # Iterate through the unique atomic numbers found in the system
        # Populate atomic_number_to_weight
        if atomic_num in ATOMIC_WEIGHTS:
            state.atomic_number_to_weight[atomic_num] = ATOMIC_WEIGHTS[atomic_num]
        else:
            # This error should ideally not happen if ATOMIC_WEIGHTS is comprehensive
            raise ValueError(f"Error: Atomic weight for number {atomic_num} not found in global ATOMIC_WEIGHTS dictionary.")

        # Populate atomic_number_to_symbol
        if atomic_num in ELEMENT_SYMBOLS_REVERSE:
            state.atomic_number_to_symbol[atomic_num] = ELEMENT_SYMBOLS_REVERSE[atomic_num]
        else:
            # This error should ideally not happen if ELEMENT_SYMBOLS is comprehensive
            raise ValueError(f"Error: Element symbol for atomic number {atomic_num} not found via ELEMENT_SYMBOLS.")
    
    state.imolec = [0] * (state.num_molecules + 1)
    current_atom_for_imolec = 0
    for i, mol_data in enumerate(molecule_definitions):
        state.imolec[i] = current_atom_for_imolec
        current_atom_for_imolec += mol_data.num_atoms
    state.imolec[state.num_molecules] = current_atom_for_imolec

    return molecule_definitions

# 12. QM Program Details Mapping
QM_PROGRAM_DETAILS = {
    1: {
        "name": "gaussian",
        "default_exe": "g09", # Common executable name.
        "input_ext": ".gjf",
        "output_ext": ".log",
        "energy_regex": r"SCF Done:\s+E\(.+\)\s*=\s*([-\d.]+)\s+A\.U\.",
        "termination_string": "Normal termination",
    },
    2: {
        "name": "orca",
        "default_exe": "orca", # Common executable name.
        "input_ext": ".inp",
        "output_ext": ".out",
        "energy_regex": r"FINAL SINGLE POINT ENERGY\s*([-\d.]+)",
        "termination_string": "ORCA TERMINATED NORMALLY",
    },
    # Add more QM programs here as needed
    # Example for a hypothetical "Mopac" program:
    # 3: {
    #     "name": "mopac",
    #     "default_exe": "mopac",
    #     "input_ext": ".mop",
    #     "output_ext": ".out",
    #     "energy_regex": r"TOTAL ENERGY\s*=\s*([-\d.]+)\s*EV", # Example regex
    #     "termination_string": "MOPAC DONE",
    # },
}

# Helper function to format and limit stream output (stdout/stderr)
def _format_stream_output(stream_content, max_lines=10, prefix="  "):
    if not stream_content:
        return ""
    
    lines = stream_content.splitlines()
    output_str = ""
    for i, line in enumerate(lines):
        if i < max_lines:
            output_str += f"{prefix}{line}\n"
        else:
            output_str += f"{prefix}... ({len(lines) - max_lines} more lines)\n"
            break
    return output_str

# 13. Calculate energy function
# It will run the QM calculation.
def calculate_energy(coords: np.ndarray, atomic_numbers: np.ndarray, state: SystemState, run_dir: str) -> Tuple[float, int]:
    """
    Calculates the energy of the given configuration using the external QM program.
    Returns the energy and a status code (1 for success, 0 for failure).
    Cleans up QM input/output/checkpoint files immediately after execution.
    """
    state.qm_call_count += 1
    call_id = state.qm_call_count
    
    qm_input_filename = f"qm_input_{call_id}.gjf"
    qm_output_filename = f"qm_output_{call_id}.log"
    qm_chk_filename = f"qm_chk_{call_id}.chk" # Checkpoint file name

    qm_input_path = os.path.join(run_dir, qm_input_filename)
    qm_output_path = os.path.join(run_dir, qm_output_filename)
    qm_chk_path = os.path.join(run_dir, qm_chk_filename) # Full path for checkpoint

    energy = 0.0
    status = 0 # 0 for failure, 1 for success
    
    try:
        # Generate QM input file
        with open(qm_input_path, 'w') as f:
            f.write(f"%chk={qm_chk_path}\n") # Use the full path for %chk
            f.write(f"%mem={state.qm_memory}\n")
            f.write(f"%nproc={state.qm_nproc}\n")
            f.write(f"# {state.qm_method}/{state.qm_basis_set} opt\n\n") # Added 'opt' keyword for optimization
            f.write("ASCEC QM Calculation\n\n")
            f.write(f"{state.charge} {state.multiplicity}\n")
            for i in range(state.natom):
                symbol = state.atomic_number_to_symbol.get(atomic_numbers[i], "X")
                f.write(f"{symbol} {coords[i, 0]:.6f} {coords[i, 1]:.6f} {coords[i, 2]:.6f}\n")
            f.write("\n") # Blank line to end molecule specification
            f.write(f"{state.qm_additional_keywords}\n") # Add additional keywords
            f.write("\n") # Blank line after additional keywords
    except IOError as e:
        print(f"Error writing QM input file {qm_input_path}: {e}", file=sys.stderr)
        return 0.0, 0 # Indicate failure

    # Determine QM command
    if state.qm_program == "g09":
        qm_command = f"g09 {qm_input_path} {qm_output_path}"
    elif state.qm_program == "orca":
        qm_command = f"orca {qm_input_path} > {qm_output_path}"
    elif state.qm_program == "qchem":
        qm_command = f"qchem {qm_input_path} {qm_output_path}"
    else:
        print(f"Error: Unsupported QM program '{state.qm_program}'", file=sys.stderr)
        # No need for specific cleanup here, the finally block will handle all generated files
        return 0.0, 0

    try:
        process = subprocess.run(qm_command, shell=True, capture_output=True, text=True, cwd=run_dir, check=False)
        
        # Print first few lines of stdout/stderr for debugging
        if process.stdout:
            print(f"{process.stdout[:500]}\n", file=sys.stderr)
        if process.stderr:
            print(f"{process.stderr[:500]}\n", file=sys.stderr)

        if process.returncode != 0:
            print(f"'{state.qm_program}' exited with non-zero status: {process.returncode}.\n\n", file=sys.stderr)
            status = 0 # Indicate failure
        elif not os.path.exists(qm_output_path):
            print(f"QM output file '{qm_output_path}' was not generated.", file=sys.stderr)
            status = 0 # Indicate failure
        else:
            # Parse energy from output file
            with open(qm_output_path, 'r') as f:
                output_content = f.read()
            
            # This regex will need to be robust for different QM programs
            if state.qm_program == "g09":
                match = re.search(r'SCF Done:  E\(\S+\) = \s*(-?\d+\.\d+)', output_content)
                if match:
                    energy = float(match.group(1))
                    status = 1
                else:
                    print(f"Could not find energy in G09 output file: {qm_output_path}", file=sys.stderr)
                    status = 0
            elif state.qm_program == "orca":
                # Example for ORCA: Final Single Point Energy: -76.432109879133 Eh
                match = re.search(r'Final Single Point Energy:\s*(-?\d+\.\d+)\s*Eh', output_content)
                if match:
                    energy = float(match.group(1))
                    status = 1
                else:
                    print(f"Could not find energy in ORCA output file: {qm_output_path}", file=sys.stderr)
                    status = 0
            elif state.qm_program == "qchem":
                # Example for Q-Chem: Total energy in the final basis set: -76.432109879133
                match = re.search(r'Total energy in the final basis set:\s*(-?\d+\.\d+)', output_content)
                if match:
                    energy = float(match.group(1))
                    status = 1
                else:
                    print(f"Could not find energy in Q-Chem output file: {qm_output_path}", file=sys.stderr)
                    status = 0
            else:
                print(f"Energy parsing not implemented for QM program '{state.qm_program}'", file=sys.stderr)
                status = 0
    
    except Exception as e:
        print(f"An error occurred during QM calculation or parsing: {e}", file=sys.stderr)
        status = 0
    finally:
        # Immediate cleanup of QM files for this specific call
        for fpath in [qm_input_path, qm_output_path, qm_chk_path]:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError as e:
                    print(f"  Error cleaning up {fpath}: {e}", file=sys.stderr)
        
    return energy, status

# 14. xyz confifuration 
# Helper function to write a single XYZ configuration to an open file object
def write_single_xyz_configuration(file_handle, natom, rp, iznu, energy, config_idx, 
                                   atomic_number_to_symbol, random_generate_config_mode, 
                                   remark="", include_dummy_atoms=False, state=None): # ADDED include_dummy_atoms and state
    """
    Writes a single XYZ configuration to a file handle.
    Optionally includes dummy 'X' atoms for the box corners.
    The 'remark' argument provides an optional string for the comment line.
    The output format of the second line changes based on 'random_generate_config_mode'.
    
    Args:
        file_handle (io.TextIOWrapper): The open file handle to write to.
        natom (int): Total number of actual atoms in the configuration.
        rp (np.ndarray): (N, 3) array of atomic coordinates.
        iznu (np.ndarray): (N,) array of atomic numbers.
        energy (float): Energy of the configuration.
        config_idx (int): The index/number of the current configuration.
        atomic_number_to_symbol (dict): Mapping from atomic number to symbol.
        random_generate_config_mode (int): The simulation mode (0 for random, 1 for annealing).
        remark (str, optional): An additional remark for the comment line.
        include_dummy_atoms (bool): If True, adds 8 dummy 'X' atoms for box visualization.
        state (SystemState, optional): The SystemState object, REQUIRED if include_dummy_atoms is True
                                       to get the 'xbox' (box dimensions).
    """
    if include_dummy_atoms and (state is None or not hasattr(state, 'xbox')):
        raise ValueError("SystemState object with 'xbox' attribute must be provided to "
                         "write_single_xyz_configuration when include_dummy_atoms is True "
                         "(to get box dimensions).")
    
    # Get box length. Only relevant if dummy atoms are being added.
    L = state.xbox if include_dummy_atoms and state else 0.0 

    # Define the 8 corner coordinates of the box (assuming L is correctly set)
    box_corners = np.array([
        [0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L],
        [L, L, 0], [L, 0, L], [0, L, L], [L, L, L]
    ])
    dummy_atom_count = len(box_corners) # Always 8 for a cube

    total_atoms_in_frame = natom
    if include_dummy_atoms:
        total_atoms_in_frame += dummy_atom_count

    file_handle.write(f"{total_atoms_in_frame}\n")
    
    # Construct the base comment line based on random_generate_config_mode
    base_comment_line = ""
    if random_generate_config_mode == 0: # Random Configuration Generation Mode
        base_comment_line = f"Configuration: {config_idx + 1}"
    else: # Annealing Mode (or any other mode where energy is relevant)
        base_comment_line = f"Energy: {energy:.8f} a.u. (Config {config_idx})"
        if remark:
            base_comment_line += f" - {remark}"
    
    # Append box information if dummy atoms are included
    if include_dummy_atoms:
        if L > 0: # Only add box length if it's a meaningful value
            base_comment_line += f" BoxL={L:.3f}"
        base_comment_line += f" (with {DUMMY_ATOM_SYMBOL} box markers)"

    file_handle.write(f"{base_comment_line}\n")
    
    # Write actual atom coordinates with user's specified formatting
    for i in range(natom):
        # Keep original fallback to 'X' if symbol not found for the atomic number
        symbol = atomic_number_to_symbol.get(iznu[i], 'X') 
        file_handle.write(f"{symbol: <3} {rp[i, 0]: 12.6f} {rp[i, 1]: 12.6f} {rp[i, 2]: 12.6f}\n")

    # Add dummy atom coordinates if requested, using the same formatting
    if include_dummy_atoms:
        for coords in box_corners:
            file_handle.write(f"{DUMMY_ATOM_SYMBOL: <3} {coords[0]: 12.6f} {coords[1]: 12.6f} {coords[2]: 12.6f}\n")

    file_handle.flush()

# 15. Propose Move Function
def propose_move(state: SystemState, current_rp: np.ndarray, current_imolec: List[int]):
    """
    Proposes a random translation and rotation for a single molecule,
    applying Fortran's 'trans' (CM bounce) and 'rotac' (Euler angle rotation) logic.
    """
    # Randomly select a molecule to move
    molecule_idx = np.random.randint(0, state.num_molecules)

    start_atom_idx = current_imolec[molecule_idx] # imolec(imo-1)+1 in Fortran
    end_atom_idx = current_imolec[molecule_idx + 1] # imolec(imo) in Fortran
    
    # Prepare proposed_rf as a copy of current_rp
    proposed_rf = np.copy(current_rp)

    # Get the current coordinates of the selected molecule
    mol_coords_current = current_rp[start_atom_idx:end_atom_idx, :]
    
    # Calculate current center of mass for the selected molecule
    # Ensure state.iznu and state.atomic_number_to_mass are correctly populated
    mol_atomic_numbers = [state.iznu[i] for i in range(start_atom_idx, end_atom_idx)]
    mol_masses = np.array([state.atomic_number_to_mass.get(anum, 1.0) # Default to 1.0 if mass not found
                           for anum in mol_atomic_numbers])
    
    current_rcm = calculate_mass_center(mol_coords_current, mol_masses)

    # --- Translation Part (matching Fortran's 'trans' subroutine) ---
    # Generate random displacement vector for CM, range [-max_displacement_a, +max_displacement_a]
    # Fortran's 'ds' is state.max_displacement_a
    random_displacement_vector = (state.ran0_method(size=3) - 0.5) * 2.0 * state.max_displacement_a

    # We need to apply the 'bounce' to the displacement vector itself
    # and then update current_rcm (which effectively becomes the new CM in the Fortran loop)
    
    # Fortran's 'xbox' is equivalent to your 'state.cube_length'
    half_xbox = state.cube_length / 2.0 
    
    # Create a temporary 'new_rcm' that will be updated in place, mimicking Fortran's rcm(imo,i)
    # This also serves as the center for atoms' relative coordinates after translation.
    new_rcm_after_translation = np.copy(current_rcm)

    # Displacement to be applied to individual atoms after CM bounce
    actual_atom_displacement = np.copy(random_displacement_vector)

    for dim in range(3):
        # Update CM (rcm(imo,i)=rcm(imo,i)+s in Fortran)
        new_rcm_after_translation[dim] += random_displacement_vector[dim]

        # Apply Fortran's CM boundary bounce logic for this dimension
        if np.abs(new_rcm_after_translation[dim]) > half_xbox:
            # If CM goes out, reverse the displacement that was applied to CM
            # and reverse the 's' for atoms.
            new_rcm_after_translation[dim] -= 2.0 * random_displacement_vector[dim] # Reverse CM pos
            actual_atom_displacement[dim] = -random_displacement_vector[dim] # Reverse atom displacement
    
    # Apply the final (potentially bounced) displacement to all atoms of the selected molecule
    # This matches `r(inu,i)=r(inu,i)+s` in Fortran's `trans`.
    proposed_rf[start_atom_idx:end_atom_idx, :] += actual_atom_displacement

    # --- Rotation Part (matching Fortran's 'rotac' subroutine with Euler angles) ---
    # Generate random Euler angles (alpha, beta, gamma) for Z, Y, X rotations respectively
    # Fortran's 'dphi' is state.max_rotation_angle_rad
    alpha = (state.ran0_method() - 0.5) * 2.0 * state.max_rotation_angle_rad # Rotation around Z-axis (za in Fortran)
    beta = (state.ran0_method() - 0.5) * 2.0 * state.max_rotation_angle_rad  # Rotation around Y-axis (ya in Fortran)
    gamma = (state.ran0_method() - 0.5) * 2.0 * state.max_rotation_angle_rad # Rotation around X-axis (xa in Fortran)

    # Fortran's combined rotation matrix (Rz @ Ry @ Rx)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(gamma), -np.sin(gamma)],
        [0, np.sin(gamma), np.cos(gamma)]
    ], dtype=np.float64)

    Ry = np.array([
        [np.cos(beta), 0, np.sin(beta)],
        [0, 1, 0],
        [-np.sin(beta), 0, np.cos(beta)]
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(alpha), -np.sin(alpha), 0],
        [np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    rotation_matrix = Rz @ Ry @ Rx

    # Get the coordinates of the molecule *after translation* (from proposed_rf)
    # Calculate the CM of the molecule *after translation* (this will be the 'new_rcm_after_translation' we calculated)
    
    # Translate atoms to CM origin, rotate, then translate back
    # `rel` in Fortran is `r - rcm`. So, `mol_coords_relative_to_cm = (mol_coords_after_translation - new_rcm_after_translation)`
    mol_coords_relative_to_cm = proposed_rf[start_atom_idx:end_atom_idx, :] - new_rcm_after_translation
    
    # Apply rotation
    rotated_relative_coords = (rotation_matrix @ mol_coords_relative_to_cm.T).T

    # Translate back to the CM position
    # This matches `r(inu,k)=rel(inu,k)+rcm(imo,k)` in Fortran's `rotac`.
    proposed_rf[start_atom_idx:end_atom_idx, :] = rotated_relative_coords + new_rcm_after_translation

    # --- NO ADDITIONAL GLOBAL BOUNDARY CHECKS HERE ---
    # The Fortran program relies on the 'trans' subroutine's CM bounce
    # and the implicit behavior of rotations to keep things within bounds,
    # or expects the viewer to handle wrapping.

    return proposed_rf, proposed_rf[start_atom_idx:end_atom_idx, :], molecule_idx, "translate_rotate"

# Function to clean up generated QM input/output files
def cleanup_qm_files(files_to_clean: List[str]):
    """
    Cleans up any QM-related files explicitly added to the list.
    This function now primarily serves as a safeguard for files that might not have been
    cleaned by individual calculate_energy calls due to unexpected crashes or other issues.
    """
    cleaned_count = 0
    files_to_remove = list(files_to_clean) # Create a copy to iterate
    for fpath in files_to_remove:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                # Remove from the original list if successful. This means the list is mutated.
                # It's better to clear the list at the end of main_ascec_integrated,
                # as calculate_energy no longer uses this list for adding.
                cleaned_count += 1
            except OSError as e:
                print(f"Error removing file {fpath} during final cleanup: {e}", file=sys.stderr)
    if cleaned_count > 0:
        print(f"Cleaned up {cleaned_count} leftover QM files during final cleanup.", file=sys.stderr)
    files_to_clean.clear() # Clear the list so it's empty for next run if script is not fully restarted.

# 16. main ascec integrated function
def main_ascec_integrated():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="ASCEC: Annealing Simulation")
    parser.add_argument("input_file", help="Path to the input file (e.g., w5-apy.in)")
    args = parser.parse_args()

    # Initialize file handles to None
    out_file_handle = None 
    xyz_file_handle = None 
    box_xyz_copy_handle = None # NEW handle for the box copy file
    failed_initial_configs_xyz_handle = None 
    failed_configs_path = None 

    # Determine the directory where the input file is located
    input_file_path_full = os.path.abspath(args.input_file)
    run_dir = os.path.dirname(input_file_path_full)
    if not run_dir:
        run_dir = os.getcwd()

    qm_files_to_clean: List[str] = []
    initial_qm_successful = False 

    try:
        state = SystemState()
        state.qm_call_count = 0 

        read_input_file(state, args.input_file)

        if not state.atomic_number_to_symbol:
             state.atomic_number_to_symbol = {v: k for k, v in ELEMENT_SYMBOLS.items()}

        state.current_energy = 0.0 
        state.lowest_energy = float('inf')

        if state.random_seed is None or \
           not isinstance(state.random_seed, int) or \
           not (0 <= state.random_seed <= (2**32 - 1)):
            
            generated_seed = random.randint(100000, 999999)
            if state.random_seed is not None: 
                state.random_seed = generated_seed
        
        print(f"\nUsing random seed: {state.random_seed}\n", file=sys.stderr)

        random.seed(state.random_seed)
        np.random.seed(state.random_seed)

        input_base_name = os.path.splitext(os.path.basename(args.input_file))[0]

        # Define and Open .out File
        out_filename = f"{input_base_name}.out" 
        out_file_path = os.path.join(run_dir, out_filename)
        
        try:
            out_file_handle = open(out_file_path, 'w')
        except IOError as e:
            print(f"CRITICAL ERROR: Could not open out file '{out_file_path}': {e}", file=sys.stderr)
            sys.exit(1)

        # Define and Open ORIGINAL XYZ File (mto_seed.xyz)
        xyz_filename = f"mto_{state.random_seed}.xyz" # Base name for the original XYZ
        xyz_file_path = os.path.join(run_dir, xyz_filename) 
        
        try:
            xyz_file_handle = open(xyz_file_path, 'w')
        except IOError as e:
            print(f"CRITICAL ERROR: Could not open XYZ file '{xyz_file_path}': {e}", file=sys.stderr)
            if out_file_handle:
                print(f"CRITICAL ERROR: Could not open XYZ file '{xyz_file_path}': {e}", file=out_file_handle)
            sys.exit(1)

        # Define and Open BOX XYZ COPY File (mtobox_seed.xyz) if requested
        if CREATE_BOX_XYZ_COPY:
            # Dynamically create the name for the copy: mtobox_seed.xyz
            box_xyz_copy_filename = xyz_filename.replace("mto_", "mtobox_")
            box_xyz_copy_path = os.path.join(run_dir, box_xyz_copy_filename)
            
            try:
                box_xyz_copy_handle = open(box_xyz_copy_path, 'w')
            except IOError as e:
                print(f"CRITICAL ERROR: Could not open BOX XYZ copy file '{box_xyz_copy_path}': {e}", file=sys.stderr)
                if out_file_handle: 
                    print(f"CRITICAL ERROR: Could not open BOX XYZ copy file '{box_xyz_copy_path}': {e}", file=out_file_handle)
                sys.exit(1)

        # Write Simulation Summary to .out File (passing original XYZ filename)
        write_simulation_summary(state, out_file_handle, xyz_filename)
        out_file_handle.flush() 

        if state.random_generate_config == 1: 
            failed_configs_filename = os.path.splitext(os.path.basename(args.input_file))[0] + "_failed_initial_configs.xyz"
            failed_configs_path = os.path.join(run_dir, failed_configs_filename)
            failed_initial_configs_xyz_handle = open(failed_configs_path, 'w')
            print("Keeping initial configuration attempts.\n", file=sys.stderr) 
            initial_failed_config_idx = 0 

        # Define half_xbox once for coordinate system shift
        half_xbox = state.xbox / 2.0

        # Random Configuration Generation Mode
        if state.random_generate_config == 0: 
            print(f"Generating {state.num_random_configs} random configurations.\n", file=sys.stderr)
            
            current_config_idx = 0 
            for i in range(state.num_random_configs):
                state.rp = np.zeros((state.natom, 3), dtype=np.float64) 
                config_molecules(state.natom, state.num_molecules, state.rp, state) 
                
                # --- APPLY SHIFT FOR VISUALIZATION HERE ---
                # Shift coordinates from [-L/2, L/2] to [0, L] range for visualization output
                rp_for_viz = state.rp + half_xbox 
                
                # ALWAYS write to the ORIGINAL XYZ file handle (no dummy atoms)
                write_single_xyz_configuration(
                    xyz_file_handle,
                    state.natom,
                    rp_for_viz, # Use shifted coordinates
                    state.iznu,
                    state.current_energy, 
                    current_config_idx, 
                    state.atomic_number_to_symbol,
                    state.random_generate_config, 
                    remark="", 
                    include_dummy_atoms=False, 
                    state=state 
                )                
                
                # Conditionally write to the BOX XYZ COPY file handle (with dummy atoms)
                if CREATE_BOX_XYZ_COPY:
                    write_single_xyz_configuration(
                        box_xyz_copy_handle,
                        state.natom,
                        rp_for_viz, # Use shifted coordinates
                        state.iznu,
                        state.current_energy, 
                        current_config_idx, 
                        state.atomic_number_to_symbol,
                        state.random_generate_config, 
                        remark="", 
                        include_dummy_atoms=True, 
                        state=state 
                    )

                current_config_idx += 1
            print("Random configuration generation complete.\n", file=sys.stderr)

        # Annealing Mode (state.random_generate_config = 1)
        elif state.random_generate_config == 1: 
            print("Starting annealing simulation...\n", file=sys.stderr)
            state.current_temp = state.linear_temp_init if state.quenching_routine == 1 else state.geom_temp_init

            max_initial_qm_retries = state.initial_qm_retries 

            initial_qm_calculation_succeeded = False 
            for attempt in range(max_initial_qm_retries):
                if attempt > 0:
                    state.rp, state.iznu = initialize_molecular_coords_in_box(state)
                
                initial_failed_config_idx += 1 

                remark_line = f"Attempt {attempt + 1}/{max_initial_qm_retries} (before QM calc)"
                # For failed_initial_configs_xyz_handle, we still want to shift them for visualization.
                # If you need absolute (possibly negative) coordinates for debugging, you would omit this shift.
                # But for consistency with output, shifting is generally better.
                rp_for_viz_failed = state.rp + half_xbox 
                write_single_xyz_configuration(
                    failed_initial_configs_xyz_handle, 
                    state.natom, rp_for_viz_failed, state.iznu, state.current_energy, # Use shifted coords
                    initial_failed_config_idx, state.atomic_number_to_symbol,
                    state.random_generate_config, remark_line, state=state 
                )
                
                qm_attempt_failed = False 
                error_details = ""        

                try:
                    print(f"Initial energy calculation (Attempt {attempt + 1}/{max_initial_qm_retries})\n", file=sys.stderr)
                    initial_energy, jo_status = calculate_energy(state.rp, state.iznu, state, run_dir)
                    
                    if jo_status == 0:
                        raise RuntimeError("QM program returned non-zero status or could not calculate energy.") 

                    state.current_energy = initial_energy
                    state.lowest_energy = state.current_energy 
                    
                    print(f"Initial QM calculation successful. Energy: {state.current_energy:.8f} a.u.\n", file=sys.stderr)
                    initial_qm_calculation_succeeded = True
                    break 

                except RuntimeError as e:
                    qm_attempt_failed = True
                    error_details = str(e) 
                except Exception as e:
                    qm_attempt_failed = True
                    error_details = f"An unexpected internal error occurred: {e}" 
                
                if qm_attempt_failed:
                    print(f"Initial QM calculation attempt {attempt + 1}/{max_initial_qm_retries} failed.\n", file=sys.stderr)
                    
                    if attempt < max_initial_qm_retries - 1:
                        print("Generating new initial configuration and retrying.\n", file=sys.stderr)
                    else:
                        raise RuntimeError(
                            f"All {max_initial_qm_retries} attempts to perform the initial QM energy calculation failed. "
                            "Please verify your QM input parameters (method, basis set, memory, processors) "
                            "or inspect the individual QM program output files in the run directory for specific errors. "
                            "Cannot proceed with annealing simulation."
                        )
            
            if not initial_qm_calculation_succeeded:
                raise RuntimeError("Initial QM energy calculation did not succeed after all retries. Exiting.")

            if state.quenching_routine == 1: 
                total_annealing_steps = state.linear_num_steps
            elif state.quenching_routine == 2: 
                total_annealing_steps = state.geom_num_steps
            else:
                print("Error: Invalid quenching_routine specified for annealing mode. Must be 1 (linear) or 2 (geometric).", file=sys.stderr)
                return 

            current_config_idx = 1 # Initialize for main annealing trajectory

            # Write the first *successful* QM configuration to the main XYZ file
            # --- APPLY SHIFT HERE FOR SUCCESSFUL TRAJECTORY ---
            rp_for_viz = state.rp + half_xbox 
            write_single_xyz_configuration(
                xyz_file_handle, 
                state.natom, rp_for_viz, state.iznu, state.current_energy, # Use shifted coordinates
                current_config_idx, state.atomic_number_to_symbol,
                state.random_generate_config, state=state 
            )
            # Conditionally write the box copy for the first successful configuration
            if CREATE_BOX_XYZ_COPY:
                write_single_xyz_configuration(
                    box_xyz_copy_handle, 
                    state.natom, rp_for_viz, state.iznu, state.current_energy, # Use shifted coordinates
                    current_config_idx, state.atomic_number_to_symbol,
                    state.random_generate_config, include_dummy_atoms=True, state=state 
                )

            current_config_idx += 1 

            for step in range(total_annealing_steps):
                print(f"\nAnnealing Step {step + 1}/{total_annealing_steps} at Temperature: {state.current_temp:.2f} K", file=sys.stderr)
                
                for cycle in range(state.max_cycle):
                    # propose_move returns proposed_rp in the [-L/2, L/2] system
                    proposed_rp, proposed_rf_unused, molecule_idx, move_type = propose_move(state, state.rp, state.imolec)
                    
                    proposed_energy, jo_status = calculate_energy(proposed_rp, state.iznu, state, run_dir)
                    
                    if jo_status == 0:
                        print(f"  Warning: Proposed QM energy calculation failed for cycle {cycle + 1}. Rejecting move.", file=sys.stderr)
                        continue

                    accept_move = False
                    if proposed_energy < state.current_energy:
                        accept_move = True
                    else:
                        delta_e = proposed_energy - state.current_energy
                        if state.current_temp > 0:
                            acceptance_probability = math.exp(-delta_e / (B2 * state.current_temp))
                            if random.random() < acceptance_probability:
                                accept_move = True
                        
                    if accept_move:
                        state.rp = proposed_rp # state.rp is updated with the new coordinates in [-L/2, L/2]
                        state.current_energy = proposed_energy
                        
                    if state.current_energy < state.lowest_energy:
                        state.lowest_energy = state.current_energy
                        print(f"  New lowest energy found: {state.lowest_energy:.8f} a.u.", file=sys.stderr)

                # Write snapshot after each annealing step (after all cycles)
                # --- APPLY SHIFT HERE FOR ANNEALING TRAJECTORY ---
                rp_for_viz = state.rp + half_xbox 
                
                # Write to ORIGINAL XYZ file
                write_single_xyz_configuration(
                    xyz_file_handle, 
                    state.natom, rp_for_viz, state.iznu, state.current_energy, # Use shifted coordinates
                    current_config_idx, state.atomic_number_to_symbol,
                    state.random_generate_config, state=state 
                )
                # Write to BOX XYZ COPY file
                if CREATE_BOX_XYZ_COPY:
                    write_single_xyz_configuration(
                        box_xyz_copy_handle, 
                        state.natom, rp_for_viz, state.iznu, state.current_energy, # Use shifted coordinates
                        current_config_idx, state.atomic_number_to_symbol,
                        state.random_generate_config, include_dummy_atoms=True, state=state 
                    )

                current_config_idx += 1 

                if state.quenching_routine == 1:
                    state.current_temp = max(state.current_temp - state.linear_temp_decrement, 0.001)
                elif state.quenching_routine == 2:
                    state.current_temp = max(state.current_temp * state.geom_temp_factor, 0.001)
                
            print("Annealing simulation finished.\n", file=sys.stderr)
            print(f"Lowest energy found: {state.lowest_energy:.8f} a.u.", file=sys.stderr)

    finally:
        # Ensure all output files are closed
        if out_file_handle:
            try: out_file_handle.close()
            except Exception as e: print(f"Error closing main output file: {e}", file=sys.stderr)
        
        if xyz_file_handle:
            try: xyz_file_handle.close()
            except Exception as e: print(f"Error closing original XYZ file: {e}", file=sys.stderr)
        
        if box_xyz_copy_handle: 
            try: box_xyz_copy_handle.close()
            except Exception as e: print(f"Error closing BOX XYZ copy file: {e}", file=sys.stderr)
        
        if failed_initial_configs_xyz_handle:
            try: failed_initial_configs_xyz_handle.close()
            except Exception as e: print(f"Error closing failed initial configs file: {e}", file=sys.stderr)

        # Create .mol file for the ORIGINAL XYZ file
        mol_filename = os.path.splitext(xyz_file_path)[0] + ".mol"
        try:
            convert_xyz_to_mol(xyz_file_path, state.openbabel_alias)
        except Exception as e:
            print(f"Warning: Could not create .mol file for '{xyz_file_path}': {e}", file=sys.stderr)

        # Create .mol file for the BOX XYZ COPY file, if it was created
        if CREATE_BOX_XYZ_COPY and box_xyz_copy_handle:
            mol_box_filename = os.path.splitext(box_xyz_copy_path)[0] + ".mol"
            try:
                convert_xyz_to_mol(box_xyz_copy_path, state.openbabel_alias)
                
                temp_mol_path = mol_box_filename + ".tmp"
                with open(mol_box_filename, 'r') as infile, open(temp_mol_path, 'w') as outfile:
                    for line in infile:
                        # This replacement needs to be careful not to break spacing.
                        # The '*' in MOL atom lines is typically followed by two spaces and then numbers.
                        # We'll target the pattern ' * ' (space, asterisk, space) specifically.
                        # Example: '    0.0000    0.0000    0.0000 * 0  0  0  0  0  0  0  0  0  0  0  0'
                        # The atom symbol is typically right-justified in a 3-character field,
                        # so it often appears as ' *' or '* ' or 'X  ' etc.
                        # A robust way is to specifically target the atom symbol position if possible,
                        # but a simple string replacement for ' *' to ' X ' might work
                        # given your specific output example (seems like ' * ' for a single char).

                        # Let's try replacing ' * ' with ' X ' carefully,
                        # or ' * ' with ' X  ' if that's the pattern.
                        # Given your input, it looks like '* ' followed by numbers.
                        # The MOL specification often has fixed width fields.
                        # Let's try string slicing or a more precise regex.
                        
                        # Simpler approach: find the '*' and replace it carefully.
                        # The atom symbol in .mol is usually at character index 31-33
                        # Let's read the line and try to identify the symbol field if it's there.
                        
                        # Check if the line is long enough to contain an atom symbol field
                        # Atom lines typically start after the header and connect block
                        if len(line) >= 34 and line[31] == '*' and line[32] == ' ': # A common pattern for '*' followed by space
                            # Replace the '*' at index 31 with 'X'
                            modified_line = line[:31] + DUMMY_ATOM_SYMBOL + line[32:]
                            outfile.write(modified_line)
                        else:
                            # If it's not an atom line with a '*' at that specific position,
                            # or if it's shorter, write the line as is.
                            outfile.write(line)
                
                # Replace the original .mol file with the modified one
                shutil.move(temp_mol_path, mol_box_filename)

            except Exception as e:
                print(f"Warning: Could not create or post-process .mol file for '{box_xyz_copy_path}': {e}", file=sys.stderr)
                
        if failed_initial_configs_xyz_handle and initial_qm_successful:
            if failed_configs_path and os.path.exists(failed_configs_path):
                try:
                    os.remove(failed_configs_path)
                    print(f"Removed '{failed_configs_path}' as initial QM calculation succeeded.", file=sys.stderr)
                except OSError as e:
                    print(f"Error removing '{failed_configs_path}': {e}", file=sys.stderr)
            else:
                print(f"Keeping initial configuration attempts (QM calc not fully successful).", file=sys.stderr)
        
        cleanup_qm_files(qm_files_to_clean)

# This ensures main_ascec_integrated() is called when the script is run
if __name__ == "__main__":
    main_ascec_integrated()