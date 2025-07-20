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
from datetime import timedelta # To format wall time

# Global Constants
MAX_ATOM = 1000 # Increase this if your systems are larger than 1000 atoms
MAX_MOLE = 100  # Increase this if you have more than 100 molecules
B2 = 3.166811563e-6   # Boltzmann constant in Hartree/K (approx. 3.166811563 × 10^-6 Hartree/K)

# Constants for overlap prevention during initial configuration generation (in config_molecules)
OVERLAP_SCALE_FACTOR = 0.7 # Factor to make overlap check slightly more lenient (e.g., allow partial overlap)
MAX_OVERLAP_PLACEMENT_ATTEMPTS = 100000 # Max attempts to place a single molecule without significant overlap

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
    "Sb": 51, "Te": 52, 53: 'I',  54: 'Xe', 55: 'Cs',
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

# Atomic Weights for elements (Global Constant) - used for center of mass calculations
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

# Electronegativity for elements (used for sorting molecular formula strings)
ELECTRONEGATIVITY_VALUES = {
    'H': 2.20, 'He': 0.0, 
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
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107,
    'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, # Corrected some values based on common tables
    'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
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
        self.max_cycle: int = 0                   # Maximum Monte Carlo Cycles per Temperature (initial value from input)
        self.max_displacement_a: float = 0.0      # Maximum Displacement of each mass center (ds)
        self.max_rotation_angle_rad: float = 0.0  # Maximum Rotation angle in radians (dphi)
        self.ia: int = 0                          # QM program type (1: Gaussian, 2: ORCA, etc.)
        self.alias: str = ""                      # Program alias/executable name (e.g., "g09")
        self.qm_method: Optional[str] = None      # (e.g., "pm3", "hf") - Renamed from hamiltonian for clarity
        self.qm_basis_set: Optional[str] = None   # (e.g., "6-31G*", "STO-3G") - Renamed from basis_set for clarity
        self.charge: int = 0                      # (iQ)
        self.multiplicity: int = 0                # (iS2) - Renamed from spin_multiplicity for clarity
        self.qm_memory: Optional[str] = None      # memory - No default, will be None if not in input
        self.qm_nproc: Optional[int] = None       # nprocs
        self.qm_additional_keywords: str = ""     # if necessary
        self.num_molecules: int = 0               # (nmo) from input file
        self.output_dir: str = "."                # Directory for output files
        self.ivalE: int = 0                       # Evaluate Energy flag (0: No energy evaluation, just mto movements)
        self.mto: int = 0                         # Number of random configurations if ivalE=0 (Fortran's nrandom)
        self.qm_program: Optional[str] = None     # Will store "gaussian", "orca", etc., derived from alias or ia
        self.molecules_to_add = []                # Initialize as an empty list
        self.all_molecule_definitions = []        # Also good to initialize this if not already
        self.openbabel_alias = "obabel"           # Default Open Babel executable name
        
        # Add these from global constants
        self.R_ATOM = R_ATOM.copy() 
        self.OVERLAP_SCALE_FACTOR = OVERLAP_SCALE_FACTOR
        self.MAX_OVERLAP_PLACEMENT_ATTEMPTS = 100000 # Max attempts to place a single molecule without significant overlap

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
        self.lowest_energy: float = float('inf')  # oldE (stores the lowest energy found, initialize to large value)
        self.lowest_energy_rp: Optional[np.ndarray] = None # Coordinates of the lowest energy config
        self.lowest_energy_iznu: Optional[List[int]] = None # Atomic numbers of the lowest energy config
        self.lowest_energy_config_idx: int = 0    # Index of the lowest energy config
        self.current_temp: float = 0.0            # Ti (current annealing temperature)
        self.maxstep: int = 0                     # Current maximum steps at a temperature (derived from max_cycle initially, then reduced)
        self.natom: int = 0                       # Total number of atoms, calculated from molecule definitions
        self.nmo: int = 0                         # Number of molecules (often kept in sync with num_molecules)
        self.jdum: str = ""                       # Timestamp/label for output files (can be populated from alias or datetime)
        self.xbox: float = 0.0                    # Length of the simulation cube (synced with cube_length)
        self.ds: float = 0.0                      # Max displacement (synced with max_displacement_a)
        self.dphi: float = 0.0                    # Max rotation angle (synced with max_rotation_angle_rad)
        self.last_accepted_qm_call_count: int = 0 # Stores qm_call_count at last accepted configuration for history
        self.total_accepted_configs: int = 0      # Counter for total accepted configurations written to XYZ
        self.consecutive_na_steps: int = 0        # Counter for consecutive N/A steps in annealing
        
        # New variable to track the global QM call count at the last history line written
        self.last_history_qm_call_count: int = 0

        # --- Molecular and Atomic Data ---
        self.rp: np.ndarray = np.empty((0, 3), dtype=np.float64) # Current atomic coordinates
        self.rf: np.ndarray = np.empty((0, 3), dtype=np.float64) # Proposed atomic coordinates (not directly used as state, but as return value)
        self.rcm: np.ndarray = np.empty((MAX_MOLE, 3), dtype=np.float64) # Center of mass for each molecule
        self.imolec: List[int] = []               # Indices defining molecules (which atoms belong to which molecule)
        self.iznu: List[int] = []                 # Atomic numbers for each atom in the system
        self.all_molecule_definitions: List['MoleculeData'] = [] # Stores parsed molecule data from input
        self.coords_per_molecule: np.ndarray = np.empty((0,3), dtype=np.float64) # Template coordinates for one molecule
        self.atomic_numbers_per_molecule: List[int] = [] # Template atomic numbers for one molecule
        self.natom_per_molecule: int = 0          # Number of atoms in the template molecule
        self.molecule_label: str = ""             # Label of the template molecule

        # Element Maps
        self.atomic_number_to_symbol: Dict[int, str] = {}
        self.atomic_number_to_weight: Dict[int, float] = {} # Populated from ATOMIC_WEIGHTS
        self.atomic_number_to_mass: Dict[int, float] = {}   # Alias for atomic_number_to_weight, used for clarity in COM calcs

        # These Fortran-style arrays might be redundant if the dicts above are the primary source.
        self.sym: List[str] = [''] * 120          # From Fortran's symelem (symbol for atomic number index)
        self.wt: List[float] = [0.0] * 120        # From Fortran's wtelem (weight for atomic number index)
        
        # Additional QM related attributes
        self.qm_call_count: int = 0 # Counter for QM calculation calls (cumulative global count)
        self.initial_qm_retries: int = 20 # Number of retries for initial QM calculation, default to 20
        self.verbosity_level: int = 0 # 0: default, 1: --v (every 10 steps), 2: --va (all steps)
        self.use_standard_metropolis: bool = False # Flag for Metropolis criterion

    def ran0_method(self) -> float:
        """
        A Python implementation of Numerical Recipes' ran0 random number generator.
        This method is kept for historical context/compatibility with the original Fortran.
        For new Python code, `random.random()` or `np.random.rand()` are generally preferred.
        """
        IA = 16807
        IM = 2147483647 # 2^31 - 1
        AM = 1.0 / IM
        IQ = 127773
        IR = 2836
        NTAB = 32
        NDIV = 1 + (IM - 1) // NTAB
        EPS = 1.2e-7
        RNMX = 1.0 - EPS

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

        self.IVEC[j] = self.random_seed # This line was incorrect in previous version (used self.IVEC[j] = self.IY)
        self.IY = self.IVEC[j] # Correct usage: self.IY becomes the value from IVEC[j]

        if self.IY > RNMX * IM:
            return RNMX
        else:
            return float(self.IY) * AM

    def get_molecule_label_from_atom_index(self, atom_abs_idx: int) -> str:
        """
        Determines the label of the molecule to which a given atom belongs.
        Assumes `imolec` and `all_molecule_definitions` are correctly populated.
        """
        if not self.imolec or not self.all_molecule_definitions:
            return "Unknown"

        # Find which molecule this atom belongs to based on imolec
        # imolec stores the starting atom index for each molecule definition,
        # plus the total number of atoms at the end.
        for i in range(self.num_molecules):
            start_idx = self.imolec[i]
            end_idx = self.imolec[i+1] # This is the start of the *next* molecule, or total atoms

            if start_idx <= atom_abs_idx < end_idx:
                # The molecule_definitions list is indexed from 0 to (num_molecules - 1)
                # This needs to correctly map to the *instance* of the molecule.
                # Since molecules_to_add now stores the correct definition index for each instance, use that.
                mol_def_idx = self.molecules_to_add[i]
                return self.all_molecule_definitions[mol_def_idx].label
        return "Unknown"

# Helper for verbose printing
def _print_verbose(message: str, level: int, state: SystemState, file=sys.stderr):
    """
    Prints a message to stderr if the state's verbosity level meets or exceeds the required level.
    level 0: always print (critical errors, final summary, accepted configs, and key annealing steps)
    level 1: --v (every 10 steps), plus level 0)
    level 2: --va (all steps, plus level 0 and 1)
    """
    if state.verbosity_level >= level:
        print(message, file=file)
        file.flush()


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

def _post_process_mol_file(mol_filepath: str, state: SystemState):
    """
    Post-processes a .mol file to replace Open Babel's '*' dummy atom symbol with 'X'
    if the DUMMY_ATOM_SYMBOL is 'X'.
    """
    if DUMMY_ATOM_SYMBOL != "X": # Only needed if we actually use 'X' and obabel uses '*'
        return # Skip post-processing if not using 'X' as dummy or if dummy is default *

    temp_mol_path = mol_filepath + ".tmp"
    try:
        with open(mol_filepath, 'r') as infile, open(temp_mol_path, 'w') as outfile:
            for line in infile:
                # MOL format atom block: ATOM_NUMBER X Y Z ELEMENT_SYMBOL ...
                # Open Babel sometimes uses '*' as a dummy atom. The symbol is typically at column 32 (index 31)
                # Example: "  1 C       0.0000    0.0000    0.0000  0  0  0  0  0  0  0  0"
                # If there's a '*' at position 31 and a space after it, it's likely a dummy atom placeholder from OB.
                if len(line) >= 34 and line[31] == '*' and line[32] == ' ':
                            modified_line = line[:31] + DUMMY_ATOM_SYMBOL + line[32:] 
                            outfile.write(modified_line)
                else:
                    outfile.write(line)
        shutil.move(temp_mol_path, mol_filepath)
        _print_verbose(f"    Post-processed '{os.path.basename(mol_filepath)}' for dummy atoms.", 1, state)
    except Exception as e:
        _print_verbose(f"    Warning: Could not post-process .mol file '{os.path.basename(mol_filepath)}': {e}", 0, state)


def convert_xyz_to_mol(xyz_filepath: str, openbabel_alias: str = "obabel", state: SystemState = None) -> bool:
    """
    Converts a single .xyz file to a .mol file using Open Babel.
    Prints verbose messages indicating success or failure to sys.stderr.
    Returns True on successful conversion, False otherwise.
    """
    mol_filepath = os.path.splitext(xyz_filepath)[0] + ".mol"

    openbabel_full_path = shutil.which(openbabel_alias)
    if not openbabel_full_path:
        _print_verbose(f"\n  Open Babel ({openbabel_alias}) command not found or not executable. Skipping .mol conversion for '{os.path.basename(xyz_filepath)}'.", 0, state)
        _print_verbose("  Please ensure Open Babel is installed and added to your system's PATH, or provide the correct alias.", 0, state)
        return False

    try:
        conversion_command = [openbabel_full_path, "-i", "xyz", xyz_filepath, "-o", "mol", "-O", mol_filepath]
        _print_verbose(f"  Converting '{os.path.basename(xyz_filepath)}' to MOL...", 1, state)
        
        process = subprocess.run(conversion_command, check=False, capture_output=True, text=True, timeout=60) # Added timeout
        
        if process.returncode != 0:
            _print_verbose(f"  Open Babel conversion failed for '{os.path.basename(xyz_filepath)}'.", 1, state)
            # Only print detailed stdout/stderr if verbosity is high
            if state.verbosity_level >= 2:
                _print_verbose(f"  STDOUT (first 5 lines):\n{_format_stream_output(process.stdout, max_lines=5, prefix='    ')}", 2, state)
                _print_verbose(f"  STDERR (first 5 lines):\n{_format_stream_output(process.stderr, max_lines=5, prefix='    ')}", 2, state)
            return False

        if os.path.exists(mol_filepath):
            return True
        else:
            _print_verbose(f"  Open Babel conversion failed to create '{os.path.basename(mol_filepath)}'. Output file not found after command.", 1, state)
            return False

    except FileNotFoundError:
        _print_verbose(f"  Error: Open Babel executable '{openbabel_alias}' not found. Please ensure it's installed and in your PATH.", 0, state)
        return False
    except subprocess.TimeoutExpired:
        _print_verbose(f"  Error: Open Babel conversion timed out for '{os.path.basename(xyz_filepath)}'.", 0, state)
        return False
    except Exception as e:
        _print_verbose(f"  An unexpected error occurred during Open Babel conversion for '{os.path.basename(xyz_filepath)}': {e}", 0, state)
        return False

# 2. Output write
def write_simulation_summary(state: SystemState, output_file_handle, xyz_output_file_display_name: str, rless_filename: str, tvse_filename: str):
    """
    Writes a summary of the simulation parameters to the main out file.
    Includes dynamic filenames and lowest energy info.
    """
    # Construct the elemental composition string
    if hasattr(state, 'element_types') and state.element_types:
        # Determine max length of element symbol for alignment
        max_symbol_len = max(len(state.atomic_number_to_symbol.get(z, f"Unk({z})")) for z, _ in state.element_types)
        element_composition_lines = [
            f"   {state.atomic_number_to_symbol.get(atomic_num, f'Unk({atomic_num})'):<{max_symbol_len}} {count:>3}"
            for atomic_num, count in state.element_types
        ]
    else:
        element_composition_lines = ["  (Elemental composition not available)"]

    # Construct the molecular composition string
    if hasattr(state, 'all_molecule_definitions') and state.all_molecule_definitions and \
       hasattr(state, 'molecules_to_add') and state.molecules_to_add:
        # Determine max length of molecule instance name for alignment
        max_mol_name_len = max(
            len(state.all_molecule_definitions[mol_def_idx].label)
            for mol_def_idx in state.molecules_to_add
            if mol_def_idx < len(state.all_molecule_definitions)
        )
        molecular_composition_lines = []
        for i, mol_def_idx in enumerate(state.molecules_to_add):
            if mol_def_idx < len(state.all_molecule_definitions):
                mol_def = state.all_molecule_definitions[mol_def_idx]
                
                atom_symbols_list = [
                    state.atomic_number_to_symbol.get(atom_data[0], '')
                    for atom_data in mol_def.atoms_coords
                ]
                molecular_formula = get_molecular_formula_string(atom_symbols_list)
                
                molecule_instance_name = mol_def.label # This is the label from the input, e.g., "water1", "water2"
                
                molecular_composition_lines.append(f"  {molecule_instance_name:<{max_mol_name_len}} {molecular_formula}")
            else:
                molecular_composition_lines.append(f"  mol{i+1:<5} (Definition not found for index {mol_def_idx})")
    else:
        molecular_composition_lines = ["  (No molecules specified for addition or 'molecules_to_add' is empty/missing)"]

    # Determine the message for energy evaluation
    energy_eval_message = ""
    output_config_message = ""
    
    if state.random_generate_config == 0: # Random configuration generation mode
        energy_eval_message = "** Energy will not be evaluated **"
        output_config_message = f"Will produce {state.num_random_configs:>2} random configurations"
    elif state.random_generate_config == 1: # Annealing mode
        energy_eval_message = "** Energy will be evaluated **"
        if state.quenching_routine == 1:
            output_config_message = f"Linear quenching route.\n  To = {state.linear_temp_init:.1f} K    dT = {state.linear_temp_decrement:.1f}     nT = {state.linear_num_steps} steps"
        elif state.quenching_routine == 2:
            # Adjusted spacing and added newline
            output_config_message = f"Geometrical quenching route.\n  To = {state.geom_temp_init:.1f} K  %dism = {(1.0 - state.geom_temp_factor) * 100:.1f} %  nT = {state.geom_num_steps} steps"

    # Print centered headers
    centered_header1 = "Annealing Simulado Con Energía Cuántica (ASCEC)".center(60)
    centered_header2 = "* ASCEC-V04: Jun-2025 *".center(60)

    # Write to the file handle
    print("=" * 60 + "\n", file=output_file_handle)
    print(centered_header1, file=output_file_handle)
    print(centered_header2 + "\n", file=output_file_handle)
    print("=" * 60 + "\n", file=output_file_handle) # New separator
    print("Elemental composition of the system:", file=output_file_handle)
    for line in element_composition_lines:
        print(line, file=output_file_handle)
    print(f"There are a total of {state.natom:>2} nuclei", file=output_file_handle)
    # Changed spacing to one space
    print(f"\nCube's length = {state.cube_length:.2f} A", file=output_file_handle) 
    if hasattr(state, 'max_molecular_extent') and state.max_molecular_extent > 0:
        print(f"Largest molecular extent: {state.max_molecular_extent:.2f} A", file=output_file_handle) 
        print(f"Suggested length = {state.max_molecular_extent + 2 * 8.0:.1f} A", file=output_file_handle) 
    
    print("\nNumber of molecules:", state.num_molecules, file=output_file_handle)
    print("\nMolecular composition", file=output_file_handle)
    for line in molecular_composition_lines:
        print(line, file=output_file_handle)

    # Changed spacing to one space
    print(f"\nMaximum displacement of each mass center = {state.max_displacement_a:.2f} A", file=output_file_handle) 
    print(f"Maximum rotation angle = {state.max_rotation_angle_rad:.2f} radians\n", file=output_file_handle) 
    
    # QM program details - formatted as requested
    print(f"Energy calculated with {state.qm_program.capitalize()}", file=output_file_handle)
    print(f" Hamiltonian: {state.qm_method}", file=output_file_handle)
    print(f" Basis set: {state.qm_basis_set}", file=output_file_handle)
    print(f" Charge = {state.charge}   Multiplicity = {state.multiplicity}", file=output_file_handle)
    
    print(f"\nSeed = {state.random_seed:>6}\n", file=output_file_handle)
    
    print(f"{energy_eval_message}\n", file=output_file_handle)
    print(output_config_message, file=output_file_handle)

    if state.random_generate_config == 0: # Only print this line for random config mode
         print(f"\nCoordinates stored in {xyz_output_file_display_name}", file=output_file_handle)

    # Conditionally print the History header based on simulation mode
    if state.random_generate_config == 1: # Only for annealing mode
        print("\n" + "=" * 60 + "\n", file=output_file_handle) # This separator is only for annealing history
        print("History: [T(K), E(u.a.), n-eval, Criterion]", file=output_file_handle)
        print("", file=output_file_handle) # Blank line after history header
    
    output_file_handle.flush()


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
            state.sym[z] = symbol.strip()

# Helper function to get element symbol (used in rless.out and history output)
def get_element_symbol(atomic_number: int) -> str:
    """Retrieves the element symbol for a given atomic number."""
    return ELEMENT_SYMBOLS.get(atomic_number, 'X') # Default to 'X' for unknown/dummy

# 4. Initialize element weights
def initialize_element_weights(state: SystemState):
    """
    Initializes a dictionary mapping atomic numbers to their atomic weights
    and populates the list state.wt. Also populates atomic_number_to_mass.
    """
    state.atomic_number_to_weight = ATOMIC_WEIGHTS.copy()
    state.atomic_number_to_mass = ATOMIC_WEIGHTS.copy() # Populate this for COM calculations

    for atomic_num, weight in ATOMIC_WEIGHTS.items():
        if 0 < atomic_num < len(state.wt):
            state.wt[atomic_num] = weight

# 6. Calculate_mass_center
def calculate_mass_center(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Calculates the center of mass for a set of atoms."""
    total_mass = np.sum(masses)
    if total_mass == 0:
        return np.zeros(3)
    
    return np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass

# 7. Translated draw_molekel.f (draw_molekel) - KEPT FOR REFERENCE, NOT USED DIRECTLY IN CURRENT FLOW
def draw_molekel(natom: int, r_coords: np.ndarray, cycle: int, anneal_flag: int, energy: float, state: SystemState):
    """
    Fortran subroutine draw_molekel: Writes output files for visualization.
    This is a conceptual translation. Actual implementation needs to match Fortran's output format.
    NOTE: This function is currently not directly called in the main loop, as `write_single_xyz_configuration`
    is used instead for more flexible XYZ writing. Kept for reference.
    """
    output_filename = ""
    if anneal_flag == 0: # Random generation mode (mto_ files)
        output_filename = f"mto_{state.jdum}_{cycle:04d}.xyz"
    else: # Annealing mode (result_ files)
        output_filename = f"result_{state.jdum}_{cycle:04d}.xyz"

    full_output_path = os.path.join(state.output_dir, output_filename)

    with open(full_output_path, 'w') as f:
        f.write(f"{natom}\n")
        f.write(f"Energy: {energy:.6f}\n") 
        for i in range(natom):
            atomic_num = state.iznu[i] if hasattr(state, 'iznu') and len(state.iznu) > i else 0
            symbol = state.sym[atomic_num] if atomic_num > 0 and atomic_num < len(state.sym) else "X"
            f.write(f"{symbol} {r_coords[i,0]:10.6f} {r_coords[i,1]:10.6f} {r_coords[i,2]:10.6f}\n")


# 8. Config_molecules
def config_molecules(natom: int, nmo: int, r_coords: np.ndarray, state: SystemState):
    """
    Fortran subroutine config: Configures initial molecular positions and orientations.
    Modifies r_coords in place. Applies PBC.
    Now includes overlap prevention using R_ATOM during initial random configuration.
    Also dynamically adjusts placement ranges if a molecule is difficult to place,
    mimicking the Fortran's aggressive self-correction for steric clashes.
    """
    current_atom_idx = 0
    final_rp_coords = np.zeros_like(r_coords) 

    if not state.all_molecule_definitions:
        _print_verbose("Error: No molecule definitions found for configuration generation. Cannot proceed.", 0, state)
        return

    placed_atoms_data = [] # Stores (atomic_num, x, y, z) of placed atoms

    # Initialize local scaling factors for translation and rotation ranges.
    # These will increase if a molecule is hard to place without overlaps.
    local_translation_range_factor = 1.0
    local_rotation_range_factor = 1.0
    
    # Define the step for increasing the range (e.g., every 5000 attempts)
    RANGE_INCREASE_STEP = 5000 
    # Initialize the next threshold at which the range should increase
    next_increase_threshold_for_this_molecule = RANGE_INCREASE_STEP

    # Factor by which to increase the range (e.g., 10% increase)
    RANGE_INCREASE_FACTOR = 1.1 
    # Maximum factor for the range (e.g., up to 2 times the original range)
    MAX_RANGE_FACTOR = 2.0 

    for i, mol_def_idx in enumerate(state.molecules_to_add): # Iterate through the actual molecules to add
        mol_def = state.all_molecule_definitions[mol_def_idx] # Get the definition from the list

        overlap_found = True
        attempts = 0
        
        # Reset local scales and the next increase threshold for each new molecule
        local_translation_range_factor = 1.0
        local_rotation_range_factor = 1.0
        next_increase_threshold_for_this_molecule = RANGE_INCREASE_STEP

        while overlap_found and attempts < state.MAX_OVERLAP_PLACEMENT_ATTEMPTS:
            attempts += 1
            
            # Dynamically adjust translation and rotation ranges if placement is difficult.
            # This mimics Fortran's 'ds' adjustment to help break out of steric traps.
            if attempts >= next_increase_threshold_for_this_molecule:
                local_translation_range_factor = min(MAX_RANGE_FACTOR, local_translation_range_factor * RANGE_INCREASE_FACTOR)
                local_rotation_range_factor = min(MAX_RANGE_FACTOR, local_rotation_range_factor * RANGE_INCREASE_FACTOR)
                _print_verbose(f"    Increasing placement ranges for molecule {mol_def.label} (instance {i+1}). Current scales: Trans={local_translation_range_factor:.2f}x, Rot={local_rotation_range_factor:.2f}x", 2, state)
                # Move to the next threshold (e.g., from 5000 to 10000, then to 15000, etc.)
                next_increase_threshold_for_this_molecule += RANGE_INCREASE_STEP 

            # Generate a random translation for the molecule's center, scaled by the dynamic factor
            translation = np.random.uniform(-state.xbox/2 * local_translation_range_factor, state.xbox/2 * local_translation_range_factor, size=3)

            # Generate three random Euler angles (yaw, pitch, roll) in radians, scaled by the dynamic factor
            alpha = np.random.uniform(0, 2 * np.pi * local_rotation_range_factor) 
            beta = np.random.uniform(0, 2 * np.pi * local_rotation_range_factor)  
            gamma = np.random.uniform(0, 2 * np.pi * local_rotation_range_factor) 

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

            rotation_matrix = Rz @ Ry @ Rx

            proposed_mol_atoms = [] # (atomic_num, abs_coords_array)
            for atom_data_in_mol in mol_def.atoms_coords: # Corrected variable name here
                atomic_num, x_rel, y_rel, z_rel = atom_data_in_mol # Corrected variable name here
                relative_coords_vector = np.array([x_rel, y_rel, z_rel])
                rotated_coords = np.dot(rotation_matrix, relative_coords_vector)
                abs_coords = rotated_coords + translation
                proposed_mol_atoms.append((atomic_num, abs_coords))

            overlap_found = False
            for prop_atom_num, prop_coords in proposed_mol_atoms:
                for placed_atom_data in placed_atoms_data:
                    placed_atom_num = placed_atom_data[0] 
                    placed_coords = np.array(placed_atom_data[1:]) 

                    distance = np.linalg.norm(prop_coords - placed_coords)

                    radius1 = state.R_ATOM.get(prop_atom_num, 0.5)
                    radius2 = state.R_ATOM.get(placed_atom_num, 0.5)

                    min_distance_allowed = (radius1 + radius2) * state.OVERLAP_SCALE_FACTOR

                    if distance < min_distance_allowed and distance > 1e-4:
                        overlap_found = True
                        # If overlap is found, this molecule's placement attempt fails.
                        # The while loop will continue to retry placing *this same molecule*
                        # with potentially increased ranges until a non-overlapping position is found
                        # or MAX_OVERLAP_PLACEMENT_ATTEMPTS is reached.
                        break 
                if overlap_found:
                    break 
            
        if overlap_found: 
            _print_verbose(f"Warning: Could not find non-overlapping placement for molecule {mol_def.label} (instance {i+1}) after {state.MAX_OVERLAP_PLACEMENT_ATTEMPTS} attempts. Placing it anyway, may cause QM errors.", 1, state)

        for atom_data in proposed_mol_atoms:
            atomic_num, abs_coords = atom_data
            final_rp_coords[current_atom_idx, :] = abs_coords
            placed_atoms_data.append((atomic_num, abs_coords[0], abs_coords[1], abs_coords[2]))
            state.iznu[current_atom_idx] = atomic_num 
            current_atom_idx += 1
    
    state.rp[:] = final_rp_coords
    
    # PERIODIC BOUNDARY CONDITIONS (PBC) - commented out as per Fortran's implied behavior
    # state.rp[:] = state.rp - state.xbox * np.floor(state.rp / state.xbox)

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
    This function creates the *initial* configuration from the template molecule.
    """
    if not hasattr(state, 'coords_per_molecule') or state.coords_per_molecule is None or \
       not hasattr(state, 'atomic_numbers_per_molecule') or not state.atomic_numbers_per_molecule or \
       state.natom_per_molecule == 0:
        _print_verbose("Error: Molecular definitions (coords_per_molecule, atomic_numbers_per_molecule) not loaded from input file.", 0, state)
        _print_verbose("Please ensure the 'Molecule Definition' section in your input file is correct and parsed by read_input_file.", 0, state)
        raise ValueError("Cannot initialize coordinates: Molecular definition is missing or invalid.")

    total_atoms = state.num_molecules * state.natom_per_molecule
    state.natom = total_atoms 

    rp = np.zeros((total_atoms, 3), dtype=np.float64)
    iznu = [0] * total_atoms

    template_coords = state.coords_per_molecule
    template_atomic_numbers = state.atomic_numbers_per_molecule

    template_min_coords = np.min(template_coords, axis=0)
    template_max_coords = np.max(template_coords, axis=0)
    
    template_geometric_center = (template_min_coords + template_max_coords) / 2.0

    box_center = state.cube_length / 2.0 * np.ones(3) 

    initial_centering_offset = box_center - template_geometric_center

    for i in range(state.num_molecules):
        start_idx = i * state.natom_per_molecule
        end_idx = start_idx + state.natom_per_molecule

        rp[start_idx:end_idx, :] = template_coords + initial_centering_offset

        iznu[start_idx:end_idx] = template_atomic_numbers
    
    # Populate the initial iznu array for the state object
    state.iznu = iznu 
    state.rp = rp

    # Now, randomly translate and rotate molecules from this superimposed state
    # This calls the more general config_molecules that includes overlap checking
    config_molecules(state.natom, state.num_molecules, state.rp, state)

    return state.rp, state.iznu

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

    lines_iterator = iter(lines) 

    # PHASE 1: Read Fixed Configuration Parameters (Lines 1-12)
    config_lines_parsed = 0 

    # We expect 12 lines for fixed configuration parameters (no alpha)
    while config_lines_parsed < 12: # Changed from 13 to 12
        try:
            raw_line = next(lines_iterator)
        except StopIteration:
            raise EOFError(f"Unexpected end of input while reading configuration parameters. Expected 12 lines, but found only {config_lines_parsed}.")
        
        line_num = lines.index(raw_line) + 1 
        line = clean_line(raw_line)

        if not line: 
            continue

        parts = line.split()
        if not parts: 
            continue

        # Parsing config lines based on count
        if config_lines_parsed == 0: # Line 1: Simulation Mode & Number of Config
            state.random_generate_config = int(parts[0])
            state.num_random_configs = int(parts[1])
            
            if state.random_generate_config == 0: 
                state.ivalE = 0 
                state.mto = state.num_random_configs 
            elif state.random_generate_config == 1: 
                state.ivalE = 1 
                state.mto = 0   

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
            # Parse the geometric factor. If quenching_routine is 2, interpret as percentage decrease.
            if state.quenching_routine == 2:
                # User specified "decrease a 5%". So 5.0 in input means 0.95 factor.
                factor_percentage = float(parts[1])
                if not (0.0 < factor_percentage < 100.0):
                    raise ValueError(f"Error parsing geometric temperature factor on line {line_num}: Expected a percentage between 0 and 100, got '{parts[1]}'.")
                state.geom_temp_factor = 1.0 - (factor_percentage / 100.0)
            else:
                # If not geometric, just read it directly (though it shouldn't be used).
                state.geom_temp_factor = float(parts[1])
            state.geom_num_steps = int(parts[2])

        elif config_lines_parsed == 5: # Line 6: Maximum Monte Carlo Cycles per T
            state.max_cycle = int(parts[0])
            state.maxstep = state.max_cycle # Initialize maxstep with the initial max_cycle from input

        elif config_lines_parsed == 6: # Line 7: Maximum Displacement & Rotation
            state.max_displacement_a = float(parts[0])
            state.max_rotation_angle_rad = float(parts[1])
            state.ds = state.max_displacement_a
            state.dphi = state.max_rotation_angle_rad

        elif config_lines_parsed == 7: # Line 8: QM Program Index & Alias (e.g., "1 g09")
            state.ia = int(parts[0])      
            state.qm_program = parts[1]   
            state.alias = parts[1]        
            state.jdum = state.alias      

        elif config_lines_parsed == 8: # Line 9: Hamiltonian & Basis Set (e.g., "pm3 zdo")
            state.qm_method = parts[0]      
            state.qm_basis_set = parts[1]   

        elif config_lines_parsed == 9:      # Line 10: nprocs & maxmemory
            state.qm_nproc = int(parts[0])   
            # Only set qm_memory if a second part is explicitly provided in the input file.
            # Otherwise, it remains None, and the QM program will decide.
            if len(parts) > 1: 
                 state.qm_memory = parts[1]
            else:
                 state.qm_memory = None # Ensure it's None if not provided

        elif config_lines_parsed == 10:     # Line 11: Charge & Spin Multiplicity
            state.charge = int(parts[0])
            state.multiplicity = int(parts[1]) 

        elif config_lines_parsed == 11:     # Line 12: Number of Molecules (was Line 13)
            try:
                state.num_molecules = int(parts[0])
            except ValueError:
                raise ValueError(f"Error parsing 'Number of Molecules' on line {line_num}: Expected an integer, but found '{parts[0]}'. "
                                 "Please ensure line 12 of your input file contains the total number of molecules.")
            state.nmo = state.num_molecules
        else:
            _print_verbose(f"Warning: Unexpected configuration line at index {config_lines_parsed}. Line: {line}", 1, state)

        config_lines_parsed += 1 

    # PHASE 2: Read Molecule Definitions
    reading_molecule = False
    current_molecule_num_atoms_expected = 0
    current_molecule_label = ""
    current_molecule_atoms: List[Tuple[int, float, float, float]] = []
    atoms_read_in_current_molecule = 0

    for raw_line in lines_iterator: 
        line_num = lines.index(raw_line) + 1 
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
                
                current_molecule_num_atoms_expected = 0
                current_molecule_label = ""
                current_molecule_atoms = []
                atoms_read_in_current_molecule = 0
                
                reading_molecule = True 
                continue 

            else: # Found '*' and was NOT reading a molecule -> this must be the very first '*' opening the first molecule
                reading_molecule = True
                continue 

        elif reading_molecule: # We are inside a molecule block 
            if current_molecule_num_atoms_expected == 0: 
                try:
                    current_molecule_num_atoms_expected = int(parts[0])
                except ValueError:
                    raise ValueError(f"Error parsing molecule block near line {line_num}: Expected number of atoms, got '{parts[0]}'.")
            elif not current_molecule_label: 
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
        else: 
            _print_verbose(f"Warning: Unexpected content '{raw_line.strip()}' found outside of defined blocks near line {line_num}.", 1, state)
            continue 
            
    if reading_molecule and current_molecule_num_atoms_expected > 0: 
        if current_molecule_num_atoms_expected == atoms_read_in_current_molecule:
            molecule_definitions.append(
                MoleculeData(current_molecule_label, current_molecule_num_atoms_expected, current_molecule_atoms)
            )
        else:
            raise ValueError(f"Error: Last molecule block not properly closed or incomplete for molecule {current_molecule_label}.")

    if len(molecule_definitions) != state.num_molecules:
        raise ValueError(f"Error: Number of molecules defined in input ({len(molecule_definitions)}) does not match expected ({state.num_molecules}).")

    state.all_molecule_definitions = molecule_definitions
    # Populate molecules_to_add with indices corresponding to the order they were defined
    # This ensures that each instance (water1, water2, etc.) refers to its specific definition.
    state.molecules_to_add = list(range(state.num_molecules))

    state.natom = sum(mol.num_atoms for mol in molecule_definitions)
    
    if molecule_definitions: 
        # The template molecule for initial placement is still the first one defined
        first_molecule_data = molecule_definitions[0]
        state.natom_per_molecule = first_molecule_data.num_atoms
        state.molecule_label = first_molecule_data.label 

        coords_list = [[atom[1], atom[2], atom[3]] for atom in first_molecule_data.atoms_coords]
        state.coords_per_molecule = np.array(coords_list, dtype=np.float64)

        state.atomic_numbers_per_molecule = [atom[0] for atom in first_molecule_data.atoms_coords]
        
        # Calculate max molecular extent for suggestion
        min_coords_all_mols = np.min(state.coords_per_molecule, axis=0)
        max_coords_all_mols = np.max(state.coords_per_molecule, axis=0)
        max_extent = np.max(max_coords_all_mols - min_coords_all_mols)
        state.max_molecular_extent = max_extent
    else:
        raise ValueError("No molecule definitions found in the input file.")

    state.iznu = [0] * state.natom 
    
    unique_elements = {}
    atom_idx = 0
    for mol_def_idx in state.molecules_to_add: # Iterate through the *indices* of molecules to add
        mol_data = molecule_definitions[mol_def_idx] # Get the definition based on its index
        for atomic_num, _, _, _ in mol_data.atoms_coords:
            state.iznu[atom_idx] = atomic_num
            unique_elements[atomic_num] = unique_elements.get(atomic_num, 0) + 1
            atom_idx += 1

    state.num_elements_defined = len(unique_elements) 
    state.element_types = [(z, unique_elements[z]) for z in sorted(list(unique_elements.keys()))]
    state.iz_types = [z for z, _ in state.element_types] 
    state.nnu_types = [count for _, count in state.element_types] 

    initialize_element_symbols(state)
    initialize_element_weights(state) # This populates atomic_number_to_weight and atomic_number_to_mass

    state.imolec = [0] * (state.num_molecules + 1)
    current_atom_for_imolec = 0
    for i, mol_def_idx in enumerate(state.molecules_to_add): # Iterate through instances to add
        mol_data = molecule_definitions[mol_def_idx] # Get the definition
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
        "energy_regex": r"FINAL SINGLE POINT ENERGY:\s*([-\d.]+)\s*Eh", # More precise for ORCA
        "termination_string": "ORCA TERMINATED NORMALLY",
    },
}

# 13. Calculate energy function
def calculate_energy(coords: np.ndarray, atomic_numbers: List[int], state: SystemState, run_dir: str) -> Tuple[float, int]:
    """
    Calculates the energy of the given configuration using the external QM program.
    Returns the energy and a status code (1 for success, 0 for failure).
    Cleans up QM input/output/checkpoint files immediately after execution.
    """
    state.qm_call_count += 1 # Increment total QM calls
    call_id = state.qm_call_count
    
    qm_input_filename = f"qm_input_{call_id}{QM_PROGRAM_DETAILS[state.ia]['input_ext']}"
    qm_output_filename = f"qm_output_{call_id}{QM_PROGRAM_DETAILS[state.ia]['output_ext']}"
    
    # Checkpoint file naming convention, might vary per program (e.g., Gaussian uses .chk)
    qm_chk_filename = f"qm_chk_{call_id}.chk" if state.qm_program == "gaussian" else None

    qm_input_path = os.path.join(run_dir, qm_input_filename)
    qm_output_path = os.path.join(run_dir, qm_output_filename)
    qm_chk_path = os.path.join(run_dir, qm_chk_filename) if qm_chk_filename else None

    energy = 0.0
    status = 0 # 0 for failure, 1 for success
    
    temp_files_to_clean = [qm_input_path, qm_output_path]
    if qm_chk_path:
        temp_files_to_clean.append(qm_chk_path)

    try:
        # Generate QM input file
        with open(qm_input_path, 'w') as f:
            if state.qm_program == "gaussian":
                if qm_chk_path: f.write(f"%chk={os.path.basename(qm_chk_path)}\n") # Only filename for %chk
                # Only write %mem if qm_memory was explicitly provided in the input file
                if state.qm_memory: f.write(f"%mem={state.qm_memory}\n")
                if state.qm_nproc: f.write(f"%nproc={state.qm_nproc}\n")
                
                # Changed from / to space as requested: pm3 zdo
                f.write(f"# {state.qm_method} {state.qm_basis_set}\n\n") 
                f.write("ASCEC QM Calculation\n\n")
                f.write(f"{state.charge} {state.multiplicity}\n")
                for i in range(state.natom):
                    symbol = state.atomic_number_to_symbol.get(atomic_numbers[i], "X")
                    f.write(f"{symbol} {coords[i, 0]:.6f} {coords[i, 1]:.6f} {coords[i, 2]:.6f}\n")
                f.write("\n") 
                if state.qm_additional_keywords: f.write(f"{state.qm_additional_keywords}\n")
                f.write("\n") 
            elif state.qm_program == "orca":
                # Changed from / to space if a basis set is present, otherwise no change.
                # ORCA typically uses space for keywords unless it's a specific block definition.
                f.write(f"! {state.qm_method} {state.qm_basis_set}\n") 
                if state.qm_additional_keywords: f.write(f"! {state.qm_additional_keywords}\n")
                
                # Only write %maxcore if qm_memory was explicitly provided in the input file
                if state.qm_memory:
                    mem_val = state.qm_memory.replace('GB', '').replace('MB', '')
                    f.write(f"%maxcore {mem_val}\n") 
                if state.qm_nproc: f.write(f"%pal nprocs {state.qm_nproc} end\n")
                
                f.write("* xyz {state.charge} {state.multiplicity}\n")
                for i in range(state.natom):
                    symbol = state.atomic_number_to_symbol.get(atomic_numbers[i], "X")
                    f.write(f"{symbol} {coords[i, 0]:.6f} {coords[i, 1]:.6f} {coords[i, 2]:.6f}\n")
                f.write("*\n")
            else:
                raise ValueError(f"QM input generation not implemented for program '{state.qm_program}'")
    except IOError as e:
        _print_verbose(f"Error writing QM input file {qm_input_path}: {e}", 0, state)
        return 0.0, 0 
    except ValueError as e:
        _print_verbose(f"Error in QM input generation: {e}", 0, state)
        return 0.0, 0

    # Determine QM command
    qm_exe = QM_PROGRAM_DETAILS[state.ia]["default_exe"]
    if state.qm_program == "gaussian":
        qm_command = f"{qm_exe} {qm_input_filename} {qm_output_filename}"
    elif state.qm_program == "orca":
        qm_command = f"{qm_exe} {qm_input_filename} > {qm_output_filename}"
    else:
        _print_verbose(f"Error: Unsupported QM program '{state.qm_program}' for command execution.", 0, state)
        return 0.0, 0

    try:
        process = subprocess.run(qm_command, shell=True, capture_output=True, text=True, cwd=run_dir, check=False)
        
        # Check for non-zero exit code first
        if process.returncode != 0:
            _print_verbose(f"'{state.qm_program}' exited with non-zero status: {process.returncode}.", 1, state)
            # Only print detailed stdout/stderr if verbosity is high
            if state.verbosity_level >= 2:
                _print_verbose(f"  STDOUT (first 10 lines):\n{_format_stream_output(process.stdout)}", 2, state)
                _print_verbose(f"  STDERR (first 10 lines):\n{_format_stream_output(process.stderr)}", 2, state)
            status = 0 
        elif not os.path.exists(qm_output_path):
            _print_verbose(f"QM output file '{qm_output_path}' was not generated.", 1, state)
            status = 0 
        else:
            with open(qm_output_path, 'r') as f:
                output_content = f.read()
            
            # Check for normal termination string
            if QM_PROGRAM_DETAILS[state.ia]["termination_string"] not in output_content:
                _print_verbose(f"QM program '{state.qm_program}' did not terminate normally for config {call_id}.", 1, state)
                status = 0
            else:
                match = re.search(QM_PROGRAM_DETAILS[state.ia]["energy_regex"], output_content)
                if match:
                    energy = float(match.group(1))
                    status = 1
                else:
                    _print_verbose(f"Could not find energy in {state.qm_program} output file: {qm_output_path}", 1, state)
                    status = 0
    
    except Exception as e:
        _print_verbose(f"An error occurred during QM calculation or parsing: {e}", 0, state)
        status = 0
    finally:
        # Immediate cleanup of QM files for this specific call
        for fpath in temp_files_to_clean:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError as e:
                    _print_verbose(f"  Error cleaning up {os.path.basename(fpath)}: {e}", 0, state)
    return energy, status

# Helper function to format and limit stream output (stdout/stderr)
def _format_stream_output(stream_content, max_lines=10, prefix="  "):
    if not stream_content:
        return "[No output]"
    
    lines = stream_content.splitlines()
    output_str = ""
    for i, line in enumerate(lines):
        if i < max_lines:
            output_str += f"{prefix}{line}\n"
        else:
            output_str += f"{prefix}... ({len(lines) - max_lines} more lines)\n"
            break
    return output_str

# 14. xyz confifuration 
def write_single_xyz_configuration(file_handle, natom, rp, iznu, energy, config_idx, 
                                   atomic_number_to_symbol, random_generate_config_mode, 
                                   remark="", include_dummy_atoms=False, state=None): 
    """
    Writes a single XYZ configuration to a file handle.
    Optionally includes dummy 'X' atoms for the box corners.
    The 'remark' argument provides an optional string for the comment line.
    The output format of the second line changes based on 'random_generate_config_mode'.
    
    Args:
        file_handle (io.TextIOWrapper): The open file handle to write to.
        natom (int): Total number of actual atoms in the configuration.
        rp (np.ndarray): (N, 3) array of atomic coordinates.
        iznu (List[int]): List of atomic numbers.
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
    
    L = state.xbox if include_dummy_atoms and state else 0.0 

    box_corners = np.array([
        [0, 0, 0], [L, 0, 0], [0, L, 0], [0, 0, L],
        [L, L, 0], [L, 0, L], [0, L, L], [L, L, L]
    ])
    dummy_atom_count = len(box_corners) 

    total_atoms_in_frame = natom
    if include_dummy_atoms:
        total_atoms_in_frame += dummy_atom_count

    file_handle.write(f"{total_atoms_in_frame}\n")
    
    # Updated comment line format with | separators and fixed BoxL format
    base_comment_line = f"Configuration: {config_idx} | E = {energy:.8f} a.u."
    if include_dummy_atoms:
        if L > 0: 
            base_comment_line += f" | BoxL={L:.1f} A ({DUMMY_ATOM_SYMBOL} box markers)" 
    elif random_generate_config_mode == 1: # Only add T for annealing mode, not for random mode
        base_comment_line += f" | T = {state.current_temp:.1f} K" 

    file_handle.write(f"{base_comment_line}\n")
    
    for i in range(natom):
        symbol = atomic_number_to_symbol.get(iznu[i], 'X') 
        file_handle.write(f"{symbol: <3} {rp[i, 0]: 12.6f} {rp[i, 1]: 12.6f} {rp[i, 2]: 12.6f}\n")

    if include_dummy_atoms:
        for coords in box_corners:
            file_handle.write(f"{DUMMY_ATOM_SYMBOL: <3} {coords[0]: 12.6f} {coords[1]: 12.6f} {coords[2]: 12.6f}\n")

    file_handle.flush()

# New wrapper function for writing accepted XYZ configurations
def write_accepted_xyz(prefix: str, config_number: int, energy: float, temp: float, state: SystemState, is_initial: bool = False):
    """
    Writes accepted XYZ configurations (both normal and with box) to files.
    This wraps write_single_xyz_configuration.
    
    Args:
        prefix (str): Base filename prefix (e.g., "result_123456").
        config_number (int): The sequential number for this accepted configuration.
        energy (float): The energy of the configuration.
        temp (float): The temperature at which the configuration was accepted.
        state (SystemState): The current SystemState object.
        is_initial (bool): True if this is the very first accepted configuration.
    """
    xyz_path = f"{prefix}.xyz"
    box_xyz_path = f"{prefix.replace('mto_', 'mtobox_').replace('result_', 'resultbox_')}.xyz" # Corrected replacement for box name

    remark = "" # Removed specific remarks, now handled by write_single_xyz_configuration's internal logic

    # Write to the ORIGINAL XYZ file handle (no dummy atoms)
    with open(xyz_path, 'a') as f_xyz:
        write_single_xyz_configuration(
            f_xyz,
            state.natom,
            state.rp + (state.xbox / 2.0), # Shift coords for visualization
            state.iznu,
            energy, 
            config_number, # Use the sequential config_number here
            state.atomic_number_to_symbol,
            state.random_generate_config, 
            remark=remark, # Empty remark as formatting is handled internally
            include_dummy_atoms=False, 
            state=state 
        )                
    
    # Conditionally write to the BOX XYZ COPY file handle (with dummy atoms)
    if CREATE_BOX_XYZ_COPY:
        with open(box_xyz_path, 'a') as f_box_xyz:
            write_single_xyz_configuration(
                f_box_xyz,
                state.natom,
                state.rp + (state.xbox / 2.0), # Shift coords for visualization
                state.iznu,
                energy, # Energy is 0.0 as it's not evaluated
                config_number, # Use the sequential config_number here
                state.atomic_number_to_symbol,
                state.random_generate_config, 
                remark=remark, # Empty remark as formatting is handled internally
                include_dummy_atoms=True, 
                state=state 
            )

# 15. Propose Move Function (No longer used for full system randomization in annealing)
# Keeping this function definition for reference if a future iterative single-molecule
# movement strategy is desired.
def propose_move(state: SystemState, current_rp: np.ndarray, current_imolec: List[int]) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    Proposes a random translation and rotation for a single molecule,
    applying Fortran's 'trans' (CM bounce) and 'rotac' (Euler angle rotation) logic.
    Returns the new full system coordinates, the coordinates of the moved molecule,
    the index of the moved molecule, and the move type.
    """
    # Randomly select a molecule to move
    molecule_idx = np.random.randint(0, state.num_molecules)

    start_atom_idx = current_imolec[molecule_idx] 
    end_atom_idx = current_imolec[molecule_idx + 1] 
    
    # Prepare proposed_rf as a copy of current_rp
    proposed_rp_full_system = np.copy(current_rp)

    # Get the current coordinates of the selected molecule
    mol_coords_current = current_rp[start_atom_idx:end_atom_idx, :]
    
    # Calculate current center of mass for the selected molecule
    # Ensure state.iznu and state.atomic_number_to_mass are correctly populated
    mol_atomic_numbers = [state.iznu[i] for i in range(start_atom_idx, end_atom_idx)]
    mol_masses = np.array([state.atomic_number_to_mass.get(anum, 1.0) 
                           for anum in mol_atomic_numbers])
    
    current_rcm = calculate_mass_center(mol_coords_current, mol_masses)

    # --- Translation Part (matching Fortran's 'trans' subroutine) ---
    # Generate random displacement vector for CM, range [-max_displacement_a, +max_displacement_a]
    random_displacement_vector = (np.random.rand(3) - 0.5) * 2.0 * state.max_displacement_a

    half_xbox = state.cube_length / 2.0 
    
    new_rcm_after_translation = np.copy(current_rcm)
    actual_atom_displacement = np.copy(random_displacement_vector)

    for dim in range(3):
        new_rcm_after_translation[dim] += random_displacement_vector[dim]

        if np.abs(new_rcm_after_translation[dim]) > half_xbox:
            new_rcm_after_translation[dim] -= 2.0 * random_displacement_vector[dim] 
            actual_atom_displacement[dim] = -random_displacement_vector[dim] 
    
    proposed_rp_full_system[start_atom_idx:end_atom_idx, :] += actual_atom_displacement

    # --- Rotation Part (matching Fortran's 'rotac' subroutine with Euler angles) ---
    # Generate random Euler angles (alpha, beta, gamma) for Z, Y, X rotations respectively
    alpha_rot = (np.random.rand() - 0.5) * 2.0 * state.max_rotation_angle_rad 
    beta_rot = (np.random.rand() - 0.5) * 2.0 * state.max_rotation_angle_rad  
    gamma_rot = (np.random.rand() - 0.5) * 2.0 * state.max_rotation_angle_rad 

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(gamma_rot), -np.sin(gamma_rot)],
        [0, np.sin(gamma_rot), np.cos(gamma_rot)]
    ], dtype=np.float64)

    Ry = np.array([
        [np.cos(beta_rot), 0, np.sin(beta_rot)],
        [0, 1, 0],
        [-np.sin(beta_rot), 0, np.cos(beta_rot)]
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(alpha_rot), -np.sin(alpha_rot), 0],
        [np.sin(alpha_rot), np.cos(alpha_rot), 0],
        [0, 0, 1]
    ], dtype=np.float64)

    rotation_matrix = Rz @ Ry @ Rx

    mol_coords_after_translation = proposed_rp_full_system[start_atom_idx:end_atom_idx, :]
    mol_coords_relative_to_cm = mol_coords_after_translation - new_rcm_after_translation
    
    rotated_relative_coords = (rotation_matrix @ mol_coords_relative_to_cm.T).T

    proposed_rp_full_system[start_atom_idx:end_atom_idx, :] = rotated_relative_coords + new_rcm_after_translation

    # Return the full system coordinates, the coordinates of the moved molecule (for potential debugging),
    # the index of the moved molecule, and a string indicating the move type.
    return proposed_rp_full_system, proposed_rp_full_system[start_atom_idx:end_atom_idx, :], molecule_idx, "translate_rotate"

# --- Subroutine: trans (Translates molecules randomly) ---
def trans(imo: int, r_coords: np.ndarray, rel_coords: np.ndarray, rcm_coords: np.ndarray, ds_val: float, state: SystemState):
    """
    Translates a molecule (imo) randomly.
    r_coords: Current atom coordinates (r in Fortran) - modified in place
    rel_coords: Relative atom coordinates (rel in Fortran) - modified in place
    rcm_coords: Center of mass coordinates for all molecules (rcm in Fortran) - modified in place
    ds_val: Maximum displacement parameter
    state: SystemState object for accessing imolec and xbox
    """
    for i in range(3): # Loop over x, y, z coordinates
        z = state.ran0_method() # Use state's ran0_method
        is_val = -1 if z < 0.5 else 1
        s = is_val * state.ran0_method() * ds_val

        # Store original rcm for potential rollback if out of bounds
        # original_rcm_i = rcm_coords[imo-1, i] # Not needed with current config_move logic

        # Move CM
        rcm_coords[imo, i] += s # imo is 0-indexed here

        # Apply periodic boundary conditions for CM
        if np.abs(rcm_coords[imo, i]) > state.xbox / 2.0: # Use state's xbox
            rcm_coords[imo, i] -= 2.0 * s # Effectively wraps around
            s = -s # Invert displacement for atom translation

        # Translate individual atoms within the molecule
        for inu in range(state.imolec[imo], state.imolec[imo+1]): # imo is 0-indexed
            r_coords[inu, i] += s
            rel_coords[inu - state.imolec[imo], i] = r_coords[inu, i] - rcm_coords[imo, i] # Update relative position

# --- Subroutine: rotac (Rotates molecules randomly) ---
def rotac(imo: int, r_coords: np.ndarray, rel_coords: np.ndarray, rcm_coords: np.ndarray, dphi_val: float, state: SystemState):
    """
    Rotates a molecule (imo) randomly around its center of mass.
    r_coords: Current atom coordinates (r in Fortran) - modified in place
    rel_coords: Relative atom coordinates (rel in Fortran) - modified in place
    rcm_coords: Center of mass coordinates for all molecules (rcm in Fortran) - used for reference
    dphi_val: Maximum rotation parameter
    state: SystemState object for accessing imolec and ran0_method
    """
    ix = -1 if state.ran0_method() < 0.5 else 1
    iy = -1 if state.ran0_method() < 0.5 else 1
    iz = -1 if state.ran0_method() < 0.5 else 1

    xa = ix * state.ran0_method() * dphi_val # Rotation angle around X
    ya = iy * state.ran0_method() * dphi_val # Rotation angle around Y
    za = iz * state.ran0_method() * dphi_val # Rotation angle around Z

    cos_xa = np.cos(xa)
    sin_xa = np.sin(xa)
    cos_ya = np.cos(ya)
    sin_ya = np.sin(ya)
    cos_za = np.cos(za)
    sin_za = np.sin(za)

    # Combined rotation matrix for ZYX intrinsic rotations (as implied by Fortran code)
    # The Fortran rotation matrix seems to be slightly unusual, let's replicate it directly.
    # It appears to be a specific sequence of rotations around X, Y, Z.
    # The provided Fortran code's rotation matrix corresponds to a rotation by xa around x,
    # then ya around y, then za around z (intrinsic rotations).
    # For accuracy, we'll use the exact matrix as written in Fortran.

    # Atom indices for the current molecule
    mol_start_idx = state.imolec[imo]
    mol_end_idx = state.imolec[imo+1]

    for i_rel, inu in enumerate(range(mol_start_idx, mol_end_idx)):
        X_orig = rel_coords[i_rel, 0]
        Y_orig = rel_coords[i_rel, 1]
        Z_orig = rel_coords[i_rel, 2]

        # Apply Fortran's specific rotation matrix elements
        rel_coords[i_rel, 0] = (
            X_orig * (cos_za * cos_ya) +
            Y_orig * (-cos_za * sin_ya * sin_xa + sin_za * cos_xa) +
            Z_orig * (cos_za * sin_ya * cos_xa + sin_za * sin_xa)
        )
        rel_coords[i_rel, 1] = (
            X_orig * (-sin_za * cos_ya) +
            Y_orig * (sin_za * sin_ya * sin_xa + cos_za * cos_xa) +
            Z_orig * (-sin_za * sin_ya * cos_xa + cos_za * sin_xa)
        )
        rel_coords[i_rel, 2] = (
            X_orig * (-sin_ya) +
            Y_orig * (-cos_ya * sin_xa) +
            Z_orig * (cos_ya * cos_xa)
        )

        # Update absolute coordinates based on new relative coordinates and CM
        for k in range(3):
            r_coords[inu, k] = rel_coords[i_rel, k] + rcm_coords[imo, k] # imo is 0-indexed

# --- Main Subroutine: config_move ---
def config_move(state: SystemState):
    """
    Generates a new configuration by randomly translating and rotating ALL molecules,
    with collision detection and re-attempt logic for the entire system.
    Modifies state.rp and state.rcm in place.
    """
    # Keep track of the original state for rollback if the entire proposed configuration fails
    original_rp_full_system = np.copy(state.rp)
    original_rcm_full_system = np.copy(state.rcm)

    # Initialize local ds/dphi for this *attempt* to generate a valid full configuration
    # These will be reset for each full configuration attempt, and increased if collisions persist.
    local_ds = state.ds
    local_dphi = state.dphi
    
    attempts_full_config = 0 # Counter for attempts to generate a valid full configuration

    # Max attempts for a full configuration to be generated without collision
    MAX_FULL_CONFIG_ATTEMPTS = 100000 # Similar to MAX_OVERLAP_PLACEMENT_ATTEMPTS but for the whole system

    while True: # Outer loop for generating a valid full system configuration
        attempts_full_config += 1
        
        # If too many attempts, increase the perturbation range for all molecules
        if attempts_full_config > MAX_FULL_CONFIG_ATTEMPTS:
            _print_verbose("\n** Warning **", 1, state)
            _print_verbose(f"More than {attempts_full_config-1} attempts to generate a valid full configuration.", 1, state)
            _print_verbose("Maximum displacement and rotation parameters will be enlarged by 20%", 1, state)
            local_ds *= 1.2
            local_dphi *= 1.2
            if local_ds > state.xbox:
                local_ds = state.xbox
                _print_verbose("Maximum value for ds = xbox reached", 1, state)
            if local_dphi > 2 * np.pi: # Max rotation is 2pi
                local_dphi = 2 * np.pi
                _print_verbose("Maximum value for dphi = 2*PI reached", 1, state)
            _print_verbose(f"New values: ds = {local_ds:.2f}, dphi = {local_dphi:.2f}", 1, state)
            _print_verbose("*****************\n", 1, state)
            attempts_full_config = 0 # Reset counter after adjustment

        # Create proposed state for this attempt, starting from the original (last accepted) state
        proposed_rp = np.copy(original_rp_full_system)
        proposed_rcm = np.copy(original_rcm_full_system)

        # Perturb each molecule independently from the original state
        for imo in range(state.num_molecules): # Iterate through all molecules (0-indexed)
            mol_start_idx = state.imolec[imo]
            mol_end_idx = state.imolec[imo+1]
            
            # Calculate current center of mass for the selected molecule
            mol_atomic_numbers = [state.iznu[i] for i in range(mol_start_idx, mol_end_idx)]
            mol_masses = np.array([state.atomic_number_to_mass.get(anum, 1.0) 
                                   for anum in mol_atomic_numbers])
            
            # Calculate CM based on the *current* proposed_rp for this molecule
            current_mol_rcm = calculate_mass_center(proposed_rp[mol_start_idx:mol_end_idx, :], mol_masses)
            
            # Create a temporary array for relative coordinates of the *current molecule*
            mol_rel_coords_temp = proposed_rp[mol_start_idx:mol_end_idx, :] - current_mol_rcm
            
            # Apply translation and rotation to this molecule within the proposed_rp/rcm
            # Note: trans and rotac modify proposed_rp (for the selected molecule's atoms)
            # and proposed_rcm (for the selected molecule's CM) in place.
            trans(imo, proposed_rp, mol_rel_coords_temp, proposed_rcm, local_ds, state) 
            rotac(imo, proposed_rp, mol_rel_coords_temp, proposed_rcm, local_dphi, state) 

        # Now, check for collisions in the *entire proposed configuration*
        collision_detected = False
        for i_mol in range(state.num_molecules):
            for j_mol in range(i_mol + 1, state.num_molecules): # Check unique pairs of molecules
                mol1_start_idx = state.imolec[i_mol]
                mol1_end_idx = state.imolec[i_mol+1]
                mol2_start_idx = state.imolec[j_mol]
                mol2_end_idx = state.imolec[j_mol+1]

                for atom1_idx in range(mol1_start_idx, mol1_end_idx):
                    for atom2_idx in range(mol2_start_idx, mol2_end_idx):
                        dist_sq = np.sum((proposed_rp[atom1_idx, :] - proposed_rp[atom2_idx, :])**2)
                        dist = np.sqrt(dist_sq)

                        atom1_z = state.iznu[atom1_idx]
                        atom2_z = state.iznu[atom2_idx]

                        radius1 = state.R_ATOM.get(atom1_z, 0.5)
                        radius2 = state.R_ATOM.get(atom2_z, 0.5)

                        rmin = (radius1 + radius2) * state.OVERLAP_SCALE_FACTOR

                        if dist < rmin and dist > 1e-4: # Avoid self-collision check for same atom
                            collision_detected = True
                            _print_verbose(f"  Collision detected between atom {atom1_idx} (mol {i_mol+1}) and atom {atom2_idx} (mol {j_mol+1}) in proposed config.", 2, state)
                            break # Exit inner atom2 loop
                    if collision_detected:
                        break # Exit atom1 loop
                if collision_detected:
                    break # Exit j_mol loop
        
        if collision_detected:
            _print_verbose(f"  Proposed full configuration (attempt {attempts_full_config}) failed due to collision. Retrying.", 1, state)
            continue # Re-attempt generating a new full configuration
        else:
            # If no collision, this proposed configuration is valid.
            state.rp[:] = proposed_rp[:]
            state.rcm[:] = proposed_rcm[:]
            _print_verbose(f"  Successfully generated non-colliding proposed configuration after {attempts_full_config} attempts.", 1, state)
            break # Exit the outer while loop

    # Recalculate CMs for all molecules after the move, to ensure consistency
    # (though config_move should have kept them consistent for the moved molecule)
    # This loop ensures all CMs are up-to-date in state.rcm
    for i in range(state.num_molecules):
        mol_start_idx = state.imolec[i]
        mol_end_idx = state.imolec[i+1]
        mol_atomic_numbers = [state.iznu[j] for j in range(mol_start_idx, mol_end_idx)]
        mol_masses = np.array([state.atomic_number_to_mass.get(anum, 1.0) 
                               for anum in mol_atomic_numbers])
        state.rcm[i, :] = calculate_mass_center(state.rp[mol_start_idx:mol_end_idx, :], mol_masses)


# Function to clean up generated QM input/output files (now mostly handled by calculate_energy's finally block)
def cleanup_qm_files(files_to_clean: List[str], state: SystemState):
    """
    Cleans up any QM-related files explicitly added to the list.
    This function now primarily serves as a safeguard for files that might not have been
    cleaned by individual calculate_energy calls due to unexpected crashes or other issues.
    """
    # This list is mostly unused now as calculate_energy handles its own cleanup.
    # It remains here as a safety net if a file was created outside that scope and added.
    cleaned_count = 0
    files_to_remove = list(files_to_clean) 
    for fpath in files_to_remove:
        if os.path.exists(fpath):
            try:
                os.remove(fpath)
                cleaned_count += 1
            except OSError as e:
                _print_verbose(f"Error removing file {fpath} during final cleanup: {e}", 0, state)
    if cleaned_count > 0:
        _print_verbose(f"Cleaned up {cleaned_count} leftover QM files during final cleanup.", 1, state)

# Add this function somewhere in your script, e.g., near helper functions.
def provide_box_length_advice(state: SystemState):
    """
    Analyzes molecule definitions and provides advice on appropriate box lengths
    based on the largest molecular dimension and a minimum vacuum clearance.
    """
    if not state.all_molecule_definitions:
        _print_verbose("Cannot provide box length advice: No molecule definitions found.", 0, state)
        return

    # Constant for advice
    MIN_VACUUM_PER_SIDE_ANGSTROM = 8.0 # Minimum recommended vacuum between molecule and wall

    max_molecular_extent = 0.0 # Will store the largest dimension of any molecule

    for mol_def in state.all_molecule_definitions:
        if not mol_def.atoms_coords:
            continue

        coords_array = np.array([atom[1:] for atom in mol_def.atoms_coords]) # Extract just x,y,z
        
        # Calculate min/max for each dimension for the current molecule
        min_coords = np.min(coords_array, axis=0)
        max_coords = np.max(coords_array, axis=0)
        
        # Calculate spatial extent (length, width, height) of the current molecule
        extent_x = max_coords[0] - min_coords[0]
        extent_y = max_coords[1] - min_coords[1]
        extent_z = max_coords[2] - min_coords[2]
        
        # The largest dimension of this molecule
        current_mol_max_extent = max(extent_x, extent_y, extent_z)
        
        if current_mol_max_extent > max_molecular_extent:
            max_molecular_extent = current_mol_max_extent

    if max_molecular_extent == 0.0:
        _print_verbose("Cannot provide box length advice: Molecular dimensions could not be determined.", 0, state)
        return
    
    state.max_molecular_extent = max_molecular_extent # Store this in state for summary printing

    # Calculate recommended minimum box length
    # This accounts for the largest molecule plus some vacuum on both sides.
    recommended_box_length = max_molecular_extent + (2 * MIN_VACUUM_PER_SIDE_ANGSTROM)
    
    _print_verbose("\n--- Box Length Suggestion ---", 1, state)
    _print_verbose(f"Based on your molecule definitions:", 1, state)
    _print_verbose(f"    - Largest molecule's maximum dimension: {max_molecular_extent:.2f} A", 1, state) # Changed to A
    _print_verbose(f"\nRecommendation for Simulation Cube Length:", 1, state)
    _print_verbose(f"    - Practical box length: {recommended_box_length:.2f} A", 1, state) # Changed to A
    _print_verbose(f"    - Ensures {MIN_VACUUM_PER_SIDE_ANGSTROM:.1f} A clearance per side", 1, state) # Changed to A
    _print_verbose("-----------------------------\n", 1, state)

def format_time_difference(seconds: float) -> str:
    """Formats a time difference in seconds into days, hours, minutes, seconds, milliseconds."""
    days = int(seconds // (24 * 3600))
    seconds %= (24 * 3600)
    hours = int(seconds // 3600)
    seconds %= 3600
    minutes = int(seconds // 60)
    seconds %= 60
    milliseconds = int((seconds - int(seconds)) * 1000)
    seconds = int(seconds)
    
    return f"{days} days {hours} h {minutes} min {seconds} s {milliseconds} ms"

def write_lowest_energy_config_file(state: SystemState, rless_filepath: str):
    """
    Writes the lowest energy configuration to a .out file,
    separated by molecule.
    """
    if state.lowest_energy_rp is None or state.lowest_energy_iznu is None:
        _print_verbose(f"Warning: No lowest energy configuration found to write to '{rless_filepath}'.", 0, state)
        return

    try:
        with open(rless_filepath, 'w') as f:
            # Modified format as requested (reduced spacing)
            f.write(f"# Configuration: {state.lowest_energy_config_idx} | Energy = {state.lowest_energy:.8f} u.a.\n") # Added |
            
            current_atom_idx = 0
            for i, mol_def_idx in enumerate(state.molecules_to_add):
                mol_def = state.all_molecule_definitions[mol_def_idx]
                
                f.write(f"{mol_def.num_atoms}\n")
                f.write(f"{mol_def.label}\n") # Changed to use full label from mol_def.label

                for _ in range(mol_def.num_atoms):
                    atomic_num = state.lowest_energy_iznu[current_atom_idx]
                    symbol = state.atomic_number_to_symbol.get(atomic_num, 'X')
                    coords = state.lowest_energy_rp[current_atom_idx]
                    f.write(f"{symbol: <3} {coords[0]: 12.6f} {coords[1]: 12.6f} {coords[2]: 12.6f}\n")
                    current_atom_idx += 1
        _print_verbose(f"Lowest energy configuration saved to: {rless_filepath}", 1, state)
    except IOError as e:
        _print_verbose(f"Error writing lowest energy configuration to '{rless_filepath}': {e}", 0, state)

def write_tvse_file(tvse_filepath: str, entry: Dict, state: SystemState):
    """
    Appends a single accepted configuration entry to the .dat file.
    """
    try:
        # Check if file exists and is empty to write header
        file_exists = os.path.exists(tvse_filepath)
        file_is_empty = not file_exists or os.stat(tvse_filepath).st_size == 0

        with open(tvse_filepath, 'a') as f: # Open in append mode
            if file_is_empty:
                # Adjusted header and spacing for alignment
                f.write(f"{'# n-eval (Cuml)':>16} {'T(K)':>10} {'E(u.a.)':>15}\n")
                f.write("\n") # Blank line after header
                f.write("#----------------------------------\n") # Separator line
            # Adjusted spacing for data columns to match header
            f.write(f"  {entry['n_eval']:>14} {entry['T']:>10.2f} {entry['E']:>15.6f}\n") 
        # _print_verbose(f"Energy evolution history appended to: {tvse_filepath}", 2, state) # Too verbose
    except IOError as e:
        _print_verbose(f"Error writing energy evolution history to '{tvse_filepath}': {e}", 0, state)

# 16. main ascec integrated function
def main_ascec_integrated():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="ASCEC: Annealing Simulation")
    parser.add_argument("input_file", help="Path to the input file (e.g., w5-apy.in)")
    parser.add_argument("--v", action="store_true", help="Verbose output: print detailed steps every 10 cycles.")
    parser.add_argument("--va", action="store_true", help="Very verbose output: print detailed steps for every cycle.")
    parser.add_argument("--standard", action="store_true", help="Use standard Metropolis criterion instead of modified.")
    args = parser.parse_args()

    # Initialize file handles and paths to None to prevent UnboundLocalError
    out_file_handle = None 
    failed_initial_configs_xyz_handle = None 
    failed_configs_path = None 
    
    rless_file_path = None 
    tvse_file_path = None  
    xyz_filename_base = "" # Initialize for error path
    rless_filename = ""    # Initialize for error path
    tvse_filename = ""     # Initialize for error path

    # Determine the directory where the input file is located
    input_file_path_full = os.path.abspath(args.input_file)
    run_dir = os.path.dirname(input_file_path_full)
    if not run_dir:
        run_dir = os.getcwd()

    qm_files_to_clean: List[str] = [] 
    initial_qm_successful = False 
    
    start_time = time.time() # Start wall time measurement

    state = SystemState()
    # Set verbosity level based on command line arguments
    if args.va:
        state.verbosity_level = 2
    elif args.v:
        state.verbosity_level = 1
    else:
        state.verbosity_level = 0 # Default minimal output
    
    state.use_standard_metropolis = args.standard # Set the flag in state

    # Check for Open Babel executable early
    if not shutil.which("obabel"):
        _print_verbose("\nCRITICAL ERROR: Open Babel executable 'obabel' not found in your system's PATH.", 0, state)
        _print_verbose("Please ensure Open Babel is installed and its executable is accessible from your command line.", 0, state)
        _print_verbose("Cannot proceed with .mol file generation.", 0, state)
    
    try:
        # Call read_input_file as early as possible to populate state
        read_input_file(state, args.input_file)

        # --- Call for Box Length Advice ---
        provide_box_length_advice(state) # This will print to stderr

        # Set QM program name based on QM_PROGRAM_DETAILS mapping
        state.qm_program = QM_PROGRAM_DETAILS[state.ia]["name"]

        # If random seed is not explicitly set in input or invalid, generate one
        if state.random_seed == -1: 
            state.random_seed = random.randint(100000, 999999) # Generate a 6-digit integer
            _print_verbose(f"\nUsing random seed: {state.random_seed}\n", 0, state) # Modified print
        else:
             _print_verbose(f"\nUsing random seed: {state.random_seed}\n", 0, state) # Modified print
        
        # Initialize Python's random and NumPy's random with the seed
        random.seed(state.random_seed)
        np.random.seed(state.random_seed)

        input_base_name = os.path.splitext(os.path.basename(args.input_file))[0]

        # Define output filenames
        out_filename = f"{input_base_name}.out" 
        xyz_filename_base = f"result_{state.random_seed}" if state.random_generate_config == 1 else f"mto_{state.random_seed}"
        rless_filename = f"rless_{state.random_seed}.out"
        tvse_filename = f"tvse_{state.random_seed}.dat"

        out_file_path = os.path.join(run_dir, out_filename)
        rless_file_path = os.path.join(run_dir, rless_filename) # Full path defined here
        tvse_file_path = os.path.join(run_dir, tvse_filename)   # Full path defined here

        # Define and Open .out File
        try:
            out_file_handle = open(out_file_path, 'w')
            _print_verbose(f"Main output will be written to: {out_file_path}\n", 0, state)
        except IOError as e:
            _print_verbose(f"CRITICAL ERROR: Could not open out file '{out_file_path}': {e}", 0, state)
            sys.exit(1)

        # Write Simulation Summary to .out File (passing dynamic filenames)
        write_simulation_summary(state, out_file_handle, xyz_filename_base + ".xyz", rless_filename, tvse_filename) # Pass with .xyz for display
        out_file_handle.flush() 

        # For annealing mode, open a file for failed initial configurations
        if state.random_generate_config == 1: 
            failed_configs_filename = os.path.splitext(os.path.basename(args.input_file))[0] + "_failed_initial_configs.xyz"
            failed_configs_path = os.path.join(run_dir, failed_configs_filename)
            try:
                # Open in write mode if it needs to be created or overwritten
                failed_initial_configs_xyz_handle = open(failed_configs_path, 'w')
                _print_verbose(f"Failed initial configurations will be logged to: {failed_configs_path}\n", 1, state)
            except IOError as e:
                _print_verbose(f"Warning: Could not open file for failed initial configs '{failed_configs_path}': {e}. Continuing without logging these.", 0, state)
                failed_initial_configs_xyz_handle = None # Ensure it's None if opening failed

            _print_verbose("Attempting initial QM energy calculation...\n", 0, state) 
            initial_failed_config_idx = 0 

        # Define half_xbox once for coordinate system shift for visualization
        half_xbox = state.xbox / 2.0

        # Random Configuration Generation Mode (no QM, so files are opened directly)
        if state.random_generate_config == 0: 
            _print_verbose(f"Generating {state.num_random_configs} random configurations (no energy evaluation).\n", 0, state)
            
            # Use total_accepted_configs as the sequential counter for random configs
            for i in range(state.num_random_configs):
                state.total_accepted_configs += 1 # Increment for each random config
                # Re-initialize rp and iznu for each new random configuration
                # initialize_molecular_coords_in_box generates random configuration using config_molecules
                state.rp, state.iznu = initialize_molecular_coords_in_box(state) 
                
                # Write to the ORIGINAL XYZ file handle (no dummy atoms)
                write_accepted_xyz(
                    xyz_filename_base, 
                    state.total_accepted_configs, # Pass the incremented count
                    0.0, # Energy is 0.0 as it's not evaluated in this mode
                    state.current_temp, # Temp not relevant for MTO, but passed for consistency
                    state,
                    is_initial=False 
                )
            _print_verbose("Random configuration generation complete.\n", 0, state)
            
            # --- Add the final output for random mode here ---
            with open(out_file_path, 'a') as out_file_handle_local:
                out_file_handle_local.write("\n") # Explicit blank line before the final summary separator
                out_file_handle_local.write("=" * 60 + "\n") 
                out_file_handle_local.write("\n  ** Normal random generation termination **\n") # Corrected termination message
                end_time = time.time() # End wall time measurement
                total_wall_time_seconds = end_time - start_time
                time_delta = timedelta(seconds=int(total_wall_time_seconds))
                days = time_delta.days
                hours = time_delta.seconds // 3600
                minutes = (time_delta.seconds % 3600) // 60
                seconds = time_delta.seconds % 60
                milliseconds = int((total_wall_time_seconds - int(total_wall_time_seconds)) * 1000)
                out_file_handle_local.write(f"  Total Wall time: {days} days {hours} h {minutes} min {seconds} s {milliseconds} ms\n")
                out_file_handle_local.write("\n" + "=" * 60 + "\n")


        # Annealing Mode (state.random_generate_config = 1)
        elif state.random_generate_config == 1: 
            _print_verbose("Starting annealing simulation...\n", 0, state)
            
            # Set initial temperature
            state.current_temp = state.linear_temp_init if state.quenching_routine == 1 else state.geom_temp_init

            # Initialize last_accepted_qm_call_count and last_history_qm_call_count before the first QM call
            state.last_accepted_qm_call_count = 0 
            state.last_history_qm_call_count = 0 

            # Initial QM Calculation Loop (with retries)
            initial_qm_calculation_succeeded = False 
            for attempt in range(state.initial_qm_retries):
                _print_verbose(f"  Attempt {attempt + 1}/{state.initial_qm_retries} for initial QM calculation.", 0, state)

                # Generate a new random initial configuration for each retry
                state.rp, state.iznu = initialize_molecular_coords_in_box(state)
                
                initial_failed_config_idx += 1 

                # Log initial configuration (shifted for visualization) to the failed configs file
                if failed_initial_configs_xyz_handle:
                    rp_for_viz_failed = state.rp + half_xbox 
                    # Re-using write_single_xyz_configuration for failed configs as it doesn't need full accepted_xyz logic
                    write_single_xyz_configuration(
                        failed_initial_configs_xyz_handle, 
                        state.natom, rp_for_viz_failed, state.iznu, 0.0, # Energy is 0.0 as it's not yet calculated or failed
                        initial_failed_config_idx, state.atomic_number_to_symbol,
                        state.random_generate_config, 
                        remark=f"Initial Setup Attempt {attempt + 1}", 
                        include_dummy_atoms=True, # Always include dummy atoms for failed initial configs
                        state=state 
                    )
                
                try:
                    initial_energy, jo_status = calculate_energy(state.rp, state.iznu, state, run_dir)
                    
                    if jo_status == 0:
                        raise RuntimeError("QM program returned non-zero status or could not calculate energy for initial configuration.") 

                    state.current_energy = initial_energy
                    state.lowest_energy = initial_energy 
                    state.lowest_energy_rp = np.copy(state.rp) # Store initial lowest energy coords
                    state.lowest_energy_iznu = list(state.iznu) # Store initial lowest energy atomic numbers
                    state.lowest_energy_config_idx = 1 # Initial config is always config 1
                    
                    _print_verbose(f"  Calculation successful. Energy: {state.current_energy:.8f} a.u.\n", 0, state) # Modified print
                    initial_qm_calculation_succeeded = True
                    state.total_accepted_configs += 1 # Increment for initial accepted config
                    break # Exit retry loop on success

                except RuntimeError as e:
                    _print_verbose(f"  Initial QM calculation attempt {attempt + 1} failed: {e}", 0, state)
                    if attempt < state.initial_qm_retries - 1:
                        _print_verbose("  Generating new initial configuration and retrying...\n", 0, state)
                    else:
                        raise RuntimeError(
                            f"All {state.initial_qm_retries} attempts to perform the initial QM energy calculation failed. "
                            "Please verify your QM input parameters (method, basis set, memory, processors) "
                            "or inspect the run directory for specific QM program output errors. "
                            "Cannot proceed with annealing simulation."
                        )
                except Exception as e:
                    _print_verbose(f"  An unexpected error occurred during initial QM attempt {attempt + 1}: {e}", 0, state)
                    raise # Re-raise other unexpected errors

            if not initial_qm_calculation_succeeded:
                # If initial QM failed, proceed to generate MOL for failed configs
                if failed_initial_configs_xyz_handle:
                    try: failed_initial_configs_xyz_handle.close() # Close before attempting conversion
                    except Exception as e: _print_verbose(f"Error closing failed initial configs file during final cleanup: {e}", 0, state)
                    
                    if os.path.exists(failed_configs_path):
                        failed_mol_filename = os.path.splitext(failed_configs_path)[0] + ".mol"
                        try:
                            _print_verbose(f"Attempting to convert failed initial configs XYZ '{os.path.basename(failed_configs_path)}' to MOL...", 1, state)
                            if convert_xyz_to_mol(failed_configs_path, state.openbabel_alias, state):
                                _post_process_mol_file(failed_mol_filename, state)
                                _print_verbose(f"Successfully created and post-processed '{os.path.basename(failed_mol_filename)}'.", 1, state)
                            else:
                                _print_verbose(f"Failed to create '{os.path.basename(failed_mol_filename)}'. See previous warnings.", 1, state)
                        except Exception as e:
                            _print_verbose(f"Warning: Could not create or post-process .mol file for failed initial configs '{os.path.basename(failed_configs_path)}': {e}", 0, state)

                raise RuntimeError("Initial QM energy calculation did not succeed after all retries. Exiting.")

            # If initial QM succeeded, we can remove the failed initial configs file
            if failed_initial_configs_xyz_handle and os.path.exists(failed_configs_path):
                try:
                    failed_initial_configs_xyz_handle.close() # Close before removing
                    os.remove(failed_configs_path)
                    _print_verbose(f"Removed '{failed_configs_path}' as initial QM calculation succeeded.\n", 1, state)
                except OSError as e:
                    _print_verbose(f"Error removing '{failed_configs_path}': {e}", 0, state)
                finally:
                    failed_initial_configs_xyz_handle = None # Clear handle

            # Initial calculation of all molecular centers of mass
            # This is crucial for the first call to config_move
            for i in range(state.num_molecules):
                mol_start_idx = state.imolec[i]
                mol_end_idx = state.imolec[i+1]
                mol_atomic_numbers = [state.iznu[j] for j in range(mol_start_idx, mol_end_idx)]
                mol_masses = np.array([state.atomic_number_to_mass.get(anum, 1.0) for anum in mol_atomic_numbers])
                state.rcm[i, :] = calculate_mass_center(state.rp[mol_start_idx:mol_end_idx, :], mol_masses)

            # Determine total annealing steps
            if state.quenching_routine == 1: 
                total_annealing_steps = state.linear_num_steps
            elif state.quenching_routine == 2: 
                total_annealing_steps = state.geom_num_steps
            else:
                _print_verbose("Error: Invalid quenching_routine specified for annealing mode. Must be 1 (linear) or 2 (geometric).", 0, state)
                return 

            # Add the initial successful QM calculation to history and tvse
            if initial_qm_calculation_succeeded:
                # n-eval for initial entry is 0 as per user's desired output format
                history_n_eval = 0 
                accepted_history_entry = {
                    "T": state.current_temp, # Initial temp
                    "E": state.current_energy,
                    "n_eval": history_n_eval, 
                    "criterion": "Intv"
                }
                # Write to .out file history immediately (adjusted spacing)
                with open(out_file_path, 'a') as f_out: # Open in append mode
                    # Added two spaces to n-eval column
                    f_out.write(f"  {accepted_history_entry['T']:>8.2f} {accepted_history_entry['E']:>12.6f} {accepted_history_entry['n_eval']:>7} {accepted_history_entry['criterion']:>8}\n")
                
                # Update last_history_qm_call_count after writing the initial entry
                state.last_history_qm_call_count = state.qm_call_count

                # Write to .dat file (tvse) immediately
                write_tvse_file(tvse_file_path, {
                    "n_eval": state.qm_call_count, # Use total QM calls for TVSE
                    "T": state.current_temp,
                    "E": state.current_energy
                }, state)
                
                _print_verbose(f"  Initial accepted config added to history. Initial T: {state.current_temp:.2f} K, E: {state.current_energy:.8f} a.u., QM calls: {state.qm_call_count}", 1, state)
                
                # IMPORTANT: Update last_accepted_qm_call_count after the initial accepted config
                state.last_accepted_qm_call_count = state.qm_call_count 

            # Write the first *successful* QM configuration to the main XYZ file(s)
            write_accepted_xyz(
                xyz_filename_base, 
                state.total_accepted_configs, 
                state.current_energy, 
                state.current_temp, 
                state,
                is_initial=True 
            )

            # Main Annealing Loop
            for step_num in range(total_annealing_steps):
                # Apply temperature quenching BEFORE the MC cycles for this step to use the correct temperature
                if step_num > 0: # Don't quench for the very first step, as current_temp is already init_temp
                    if state.quenching_routine == 1: # Linear quenching
                        state.current_temp = max(state.current_temp - state.linear_temp_decrement, 0.001) 
                    elif state.quenching_routine == 2: # Geometric quenching
                        state.current_temp = max(state.current_temp * state.geom_temp_factor, 0.001)

                # Always print the start of a new annealing step if not very verbose (level 0 or 1)
                if state.verbosity_level <= 1:
                    print(f"\nAnnealing Step {step_num + 1}/{total_annealing_steps} at Temperature: {state.current_temp:.2f} K", file=sys.stderr) # Modified print
                    sys.stderr.flush()
                elif state.verbosity_level == 2: # For very verbose, print every step
                     _print_verbose(f"\nAnnealing Step {step_num + 1}/{total_annealing_steps} at Temperature: {state.current_temp:.2f} K", 2, state) # Modified print
                
                # Keep a copy of the current accepted state before proposing a new move
                # These are the *reference* states for the Metropolis criterion
                original_rp_for_revert = np.copy(state.rp) 
                original_iznu_for_revert = list(state.iznu)
                original_energy_for_revert = state.current_energy
                original_rcm_for_revert = np.copy(state.rcm) # Also save rcm for revert

                qm_calls_made_this_temp_step = 0 # Counter for QM calls within this temperature step
                
                # Flag to control moving to the next temperature step
                # Set to True if LwE is accepted, or if maxstep is reached for this temperature
                should_move_to_next_temperature = False

                # Run Monte Carlo cycles at current temperature, up to maxstep evaluations
                while qm_calls_made_this_temp_step < state.maxstep:
                    qm_calls_made_this_temp_step += 1 # Increment QM calls for this temperature step
                    
                    # PROPOSE NEW CONFIGURATION by modifying the current state
                    # The config_move function will directly modify state.rp and state.rcm
                    config_move(state) 
                    
                    # Calculate energy of the proposed configuration (which is now in state.rp)
                    proposed_energy, jo_status = calculate_energy(state.rp, state.iznu, state, run_dir)
                    
                    # Verbose output for each cycle (using state.qm_call_count for global attempt number)
                    if state.verbosity_level == 2 or (state.verbosity_level == 1 and state.qm_call_count % 10 == 0):
                        _print_verbose(f"  Attempt {state.qm_call_count} (global), {qm_calls_made_this_temp_step} (step-local): T={state.current_temp:.2f} K", 1, state)


                    if jo_status == 0:
                        _print_verbose(f"  Warning: Proposed QM energy calculation failed for global attempt {state.qm_call_count}. Rejecting move.", 1, state)
                        # Revert to the original state if QM failed for this proposal
                        state.rp[:] = original_rp_for_revert[:]
                        state.iznu[:] = original_iznu_for_revert[:]
                        state.current_energy = original_energy_for_revert
                        state.rcm[:] = original_rcm_for_revert[:] # Revert rcm as well
                        continue # Continue to next MC cycle attempt

                    accept_move = False
                    # Compare proposed energy with the energy of the *original* (last accepted) configuration
                    delta_e = proposed_energy - original_energy_for_revert 
                    
                    criterion_str = "" # Reset criterion string for each attempt

                    if delta_e <= 0.0: # If proposed energy is lower or equal, always accept
                        accept_move = True
                        criterion_str = "LwE" 
                    else: # If proposed energy is higher, apply Metropolis criterion
                        # Ensure temperature is not zero for division
                        if state.current_temp < 1e-6: 
                            pE = 0.0
                        else:
                            pE = math.exp(-delta_e / (B2 * state.current_temp))
                        
                        if state.use_standard_metropolis: # Standard Metropolis
                            if random.random() < pE:
                                accept_move = True
                                criterion_str = "Mpol"
                        else: # Modified Metropolis (default)
                            # Handle division by zero for crt (if proposed_energy is zero)
                            if abs(proposed_energy) < 1e-12: # Avoid division by near-zero energy
                                crt = float('inf') 
                            else:
                                crt = delta_e / abs(proposed_energy)

                            if crt < pE: # This is the modified criterion
                                accept_move = True
                                criterion_str = "Mpol" 
                                state.iboltz += 1 

                    if accept_move:
                        # If accepted, update the system's current state to the *proposed* one
                        state.current_energy = proposed_energy 
                        # state.rp and state.rcm are already the proposed (accepted) state
                        
                        # Update original_state_for_revert to the newly accepted state
                        original_rp_for_revert[:] = state.rp[:]
                        original_iznu_for_revert[:] = state.iznu[:]
                        original_energy_for_revert = state.current_energy
                        original_rcm_for_revert[:] = state.rcm[:] # Update rcm for revert too

                        _print_verbose(f"  Attempt {state.qm_call_count} (global): Move accepted ({criterion_str}). New energy: {state.current_energy:.8f} a.u.", 1, state)
                        
                        # Update lowest energy found overall
                        if state.current_energy < state.lowest_energy:
                            state.lowest_energy = state.current_energy
                            state.lowest_energy_rp = np.copy(state.rp) 
                            state.lowest_energy_iznu = list(state.iznu) 
                            state.lowest_energy_config_idx = state.qm_call_count 
                            _print_verbose(f"  New lowest energy found: {state.lowest_energy:.8f} a.u. (Config {state.lowest_energy_config_idx})", 1, state)
                        
                        # Calculate n-eval for .out history: QM calls since last history entry
                        history_n_eval = state.qm_call_count - state.last_history_qm_call_count
                        
                        with open(out_file_path, 'a') as f_out:
                            f_out.write(f"  {state.current_temp:>8.2f} {state.current_energy:>12.6f} {history_n_eval:>7} {criterion_str:>8}\n")
                        
                        # Update last_history_qm_call_count after writing this entry
                        state.last_history_qm_call_count = state.qm_call_count

                        write_tvse_file(tvse_file_path, {
                            "n_eval": state.qm_call_count, # Use total QM calls for TVSE (cumulative)
                            "T": state.current_temp,
                            "E": state.current_energy
                        }, state)
                        
                        # Update last_accepted_qm_call_count to the current global QM call count
                        state.last_accepted_qm_call_count = state.qm_call_count

                        state.total_accepted_configs += 1 
                        write_accepted_xyz(
                            xyz_filename_base, 
                            state.total_accepted_configs, 
                            state.current_energy, 
                            state.current_temp, 
                            state,
                            is_initial=False 
                        )
                        
                        # This is the key change for loop control:
                        if criterion_str == "LwE":
                            _print_verbose(f"  Lower energy accepted. Moving to next temperature step.", 1, state)
                            should_move_to_next_temperature = True
                            break # Break out of the inner for loop (Monte Carlo cycles)
                        elif criterion_str == "Mpol":
                            _print_verbose(f"  Metropolis accepted. Continuing Monte Carlo cycles at current temperature.", 1, state)
                            # Do NOT set should_move_to_next_temperature = True, continue the loop until maxstep is reached
                        
                    else: # Move rejected or QM failed
                        _print_verbose(f"  Attempt {state.qm_call_count} (global): Move rejected. Energy: {proposed_energy:.8f} a.u. (Current: {original_energy_for_revert:.8f})", 1, state)
                        state.rp[:] = original_rp_for_revert[:]
                        state.iznu[:] = original_iznu_for_revert[:]
                        state.current_energy = original_energy_for_revert
                        state.rcm[:] = original_rcm_for_revert[:] # Revert rcm as well
                
                # After the inner loop (maxstep attempts or LwE break)
                # If we did NOT move to the next temperature due to LwE, it means maxstep was reached
                # or only Mpol was accepted (and maxstep was reached).
                # In this case, if the last history entry was NOT the current QM call count,
                # it means there were unlogged QM calls since the last history entry.
                # This happens if Mpol was accepted, or if no configuration was accepted.
                if not should_move_to_next_temperature and state.qm_call_count > state.last_history_qm_call_count:
                    # This means the inner loop completed because maxstep was reached,
                    # and either an Mpol was accepted earlier in this loop (and a line was written),
                    # or no config was accepted at all.
                    # We need to write an N/A line to account for the remaining QM calls
                    # at the *current* temperature before it quenches.
                    history_n_eval = state.qm_call_count - state.last_history_qm_call_count
                    if history_n_eval > 0: # Only write if there were actual calls since last history entry
                        with open(out_file_path, 'a') as f_out:
                            f_out.write(f"  {state.current_temp:>8.2f} {original_energy_for_revert:>12.6f} {history_n_eval:>7} {'N/A':>8}\n") 
                        state.last_history_qm_call_count = state.qm_call_count
                    _print_verbose(f"  Max cycles reached for {state.current_temp:.2f} K. Moving to next temperature step.", 1, state)
                elif should_move_to_next_temperature:
                    # If LwE was accepted, we already broke and logged it.
                    pass # Nothing more to do here, the loop will naturally go to the next step_num

                # Dynamic max_cycle reduction
                state.maxstep = max(100, int(state.maxstep * 0.90)) # Reduce by 10% with floor of 100
                _print_verbose(f"  Max QM evaluations for next temperature step reduced to: {state.maxstep}", 1, state)

            _print_verbose("\nAnnealing simulation finished.\n", 0, state)
            _print_verbose(f"Final lowest energy found: {state.lowest_energy:.8f} a.u.", 0, state)
            _print_verbose(f"Total QM calculations performed: {state.qm_call_count}", 0, state)

            # Final summary to .out file for annealing mode
            with open(out_file_path, 'a') as out_file_handle_local: # Use a local alias for clarity
                out_file_handle_local.write("\n") # Explicit blank line before the final summary separator
                out_file_handle_local.write("=" * 60 + "\n") 
                out_file_handle_local.write("\n  ** Normal annealing termination **\n") # Added \n before
                end_time = time.time() # End wall time measurement
                total_wall_time_seconds = end_time - start_time
                time_delta = timedelta(seconds=int(total_wall_time_seconds))
                days = time_delta.days
                hours = time_delta.seconds // 3600
                minutes = (time_delta.seconds % 3600) // 60
                seconds = time_delta.seconds % 60
                milliseconds = int((total_wall_time_seconds - int(total_wall_time_seconds)) * 1000)
                out_file_handle_local.write(f"  Total Wall time: {days} days {hours} h {minutes} min {seconds} s {milliseconds} ms\n")
                out_file_handle_local.write(f"  Energy was evaluated {state.qm_call_count} times\n\n")
                out_file_handle_local.write(f"Energy evolution in {tvse_filename}\n")
                out_file_handle_local.write(f"Configurations accepted by Max.-Boltz. statistics = {state.iboltz}\n")
                # Updated to include total accepted configurations
                out_file_handle_local.write(f"Accepted configurations in {xyz_filename_base}.xyz = {state.total_accepted_configs}\n") 
                out_file_handle_local.write(f"Lowest energy configuration in {rless_filename}\n")
                
                # This will print the lowest energy found by the end of the simulation
                if state.lowest_energy_rp is not None:
                    out_file_handle_local.write(f"Lowest energy = {state.lowest_energy:.8f} u.a. (Config. {state.lowest_energy_config_idx})\n")
                else:
                    out_file_handle_local.write("Lowest energy = N/A (No configurations accepted)\n")
                
                out_file_handle_local.write("\n" + "=" * 60 + "\n")

    finally:
        # Ensure all output files are closed
        if out_file_handle:
            try: out_file_handle.close()
            except Exception as e: _print_verbose(f"Error closing main output file: {e}", 0, state)
        
        # Write lowest energy config file only if it was successfully found (already handled in annealing loop)
        if state.random_generate_config == 1 and state.lowest_energy_rp is not None:
            write_lowest_energy_config_file(state, rless_file_path)
        elif state.random_generate_config == 1 and state.lowest_energy_rp is None:
            _print_verbose(f"No lowest energy configuration was successfully found and stored. Skipping '{rless_file_path}' generation.", 0, state)

        # Handle MOL conversion for XYZ files (which were created by write_accepted_xyz)
        # We need to ensure paths are defined before attempting conversion
        _print_verbose(f"\n--- Initiating .mol file conversions ---", 1, state)

        # Only attempt mol conversion if obabel was found at startup
        if shutil.which("obabel"):
            main_xyz_path = os.path.join(run_dir, f"{xyz_filename_base}.xyz")
            box_xyz_path = os.path.join(run_dir, f"{xyz_filename_base.replace('mto_', 'mtobox_').replace('result_', 'resultbox_')}.xyz")

            if os.path.exists(main_xyz_path):
                _print_verbose(f"  Processing main XYZ file for .mol conversion: '{os.path.basename(main_xyz_path)}'", 1, state)
                mol_filename = os.path.splitext(main_xyz_path)[0] + ".mol"
                try:
                    if convert_xyz_to_mol(main_xyz_path, state.openbabel_alias, state):
                        _post_process_mol_file(mol_filename, state)
                        _print_verbose(f"  Successfully processed '{os.path.basename(mol_filename)}'.", 1, state)
                    else:
                        _print_verbose(f"  Failed to create '{os.path.basename(mol_filename)}'. Review Open Babel output above for details.", 1, state)
                except Exception as e:
                    _print_verbose(f"  Warning: Unexpected error during .mol creation for '{os.path.basename(main_xyz_path)}': {e}", 0, state)

            if CREATE_BOX_XYZ_COPY and os.path.exists(box_xyz_path):
                _print_verbose(f"  Processing box XYZ file for .mol conversion: '{os.path.basename(box_xyz_path)}'", 1, state)
                mol_box_filename = os.path.splitext(box_xyz_path)[0] + ".mol"
                try:
                    if convert_xyz_to_mol(box_xyz_path, state.openbabel_alias, state):
                        _post_process_mol_file(mol_box_filename, state)
                        _print_verbose(f"  Successfully processed '{os.path.basename(mol_box_filename)}'.", 1, state)
                    else:
                        _print_verbose(f"  Failed to create '{mol_box_filename}'. Review Open Babel output above for details.", 1, state)
                except Exception as e:
                    _print_verbose(f"  Warning: Unexpected error during .mol creation for '{os.path.basename(box_xyz_path)}': {e}", 0, state)
            
            # This block now handles MOL conversion for failed initial configs *only if* the initial QM ultimately failed.
            # The file is not removed if initial_qm_successful is False, so it persists for MOL conversion.
            if state.random_generate_config == 1 and not initial_qm_successful and failed_configs_path and os.path.exists(failed_configs_path):
                if failed_initial_configs_xyz_handle:
                    try: failed_initial_configs_xyz_handle.close() # Ensure it's closed before MOL conversion
                    except Exception as e: _print_verbose(f"Error closing failed initial configs XYZ file: {e}", 0, state)
                
                _print_verbose(f"  Processing failed initial configs XYZ for .mol conversion: '{os.path.basename(failed_configs_path)}'", 1, state)
                failed_mol_filename = os.path.splitext(failed_configs_path)[0] + ".mol"
                try:
                    if convert_xyz_to_mol(failed_configs_path, state.openbabel_alias, state):
                        _post_process_mol_file(failed_mol_filename, state)
                        _print_verbose(f"  Successfully processed '{os.path.basename(failed_mol_filename)}'.", 1, state)
                    else:
                        _print_verbose(f"  Failed to create '{os.path.basename(failed_mol_filename)}'. Review Open Babel output above for details.", 1, state)
                except Exception as e:
                    _print_verbose(f"  Warning: Unexpected error during .mol creation for failed initial configs '{os.path.basename(failed_configs_path)}': {e}", 0, state)
        else:
            _print_verbose(f"  Skipping .mol file conversions because Open Babel was not found in your PATH.", 1, state)
        
        _print_verbose(f"\n--- .mol file conversions completed ---", 1, state)

        # Final cleanup of any lingering QM files (should be minimal with per-call cleanup)
        cleanup_qm_files(qm_files_to_clean, state) 

# This ensures main_ascec_integrated() is called when the script is run
if __name__ == "__main__":
    main_ascec_integrated()

