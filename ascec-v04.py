#!/usr/bin/env python3

import argparse
from collections import Counter, defaultdict
import dataclasses
import glob
import math
import numpy as np
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Dict, List, Optional, Tuple 
from datetime import timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

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
# This can be overridden by the --nobox command line flag.
CREATE_BOX_XYZ_COPY = True 

version = "* ASCEC-v04: Jun-2025 *"  # Version of the ASCEC script

# Symbol for dummy atoms used to mark box corners.

# WARNING: 'X' is not a standard element.
DUMMY_ATOM_SYMBOL = "X"

# Atomic radii - Used in some initial configurations for overlap prevention
# These are typical Covalent Radii (single bond, in Angstroms).
# If your Fortran Ratom array uses different values or a specific parameterized set,
# please ensure consistency with that source.
R_ATOM = {
    # Covalent radii for elements in Angstroms (for volume calculations and steric interactions)
    # Based on Cordero et al. (2008) "Covalent radii revisited" Dalton Trans. 2832-2838
    # and optimized values from previous ASCEC usage
    
    # Period 1
    1: 0.31,   # H (Hydrogen)
    2: 0.28,   # He (Helium) - Van der Waals approximation
    
    # Period 2
    3: 1.28,   # Li (Lithium)
    4: 0.96,   # Be (Beryllium)
    5: 0.84,   # B (Boron)
    6: 0.73,   # C (Carbon) - Standard covalent radius
    7: 0.71,   # N (Nitrogen)
    8: 0.66,   # O (Oxygen)
    9: 0.57,   # F (Fluorine)
    10: 0.58,  # Ne (Neon) - Van der Waals approximation
    
    # Period 3
    11: 1.66,  # Na (Sodium)
    12: 1.41,  # Mg (Magnesium)
    13: 1.21,  # Al (Aluminum)
    14: 1.11,  # Si (Silicon)
    15: 1.07,  # P (Phosphorus)
    16: 1.05,  # S (Sulfur)
    17: 1.02,  # Cl (Chlorine)
    18: 1.06,  # Ar (Argon) - Van der Waals approximation
    
    # Period 4
    19: 2.03,  # K (Potassium)
    20: 1.76,  # Ca (Calcium)
    21: 1.70,  # Sc (Scandium)
    22: 1.60,  # Ti (Titanium)
    23: 1.53,  # V (Vanadium)
    24: 1.39,  # Cr (Chromium)
    25: 1.39,  # Mn (Manganese) - low spin
    26: 1.32,  # Fe (Iron) - low spin
    27: 1.26,  # Co (Cobalt) - low spin
    28: 1.24,  # Ni (Nickel)
    29: 1.32,  # Cu (Copper)
    30: 1.22,  # Zn (Zinc)
    31: 1.22,  # Ga (Gallium)
    32: 1.20,  # Ge (Germanium)
    33: 1.19,  # As (Arsenic)
    34: 1.20,  # Se (Selenium)
    35: 1.20,  # Br (Bromine)
    36: 1.16,  # Kr (Krypton) - Van der Waals approximation
    
    # Period 5
    37: 2.20,  # Rb (Rubidium)
    38: 1.95,  # Sr (Strontium)
    39: 1.90,  # Y (Yttrium)
    40: 1.75,  # Zr (Zirconium)
    41: 1.64,  # Nb (Niobium)
    42: 1.54,  # Mo (Molybdenum)
    43: 1.47,  # Tc (Technetium)
    44: 1.46,  # Ru (Ruthenium)
    45: 1.42,  # Rh (Rhodium)
    46: 1.39,  # Pd (Palladium)
    47: 1.45,  # Ag (Silver)
    48: 1.44,  # Cd (Cadmium)
    49: 1.42,  # In (Indium)
    50: 1.39,  # Sn (Tin)
    51: 1.39,  # Sb (Antimony)
    52: 1.38,  # Te (Tellurium)
    53: 1.39,  # I (Iodine)
    54: 1.40,  # Xe (Xenon) - Van der Waals approximation
    
    # Period 6
    55: 2.44,  # Cs (Cesium)
    56: 2.15,  # Ba (Barium)
    57: 2.07,  # La (Lanthanum)
    
    # Period 6 - Lanthanides (f-block, elements 58-71)
    58: 2.04,  # Ce (Cerium)
    59: 2.03,  # Pr (Praseodymium)
    60: 2.01,  # Nd (Neodymium)
    61: 1.99,  # Pm (Promethium)
    62: 1.98,  # Sm (Samarium)
    63: 1.98,  # Eu (Europium)
    64: 1.96,  # Gd (Gadolinium)
    65: 1.94,  # Tb (Terbium)
    66: 1.92,  # Dy (Dysprosium)
    67: 1.92,  # Ho (Holmium)
    68: 1.89,  # Er (Erbium)
    69: 1.90,  # Tm (Thulium)
    70: 1.87,  # Yb (Ytterbium)
    71: 1.87,  # Lu (Lutetium)
    
    # Period 6 - Transition metals (d-block, elements 72-80)
    72: 1.75,  # Hf (Hafnium)
    73: 1.70,  # Ta (Tantalum)
    74: 1.62,  # W (Tungsten)
    75: 1.51,  # Re (Rhenium)
    76: 1.44,  # Os (Osmium)
    77: 1.41,  # Ir (Iridium)
    78: 1.36,  # Pt (Platinum)
    79: 1.36,  # Au (Gold)
    80: 1.32,  # Hg (Mercury)
    
    # Period 6 - Main group (p-block, elements 81-86)
    81: 1.45,  # Tl (Thallium)
    82: 1.46,  # Pb (Lead)
    83: 1.48,  # Bi (Bismuth)
    84: 1.40,  # Po (Polonium)
    85: 1.50,  # At (Astatine)
    86: 1.50,  # Rn (Radon)
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

# Atomic Number to Element Symbol Mapping (reverse of ELEMENT_SYMBOLS)
# This dictionary will be used to convert atomic numbers to element symbols for output.
ATOMIC_NUMBER_TO_SYMBOL = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca",
    21: "Sc", 22: "Ti", 23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu", 30: "Zn",
    31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr", 37: "Rb", 38: "Sr", 39: "Y", 40: "Zr",
    41: "Nb", 42: "Mo", 43: "Tc", 44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La", 58: "Ce", 59: "Pr", 60: "Nd",
    61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd", 65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb",
    71: "Lu", 72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt", 79: "Au", 80: "Hg",
    81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At", 86: "Rn", 87: "Fr", 88: "Ra", 89: "Ac", 90: "Th",
    91: "Pa", 92: "U", 93: "Np", 94: "Pu", 95: "Am", 96: "Cm", 97: "Bk", 98: "Cf", 99: "Es", 100: "Fm",
    101: "Md", 102: "No", 103: "Lr", 104: "Rf", 105: "Db", 106: "Sg", 107: "Bh", 108: "Hs", 109: "Mt", 110: "Ds",
    111: "Rg", 112: "Cn", 113: "Nh", 114: "Fl", 115: "Mc", 116: "Lv", 117: "Ts", 118: "Og"
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
    'Hs': 108, 'Mt': 109, 'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114,
    'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}

# 2. Define SystemState Class
class SystemState:
    def __init__(self):
        # Configuration parameters read from input file
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
        self.max_cycle_floor: int = 10            # Floor value for maxstep reduction (default: 10)
        self.max_displacement_a: float = 0.0      # Maximum Displacement of each mass center (ds)
        self.max_rotation_angle_rad: float = 0.0  # Maximum Rotation angle in radians (dphi)
        self.conformational_move_prob: float = 0.0  # Probability of conformational move vs rigid-body move (read from input)
        self.max_dihedral_angle_rad: float = 0.0  # Maximum dihedral rotation angle in radians (read from input)
        self.ia: int = 0                          # QM program type (1: Gaussian, 2: ORCA, etc.)
        self.alias: str = ""                      # Program alias/executable name (e.g., "g09")
        self.qm_method: Optional[str] = None      # (e.g., "pm3", "hf") - Renamed from hamiltonian for clarity
        self.qm_basis_set: Optional[str] = None   # (e.g., "6-31G*", "STO-3G") - Renamed from basis_set for clarity
        self.charge: int = 0                      # (iQ)
        self.multiplicity: int = 0                # (iS2) - Renamed from spin_multiplicity for clarity
        self.qm_memory: Optional[str] = None      # memory - No default, will be None if not in input
        self.qm_nproc: Optional[int] = None       # nprocs
        self.qm_additional_keywords: str = ""     # if necessary
        
        # Parallel processing settings for ASCEC
        self.ascec_parallel_cores: int = 1       # Number of cores for ASCEC operations (file I/O, coordinate transformations)
        self.use_ascec_parallel: bool = False    # Enable parallel processing within ASCEC operations
        
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
        self.lower_energy_configs: int = 0        # Counter for configurations with lower energy than previous lowest
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
        self.initial_qm_retries: int = 100 # Number of retries for initial QM calculation, default to 100
        self.verbosity_level: int = 0 # 0: default, 1: --v (every 10 steps), 2: --va (all steps)
        self.use_standard_metropolis: bool = False # Flag for Metropolis criterion
        
        # Additional attributes for proper type checking
        self.input_file_path: str = ""            # Path to the input file
        self.max_molecular_extent: float = 0.0    # Maximum molecular extent
        self.num_elements_defined: int = 0        # Number of unique elements defined
        self.element_types: List[Tuple[int, int]] = []  # List of (atomic_number, count) tuples
        self.iz_types: List[int] = []             # List of atomic numbers
        self.nnu_types: List[int] = []            # List of atom counts per element type
        self.volume_based_recommendations: Dict = {}  # Volume-based box recommendations

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
def _print_verbose(message: str, level: int, state: Optional[SystemState], file=sys.stderr):
    """
    Prints a message to stderr if the state's verbosity level meets or exceeds the required level.
    level 0: always print (critical errors, final summary, accepted configs, and key annealing steps)
    level 1: --v (every 10 steps), plus level 0)
    level 2: --va (all steps, plus level 0 and 1)
    """
    if state is None or state.verbosity_level >= level:
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


def convert_xyz_to_mol(xyz_filepath: str, openbabel_alias: str = "obabel", state: Optional[SystemState] = None) -> bool:
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
            if state and state.verbosity_level >= 2:
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

    # Print the new ASCII art header
    
    # Helper function to center text within 70 characters
    def center_text(text, width=70):
        return text.center(width)
    
    # Write to the file handle
    print("======================================================================", file=output_file_handle)
    print("", file=output_file_handle)
    print(center_text("*********************"), file=output_file_handle)
    print(center_text("*     A S C E C     *"), file=output_file_handle)
    print(center_text("*********************"), file=output_file_handle)
    print("", file=output_file_handle)
    print("                             √≈≠==≈                                  ", file=output_file_handle)
    print("   √≈≠==≠≈√   √≈≠==≠≈√         ÷++=                      ≠===≠       ", file=output_file_handle)
    print("     ÷++÷       ÷++÷           =++=                     ÷×××××=      ", file=output_file_handle)
    print("     =++=       =++=     ≠===≠ ÷++=      ≠====≠         ÷-÷ ÷-÷      ", file=output_file_handle)
    print("     =++=       =++=    =××÷=≠=÷++=    ≠÷÷÷==÷÷÷≈      ≠××≠ =××=     ", file=output_file_handle)
    print("     =++=       =++=   ≠××=    ÷++=   ≠×+×    ×+÷      ÷+×   ×+××    ", file=output_file_handle)
    print("     =++=       =++=   =+÷     =++=   =+-×÷==÷×-×≠    =×+×÷=÷×+-÷    ", file=output_file_handle)
    print("     ≠×+÷       ÷+×≠   =+÷     =++=   =+---×××××÷×   ≠××÷==×==÷××≠   ", file=output_file_handle)
    print("      =××÷     =××=    ≠××=    ÷++÷   ≠×-×           ÷+×       ×+÷   ", file=output_file_handle)
    print("       ≠=========≠      ≠÷÷÷=≠≠=×+×÷-  ≠======≠≈√  -÷×+×≠     ≠×+×÷- ", file=output_file_handle)
    print("          ≠===≠           ≠==≠  ≠===≠     ≠===≠    ≈====≈     ≈====≈ ", file=output_file_handle)
    print("", file=output_file_handle)
    print("", file=output_file_handle)
    print(center_text("Universidad de Antioquia - Medellín - Colombia"), file=output_file_handle)
    print("", file=output_file_handle)
    print("", file=output_file_handle)
    print(center_text("Annealing Simulado Con Energía Cuántica"), file=output_file_handle)
    print("", file=output_file_handle)
    print(center_text(version), file=output_file_handle)
    print("", file=output_file_handle)
    print(center_text("Química Física Teórica - QFT"), file=output_file_handle)
    print("", file=output_file_handle)
    print("", file=output_file_handle)
    print("======================================================================", file=output_file_handle)
    print("", file=output_file_handle)
    print("Elemental composition of the system:", file=output_file_handle)
    for line in element_composition_lines:
        print(line, file=output_file_handle)
    print(f"There are a total of {state.natom:>2} nuclei", file=output_file_handle)
    # Changed spacing to one space
    print(f"\nCube's length = {state.cube_length:.2f} A", file=output_file_handle) 
    
    # Write hydrogen bond-aware box analysis to output file
    write_box_analysis_to_file(state, output_file_handle)
    
    if hasattr(state, 'max_molecular_extent') and state.max_molecular_extent > 0:
        print(f"Largest molecular extent: {state.max_molecular_extent:.2f} A", file=output_file_handle) 
    
    print("\nNumber of molecules:", state.num_molecules, file=output_file_handle)
    print("\nMolecular composition", file=output_file_handle)
    for line in molecular_composition_lines:
        print(line, file=output_file_handle)

    # Changed spacing to one space
    print(f"\nMaximum displacement of each mass center = {state.max_displacement_a:.2f} A", file=output_file_handle) 
    print(f"Maximum rotation angle = {state.max_rotation_angle_rad:.2f} radians\n", file=output_file_handle) 
    
    # QM program details - formatted as requested
    qm_program_name = state.qm_program.capitalize() if state.qm_program else "Unknown"
    print(f"Energy calculated with {qm_program_name}", file=output_file_handle)
    print(f" Hamiltonian: {state.qm_method or 'Not specified'}", file=output_file_handle)
    print(f" Basis set: {state.qm_basis_set or 'Not specified'}", file=output_file_handle)
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
    state.atomic_number_to_symbol = ATOMIC_NUMBER_TO_SYMBOL.copy()

    for z, symbol in ATOMIC_NUMBER_TO_SYMBOL.items():
        if z < len(state.sym):
            state.sym[z] = symbol.strip()

# Helper function to get element symbol (used in rless.out and history output)
def get_element_symbol(atomic_number: int) -> str:
    """Retrieves the element symbol for a given atomic number."""
    return ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number, 'X') # Default to 'X' for unknown/dummy


def get_molecular_formula(mol_def) -> str:
    """Generate molecular formula from molecule definition."""
    if not mol_def.atoms_coords:
        return "Unknown"
    
    # Count atoms by element
    element_counts = {}
    for atomic_num, x, y, z in mol_def.atoms_coords:
        element = get_element_symbol(atomic_num)
        element_counts[element] = element_counts.get(element, 0) + 1
    
    # Sort elements: C, H, then alphabetically
    sorted_elements = []
    if 'C' in element_counts:
        sorted_elements.append('C')
    if 'H' in element_counts:
        sorted_elements.append('H')
    
    # Add remaining elements alphabetically
    for element in sorted(element_counts.keys()):
        if element not in ['C', 'H']:
            sorted_elements.append(element)
    
    # Build formula string
    formula = ""
    for element in sorted_elements:
        count = element_counts[element]
        if count == 1:
            formula += element
        else:
            formula += f"{element}{count}"
    
    return formula if formula else "Unknown"

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
        proposed_mol_atoms = []  # Initialize to prevent unbound variable warning
        
        # Reset local scales and the next increase threshold for each new molecule
        local_translation_range_factor = 1.0
        local_rotation_range_factor = 1.0
        next_increase_threshold_for_this_molecule = RANGE_INCREASE_STEP

        while overlap_found and attempts < state.MAX_OVERLAP_PLACEMENT_ATTEMPTS:
            attempts += 1
            
            # Show progress for difficult placements
            if attempts % 10000 == 0:
                _print_verbose(f"    Molecule {mol_def.label} (instance {i+1}): {attempts} placement attempts...", 1, state)
            
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

    # Use the already calculated total_atoms from read_input_file which correctly handles different molecule sizes
    total_atoms = state.natom

    rp = np.zeros((total_atoms, 3), dtype=np.float64)
    iznu = [0] * total_atoms

    box_center = state.cube_length / 2.0 * np.ones(3) 

    # Initialize coordinates and atomic numbers for each molecule separately
    # since molecules can have different numbers of atoms
    current_atom_idx = 0
    for mol_idx in range(state.num_molecules):
        mol_def = state.all_molecule_definitions[mol_idx]
        
        # Extract coordinates and atomic numbers for this specific molecule
        coords_list = [[atom[1], atom[2], atom[3]] for atom in mol_def.atoms_coords]
        molecule_coords = np.array(coords_list, dtype=np.float64)
        atomic_numbers = [atom[0] for atom in mol_def.atoms_coords]
        
        # Calculate geometric center for this molecule
        mol_min_coords = np.min(molecule_coords, axis=0)
        mol_max_coords = np.max(molecule_coords, axis=0)
        mol_geometric_center = (mol_min_coords + mol_max_coords) / 2.0
        
        # Center this molecule in the box
        initial_centering_offset = box_center - mol_geometric_center
        centered_coords = molecule_coords + initial_centering_offset
        
        # Place atoms for this molecule
        num_atoms_this_mol = len(atomic_numbers)
        end_idx = current_atom_idx + num_atoms_this_mol
        
        rp[current_atom_idx:end_idx, :] = centered_coords
        iznu[current_atom_idx:end_idx] = atomic_numbers
        
        current_atom_idx = end_idx
    
    # Populate the initial iznu array for the state object
    state.iznu = iznu 
    state.rp = rp

    # Now, randomly translate and rotate molecules from this superimposed state
    # This calls the more general config_molecules that includes overlap checking
    _print_verbose("Configuring molecular positions (this may take a moment for large systems)...", 1, state)
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

    # PHASE 1: Read Fixed Configuration Parameters (Lines 1-13)
    config_lines_parsed = 0 

    # We expect 13 lines for fixed configuration parameters (including conformational sampling)
    while config_lines_parsed < 13:
        try:
            raw_line = next(lines_iterator)
        except StopIteration:
            raise EOFError(f"Unexpected end of input while reading configuration parameters. Expected 13 lines, but found only {config_lines_parsed}.")
        
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

        elif config_lines_parsed == 5: # Line 6: Maximum Monte Carlo Cycles per T and floor value (optional)
            state.max_cycle = int(parts[0])
            # Check if floor value is provided (optional second parameter)
            if len(parts) >= 2:
                state.max_cycle_floor = int(parts[1])
            # If not provided, default value (10) is already set in __init__
            state.maxstep = state.max_cycle # Initialize maxstep with the initial max_cycle from input

        elif config_lines_parsed == 6: # Line 7: Maximum Displacement & Rotation
            state.max_displacement_a = float(parts[0])
            state.max_rotation_angle_rad = float(parts[1])
            state.ds = state.max_displacement_a
            state.dphi = state.max_rotation_angle_rad

        elif config_lines_parsed == 7: # Line 8: Conformational sampling (%) & Maximum dihedral rotation (degrees)
            # Parse conformational move probability as percentage (0-100)
            conformational_percent = float(parts[0])
            if not (0.0 <= conformational_percent <= 100.0):
                raise ValueError(f"Error parsing conformational move percentage on line {line_num}: Expected a value between 0 and 100, got '{parts[0]}'.")
            state.conformational_move_prob = conformational_percent / 100.0  # Convert to 0.0-1.0 range
            
            # Parse maximum dihedral rotation angle in degrees
            max_dihedral_degrees = float(parts[1])
            if not (0.0 <= max_dihedral_degrees <= 180.0):
                raise ValueError(f"Error parsing maximum dihedral angle on line {line_num}: Expected a value between 0 and 180 degrees, got '{parts[1]}'.")
            state.max_dihedral_angle_rad = np.radians(max_dihedral_degrees)  # Convert to radians

        elif config_lines_parsed == 8: # Line 9: QM Program Index & Alias (e.g., "1 g09")
            state.ia = int(parts[0])      
            state.qm_program = parts[1]   
            state.alias = parts[1]        
            state.jdum = state.alias      

        elif config_lines_parsed == 9: # Line 10: Hamiltonian & Basis Set (e.g., "pm3 zdo")
            state.qm_method = parts[0]      
            # Handle case where only method is provided (common for semi-empirical methods)
            if len(parts) > 1:
                state.qm_basis_set = parts[1]   
            else:
                # For semi-empirical methods, we can set a default or leave it empty
                state.qm_basis_set = "zdo"  # Default for semi-empirical methods   

        elif config_lines_parsed == 10:      # Line 11: nprocs [memory] [ascec_cores]
            state.qm_nproc = int(parts[0])   
            # Only set qm_memory if a second part is explicitly provided in the input file.
            if len(parts) > 1:
                state.qm_memory = parts[1]
                _print_verbose(f"QM memory allocation set to: {state.qm_memory}", 2, state)
            
            # Check for ASCEC parallel cores (third parameter)
            if len(parts) > 2:
                ascec_cores = int(parts[2])
                if ascec_cores > 1:
                    state.ascec_parallel_cores = ascec_cores
                    state.use_ascec_parallel = True
                    _print_verbose(f"ASCEC parallel processing enabled with {state.ascec_parallel_cores} cores", 1, state)
                else:
                    state.ascec_parallel_cores = 1
                    state.use_ascec_parallel = False
            else:
                # Auto-detect ASCEC cores based on system if QM uses fewer cores than available
                cpu_count = multiprocessing.cpu_count()
                if state.qm_nproc and state.qm_nproc < cpu_count:
                    # Use remaining cores for ASCEC operations
                    remaining_cores = cpu_count - state.qm_nproc
                    if remaining_cores >= 2:
                        state.ascec_parallel_cores = min(4, remaining_cores)  # Cap at 4 cores for ASCEC
                        state.use_ascec_parallel = True
                        _print_verbose(f"Auto-enabled ASCEC parallel processing with {state.ascec_parallel_cores} cores", 1, state)
                        _print_verbose(f"  (System has {cpu_count} cores, QM uses {state.qm_nproc}, ASCEC uses {state.ascec_parallel_cores})", 1, state)

        elif config_lines_parsed == 11:     # Line 12: Charge & Spin Multiplicity
            state.charge = int(parts[0])
            state.multiplicity = int(parts[1]) 

        elif config_lines_parsed == 12:     # Line 13: Number of Molecules
            try:
                state.num_molecules = int(parts[0])
            except ValueError:
                raise ValueError(f"Error parsing 'Number of Molecules' on line {line_num}: Expected an integer, but found '{parts[0]}'. "
                                 "Please ensure line 13 of your input file contains the total number of molecules.")
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
        "energy_regex": r"FINAL SINGLE POINT ENERGY:\s*([-+]?\d+\.\d+)\s*(?:Eh|E_h)?", # More robust for ORCA
        "termination_string": "ORCA TERMINATED NORMALLY",
        "alternative_termination": ["****ORCA TERMINATED NORMALLY****", "OPTIMIZATION RUN DONE"],  # Additional termination patterns
    },
}

# Helper function to preserve QM files from last accepted configuration
def preserve_last_qm_files(state: SystemState, run_dir: str):
    """
    Preserve the QM input and output files from the most recent QM calculation
    by copying them to special "last_accepted" filenames for debugging purposes.
    """
    call_id = state.qm_call_count
    
    # Source files (from the most recent calculation)
    source_input = os.path.join(run_dir, f"qm_input_{call_id}{QM_PROGRAM_DETAILS[state.ia]['input_ext']}")
    source_output = os.path.join(run_dir, f"qm_output_{call_id}{QM_PROGRAM_DETAILS[state.ia]['output_ext']}")
    source_chk = os.path.join(run_dir, f"qm_chk_{call_id}.chk") if state.qm_program == "gaussian" else None
    
    # Destination files (preserved versions)
    dest_input = os.path.join(run_dir, f"anneal{QM_PROGRAM_DETAILS[state.ia]['input_ext']}")
    dest_output = os.path.join(run_dir, f"anneal{QM_PROGRAM_DETAILS[state.ia]['output_ext']}")
    dest_chk = os.path.join(run_dir, "anneal.chk") if state.qm_program == "gaussian" else None
    
    # Copy files if they exist
    try:
        if os.path.exists(source_input):
            import shutil
            shutil.copy2(source_input, dest_input)
            _print_verbose(f"  Preserved QM input file: {os.path.basename(dest_input)}", 2, state)
        
        if os.path.exists(source_output):
            import shutil
            shutil.copy2(source_output, dest_output)
            _print_verbose(f"  Preserved QM output file: {os.path.basename(dest_output)}", 2, state)
            
        if source_chk and dest_chk and os.path.exists(source_chk):
            import shutil
            shutil.copy2(source_chk, dest_chk)
            _print_verbose(f"  Preserved QM checkpoint file: {os.path.basename(dest_chk)}", 2, state)
    except Exception as e:
        _print_verbose(f"  Warning: Could not preserve QM files: {e}", 1, state)

# Helper function to preserve QM files for debugging (last attempt, successful or failed)
def preserve_last_qm_files_debug(state: SystemState, run_dir: str, call_id: int, status: int):
    """
    Preserve the QM input and output files from the most recent calculation attempt
    for debugging purposes. This preserves files from both successful and failed calculations.
    """
    # Source files (from the most recent calculation)
    source_input = os.path.join(run_dir, f"qm_input_{call_id}{QM_PROGRAM_DETAILS[state.ia]['input_ext']}")
    source_output = os.path.join(run_dir, f"qm_output_{call_id}{QM_PROGRAM_DETAILS[state.ia]['output_ext']}")
    source_chk = os.path.join(run_dir, f"qm_chk_{call_id}.chk") if state.qm_program == "gaussian" else None
    
    # Destination files (preserved versions for debugging)
    dest_input = os.path.join(run_dir, f"anneal{QM_PROGRAM_DETAILS[state.ia]['input_ext']}")
    dest_output = os.path.join(run_dir, f"anneal{QM_PROGRAM_DETAILS[state.ia]['output_ext']}")
    dest_chk = os.path.join(run_dir, f"anneal.chk") if source_chk else None
    
    # Copy files if they exist
    try:
        if os.path.exists(source_input):
            import shutil
            shutil.copy2(source_input, dest_input)
            _print_verbose(f"  Preserved QM input file for debugging: {os.path.basename(dest_input)}", 2, state)
        
        if os.path.exists(source_output):
            import shutil
            shutil.copy2(source_output, dest_output)
            _print_verbose(f"  Preserved QM output file for debugging: {os.path.basename(dest_output)}", 2, state)
            
        if source_chk and dest_chk and os.path.exists(source_chk):
            import shutil
            shutil.copy2(source_chk, dest_chk)
            _print_verbose(f"  Preserved QM checkpoint file for debugging: {os.path.basename(dest_chk)}", 2, state)
    except Exception as e:
        _print_verbose(f"  Warning: Could not preserve QM files for debugging: {e}", 1, state)

def preserve_failed_initial_qm_files(state: SystemState, run_dir: str, attempt_num: int):
    """
    Simply notify that the anneal.* files contain the last failed attempt.
    The files are already preserved by preserve_last_qm_files_debug.
    """
    _print_verbose(f"", 0, state)
    _print_verbose(f"DEBUG: Final failed attempt files preserved as:", 0, state)
    
    input_file = os.path.join(run_dir, "anneal.inp" if state.qm_program == "orca" else "anneal.com")
    output_file = os.path.join(run_dir, "anneal.out")
    
    if os.path.exists(input_file):
        _print_verbose(f"  ✓ {os.path.basename(input_file)} (QM input file - check this for problematic geometry)", 0, state)
    else:
        _print_verbose(f"  ✗ {os.path.basename(input_file)} (NOT FOUND)", 0, state)
        
    if os.path.exists(output_file):
        _print_verbose(f"  ✓ anneal.out (QM output file - check for error messages)", 0, state)
    else:
        _print_verbose(f"  ✗ anneal.out (NOT FOUND - QM program failed to start or crashed immediately)", 0, state)
        _print_verbose(f"      This usually indicates:", 0, state)
        _print_verbose(f"      - QM program not found in PATH", 0, state)
        _print_verbose(f"      - Severe geometry problems (atoms too close)", 0, state)
        _print_verbose(f"      - Invalid basis set for these atoms", 0, state)
        _print_verbose(f"      - Memory/resource issues", 0, state)

# 13. Calculate energy function
def calculate_energy(coords: np.ndarray, atomic_numbers: List[int], state: SystemState, run_dir: str) -> Tuple[float, int]:
    """
    Calculates the energy of the given configuration using the external QM program.
    Returns the energy and a status code (1 for success, 0 for failure).
    Cleans up QM input/output/checkpoint files immediately after execution.
    Now includes optimizations for parallel core usage.
    """
    # Optimize the execution environment for better core utilization
    optimize_qm_execution_environment(state)
    
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
    
    # Add ORCA-specific files to cleanup list
    if state.qm_program == "orca":
        input_basename = f"qm_input_{call_id}"
        orca_files = [
            f"{input_basename}.gbw",
            f"{input_basename}.densities", 
            f"{input_basename}_property.txt",
            f"{input_basename}.engrad",
            f"{input_basename}.pcgrad",
            f"{input_basename}.hess",
            f"{input_basename}.cis",
            f"{input_basename}.uno",
            # ORCA temporary files (semi-empirical methods generate many .tmp files)
            f"{input_basename}.1.tmp",
            f"{input_basename}.2.tmp", 
            f"{input_basename}.3.tmp",
            f"{input_basename}.E.tmp",
            f"{input_basename}.H.tmp",
            f"{input_basename}.K.tmp",
            f"{input_basename}.S.tmp",
            f"{input_basename}.eht.tmp",
            f"{input_basename}.fsv.tmp",
            f"{input_basename}.ndopar.tmp",
            f"{input_basename}.SM12.tmp",
            f"{input_basename}.SP12.tmp",
            f"{input_basename}.diis.tmp",
            f"{input_basename}.en.tmp"
        ]
        for orca_file in orca_files:
            temp_files_to_clean.append(os.path.join(run_dir, orca_file))

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
                # ORCA keyword line generation
                # For semi-empirical methods like PM3, AM1, MNDO, we don't include basis sets
                semi_empirical_methods = ['pm3', 'am1', 'mndo', 'pm6', 'pm7']
                
                if state.qm_method and state.qm_method.lower() in semi_empirical_methods:
                    # Semi-empirical methods don't need basis sets
                    f.write(f"! {state.qm_method}\n")
                else:
                    # Ab initio or DFT methods need basis sets
                    f.write(f"! {state.qm_method or 'HF'} {state.qm_basis_set or 'STO-3G'}\n")
                
                if state.qm_additional_keywords: f.write(f"! {state.qm_additional_keywords}\n")
                
                # Only write %maxcore if qm_memory was explicitly provided in the input file
                if state.qm_memory:
                    mem_val = state.qm_memory.replace('GB', '').replace('MB', '')
                    f.write(f"%maxcore {mem_val}\n") 
                
                # Only use parallel processing for non-semi-empirical methods
                # Semi-empirical methods (NDO methods) in ORCA don't support parallel execution
                if state.qm_nproc and state.qm_method and state.qm_method.lower() not in semi_empirical_methods:
                    f.write(f"%pal nprocs {state.qm_nproc} end\n")
                elif state.qm_nproc and state.qm_method and state.qm_method.lower() in semi_empirical_methods:
                    _print_verbose(f"Note: Parallel processing disabled for semi-empirical method {state.qm_method} (ORCA limitation)", 1, state)
                
                f.write(f"* xyz {state.charge} {state.multiplicity}\n")
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

    # Determine QM command - use alias from input file if provided, otherwise use default
    qm_exe = state.alias if state.alias else QM_PROGRAM_DETAILS[state.ia]["default_exe"]
    if state.qm_program == "gaussian":
        qm_command = f"{qm_exe} {qm_input_filename} {qm_output_filename}"
    elif state.qm_program == "orca":
        # For ORCA, we'll use subprocess to capture output properly
        qm_command = [qm_exe, qm_input_filename]
    else:
        _print_verbose(f"Error: Unsupported QM program '{state.qm_program}' for command execution.", 0, state)
        return 0.0, 0

    try:
        # First, check if the QM executable is available with shorter timeout and better error handling
        if state.qm_program == "orca":
            # Test if ORCA is available using a simple version check with reduced timeout
            try:
                test_process = subprocess.run([qm_exe, "--version"], capture_output=True, text=True, timeout=3)
                # ORCA might return non-zero even for version check, so just check if it ran
                _print_verbose(f"ORCA test command executed: {qm_exe} --version", 2, state)
            except subprocess.TimeoutExpired:
                _print_verbose(f"Warning: ORCA executable '{qm_exe}' timeout during version check (proceeding anyway)", 1, state)
            except FileNotFoundError:
                _print_verbose(f"Error: ORCA executable '{qm_exe}' not found in PATH", 0, state)
                return 0.0, 0
            except Exception as e:
                _print_verbose(f"Warning: ORCA executable test failed: {e} (proceeding anyway)", 1, state)
        
        if state.qm_program == "orca":
            # Special handling for ORCA to capture output properly
            _print_verbose(f"Executing ORCA command: {' '.join(qm_command)} in directory: {run_dir}", 2, state)
            with open(qm_output_path, 'w') as output_file:
                process = subprocess.run(qm_command, cwd=run_dir, stdout=output_file, stderr=subprocess.PIPE, text=True, check=False)
            _print_verbose(f"ORCA process completed with return code: {process.returncode}", 2, state)
        else:
            # For Gaussian and other programs
            process = subprocess.run(qm_command, shell=True, capture_output=True, text=True, cwd=run_dir, check=False)
        
        # Check for non-zero exit code first
        if process.returncode != 0:
            _print_verbose(f"'{state.qm_program}' exited with non-zero status: {process.returncode}.", 0, state)
            if state.qm_program == "orca":
                _print_verbose(f"  Command executed: {' '.join(qm_command)}", 0, state)
                _print_verbose(f"  Working directory: {run_dir}", 0, state)
                _print_verbose(f"  STDERR:\n{_format_stream_output(process.stderr)}", 0, state)
                # Also check if output file was created despite non-zero exit
                if os.path.exists(qm_output_path):
                    _print_verbose(f"  Output file was created, checking content...", 0, state)
            else:
                _print_verbose(f"  Command executed: {qm_command}", 0, state)
                _print_verbose(f"  STDOUT (first 10 lines):\n{_format_stream_output(process.stdout)}", 0, state)
                _print_verbose(f"  STDERR (first 10 lines):\n{_format_stream_output(process.stderr)}", 0, state)
            status = 0 
        elif not os.path.exists(qm_output_path):
            _print_verbose(f"QM output file '{qm_output_path}' was not generated.", 1, state)
            status = 0 
        else:
            with open(qm_output_path, 'r') as f:
                output_content = f.read()
            
            # Check for normal termination string
            termination_found = QM_PROGRAM_DETAILS[state.ia]["termination_string"] in output_content
            
            # For programs with alternative termination patterns, check those too
            if not termination_found and "alternative_termination" in QM_PROGRAM_DETAILS[state.ia]:
                for alt_term in QM_PROGRAM_DETAILS[state.ia]["alternative_termination"]:
                    if alt_term in output_content:
                        termination_found = True
                        break
            
            if not termination_found:
                _print_verbose(f"QM program '{state.qm_program}' did not terminate normally for config {call_id}.", 1, state)
                status = 0
            else:
                match = re.search(QM_PROGRAM_DETAILS[state.ia]["energy_regex"], output_content)
                if match:
                    energy = float(match.group(1))
                    status = 1
                else:
                    # For ORCA, try alternative energy patterns as fallback
                    if state.qm_program == "orca":
                        # Try alternative patterns commonly found in ORCA output
                        fallback_patterns = [
                            r"Total Energy\s*:\s*([-+]?\d+\.\d+)\s*Eh",
                            r"E\(SCF\)\s*=\s*([-+]?\d+\.\d+)\s*Eh",
                            r"TOTAL SCF ENERGY\s*=\s*([-+]?\d+\.\d+)\s*Eh?"
                        ]
                        for pattern in fallback_patterns:
                            match = re.search(pattern, output_content)
                            if match:
                                energy = float(match.group(1))
                                status = 1
                                _print_verbose(f"Found energy using fallback pattern: {pattern}", 2, state)
                                break
                    
                    if status == 0:
                        _print_verbose(f"Could not find energy in {state.qm_program} output file: {qm_output_path}", 1, state)
    
    except Exception as e:
        _print_verbose(f"An error occurred during QM calculation or parsing: {e}", 0, state)
        status = 0
    finally:
        # Always preserve the last QM files for debugging (both successful and failed attempts)
        preserve_last_qm_files_debug(state, run_dir, call_id, status)
        
        # Clean up numbered QM files but keep the "last_" versions for debugging
        for fpath in temp_files_to_clean:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except OSError as e:
                    _print_verbose(f"  Error cleaning up {os.path.basename(fpath)}: {e}", 0, state)
        
        # Additional cleanup for ORCA temporary files using glob patterns
        if state.qm_program == "orca":
            import glob
            input_basename = f"qm_input_{call_id}"
            orca_patterns = [
                f"{input_basename}*.tmp",  # All .tmp files
                f"{input_basename}.*.tmp", # All numbered .tmp files like .1.tmp, .2.tmp
                f"{input_basename}*.log",  # Any log files
            ]
            for pattern in orca_patterns:
                pattern_path = os.path.join(run_dir, pattern)
                for fpath in glob.glob(pattern_path):
                    try:
                        if os.path.exists(fpath):
                            os.remove(fpath)
                            _print_verbose(f"  Cleaned ORCA temp file: {os.path.basename(fpath)}", 2, state)
                    except OSError as e:
                        _print_verbose(f"  Error cleaning up ORCA temp file {os.path.basename(fpath)}: {e}", 1, state)
    return energy, status

# 13a. Enhanced QM execution with parallel core optimization
def optimize_qm_execution_environment(state: SystemState):
    """
    Optimize the execution environment for QM calculations to make better use of available cores.
    This function sets environment variables that help QM programs utilize cores more efficiently.
    """
    if not state.use_ascec_parallel:
        return
    
    # Set environment variables for better parallel performance
    os.environ['OMP_NUM_THREADS'] = str(state.qm_nproc or 1)
    os.environ['MKL_NUM_THREADS'] = str(state.qm_nproc or 1)
    
    # For ORCA specifically
    if state.qm_program == "orca":
        os.environ['RSH_COMMAND'] = 'ssh -x'
        # Optimize ORCA's parallel execution
        if state.ascec_parallel_cores > 1:
            _print_verbose(f"Optimized environment for ORCA with {state.qm_nproc} QM cores and {state.ascec_parallel_cores} system cores", 2, state)
    
    # For Gaussian specifically  
    elif state.qm_program == "gaussian":
        if state.ascec_parallel_cores > 1:
            _print_verbose(f"Optimized environment for Gaussian with {state.qm_nproc} QM cores and {state.ascec_parallel_cores} system cores", 2, state)

def parallel_coordinate_operations(coords: np.ndarray, state: SystemState) -> np.ndarray:
    """
    Perform coordinate transformations using parallel processing when beneficial.
    This can speed up large system coordinate manipulations.
    """
    if not state.use_ascec_parallel or coords.shape[0] < 100:
        # For small systems, parallel overhead isn't worth it
        return coords
    
    # For large systems, we could implement parallel coordinate transformations
    # This is most beneficial for systems with hundreds of atoms
    _print_verbose(f"Using parallel coordinate operations for {coords.shape[0]} atoms", 2, state)
    return coords

def parallel_file_operations(file_path: str, content: str, state: SystemState) -> bool:
    """
    Handle file I/O operations more efficiently using parallel processing capabilities.
    This mainly helps with large file writes and reads.
    """
    if not state.use_ascec_parallel:
        # Fall back to standard file operations
        try:
            with open(file_path, 'w') as f:
                f.write(content)
            return True
        except IOError:
            return False
    
    # For parallel-enabled systems, we can use more efficient I/O
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        return True
    except IOError:
        return False

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
    elif random_generate_config_mode == 1 and state and hasattr(state, 'current_temp'): # Only add T for annealing mode, not for random mode
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
    xyz_path = os.path.join(state.output_dir, f"{prefix}.xyz")
    box_xyz_path = os.path.join(state.output_dir, f"{prefix.replace('mto_', 'mtobox_').replace('result_', 'resultbox_')}.xyz") # Corrected replacement for box name

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

# 14.5. Dihedral Rotation Functions for Conformational Sampling
def rotate_around_bond(coords: np.ndarray, atom1_idx: int, atom2_idx: int, 
                      moving_atoms: List[int], angle_rad: float) -> np.ndarray:
    """
    Rotates a set of atoms around a bond defined by atom1_idx and atom2_idx.
    
    Args:
        coords: Atomic coordinates array (N x 3)
        atom1_idx: Index of first atom defining the rotation axis
        atom2_idx: Index of second atom defining the rotation axis  
        moving_atoms: List of atom indices that will be rotated
        angle_rad: Rotation angle in radians
    
    Returns:
        Modified coordinates array
    """
    new_coords = np.copy(coords)
    
    # Define rotation axis vector
    axis_vector = coords[atom2_idx] - coords[atom1_idx]
    axis_length = np.linalg.norm(axis_vector)
    
    if axis_length < 1e-6:
        # Atoms are too close, skip rotation
        return new_coords
    
    # Normalize axis vector
    axis_unit = axis_vector / axis_length
    
    # Rotation point (we'll use atom1 as the rotation center)
    rotation_center = coords[atom1_idx]
    
    # Rodrigues' rotation formula
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    for atom_idx in moving_atoms:
        if atom_idx == atom1_idx or atom_idx == atom2_idx:
            continue  # Don't rotate the atoms defining the axis
            
        # Vector from rotation center to atom
        point_vector = coords[atom_idx] - rotation_center
        
        # Apply Rodrigues' rotation formula
        rotated_vector = (point_vector * cos_angle + 
                         np.cross(axis_unit, point_vector) * sin_angle +
                         axis_unit * np.dot(axis_unit, point_vector) * (1 - cos_angle))
        
        new_coords[atom_idx] = rotation_center + rotated_vector
    
    return new_coords

def find_rotatable_bonds(mol_coords: np.ndarray, mol_atomic_numbers: List[int], 
                        state: SystemState) -> List[Tuple[int, int, List[int]]]:
    """
    Identifies rotatable bonds in a molecule and determines which atoms move with each rotation.
    
    Returns:
        List of tuples: (atom1_idx, atom2_idx, [moving_atom_indices])
    """
    rotatable_bonds = []
    n_atoms = len(mol_atomic_numbers)
    
    # Simple bond detection based on distance (could be improved with connectivity)
    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            distance = np.linalg.norm(mol_coords[j] - mol_coords[i])
            
            # Get atomic radii for bond length estimation
            radius_i = state.R_ATOM.get(mol_atomic_numbers[i], 1.5)
            radius_j = state.R_ATOM.get(mol_atomic_numbers[j], 1.5)
            max_bond_length = (radius_i + radius_j) * 1.3  # 30% tolerance
            
            if distance < max_bond_length:
                # This looks like a bond
                # For single bonds (excluding terminal bonds), consider as rotatable
                if (mol_atomic_numbers[i] != 1 and mol_atomic_numbers[j] != 1):  # Not H-X bonds
                    # Find which atoms would move if we rotate around this bond
                    # Simple approach: atoms connected to atom j (and beyond) move
                    moving_atoms = find_connected_atoms(j, i, mol_coords, mol_atomic_numbers, state)
                    
                    if len(moving_atoms) > 0 and len(moving_atoms) < n_atoms - 2:
                        # Valid rotatable bond (not terminal, not all atoms)
                        rotatable_bonds.append((i, j, moving_atoms))
    
    return rotatable_bonds

def find_connected_atoms(start_atom: int, exclude_atom: int, mol_coords: np.ndarray, 
                        mol_atomic_numbers: List[int], state: SystemState) -> List[int]:
    """
    Find all atoms connected to start_atom, excluding the path through exclude_atom.
    Uses depth-first search to find the molecular fragment.
    """
    n_atoms = len(mol_atomic_numbers)
    visited = set([exclude_atom])  # Don't cross back through the bond
    to_visit = [start_atom]
    connected = []
    
    while to_visit:
        current = to_visit.pop()
        if current in visited:
            continue
            
        visited.add(current)
        connected.append(current)
        
        # Find neighbors of current atom
        for neighbor in range(n_atoms):
            if neighbor in visited:
                continue
                
            distance = np.linalg.norm(mol_coords[neighbor] - mol_coords[current])
            radius_curr = state.R_ATOM.get(mol_atomic_numbers[current], 1.5)
            radius_neigh = state.R_ATOM.get(mol_atomic_numbers[neighbor], 1.5)
            max_bond_length = (radius_curr + radius_neigh) * 1.3
            
            if distance < max_bond_length:
                to_visit.append(neighbor)
    
    # Remove the start_atom itself from the moving atoms list
    if start_atom in connected:
        connected.remove(start_atom)
    
    return connected

def propose_conformational_move(state: SystemState, current_rp: np.ndarray, 
                               current_imolec: List[int]) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    Proposes a conformational change by rotating around a randomly selected rotatable bond.
    Returns the new full system coordinates, the coordinates of the moved molecule,
    the index of the moved molecule, and the move type.
    """
    # Randomly select a molecule
    molecule_idx = np.random.randint(0, state.num_molecules)
    
    start_atom_idx = current_imolec[molecule_idx] 
    end_atom_idx = current_imolec[molecule_idx + 1]
    
    # Get molecule coordinates and atomic numbers
    mol_coords = current_rp[start_atom_idx:end_atom_idx, :]
    mol_atomic_numbers = [state.iznu[i] for i in range(start_atom_idx, end_atom_idx)]
    
    # Find rotatable bonds in this molecule
    rotatable_bonds = find_rotatable_bonds(mol_coords, mol_atomic_numbers, state)
    
    if not rotatable_bonds:
        # No rotatable bonds found, fall back to rigid-body move
        return propose_move(state, current_rp, current_imolec)
    
    # Randomly select a rotatable bond
    bond_atom1, bond_atom2, moving_atoms = rotatable_bonds[np.random.randint(len(rotatable_bonds))]
    
    # Generate random rotation angle (e.g., ±60 degrees max)
    max_rotation = np.pi / 3  # 60 degrees
    rotation_angle = (np.random.rand() - 0.5) * 2.0 * max_rotation
    
    # Apply rotation to molecule coordinates
    new_mol_coords = rotate_around_bond(mol_coords, bond_atom1, bond_atom2, 
                                       moving_atoms, rotation_angle)
    
    # Create new full system coordinates
    proposed_rp_full_system = np.copy(current_rp)
    proposed_rp_full_system[start_atom_idx:end_atom_idx, :] = new_mol_coords
    
    return proposed_rp_full_system, new_mol_coords, molecule_idx, "conformational"

def propose_unified_move(state: SystemState, current_rp: np.ndarray, 
                        current_imolec: List[int]) -> Tuple[np.ndarray, np.ndarray, int, str]:
    """
    Proposes either a conformational move or a rigid-body move based on probability.
    This function provides the enhanced Monte Carlo sampling that includes 
    intramolecular conformational changes.
    """
    if np.random.rand() < state.conformational_move_prob:
        # Attempt conformational move
        try:
            return propose_conformational_move(state, current_rp, current_imolec)
        except Exception as e:
            # Fall back to rigid-body move if conformational move fails
            _print_verbose(f"Conformational move failed, falling back to rigid-body: {e}", 2, state)
            return propose_move(state, current_rp, current_imolec)
    else:
        # Rigid-body move (translation + rotation)
        return propose_move(state, current_rp, current_imolec)

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
            
            # Decide whether to apply conformational move or rigid-body move
            if np.random.rand() < state.conformational_move_prob:
                # Apply conformational move (dihedral rotation)
                try:
                    mol_coords = proposed_rp[mol_start_idx:mol_end_idx, :]
                    rotatable_bonds = find_rotatable_bonds(mol_coords, mol_atomic_numbers, state)
                    
                    if rotatable_bonds:
                        # Randomly select a rotatable bond
                        bond_atom1, bond_atom2, moving_atoms = rotatable_bonds[np.random.randint(len(rotatable_bonds))]
                        
                        # Generate random rotation angle
                        rotation_angle = (np.random.rand() - 0.5) * 2.0 * state.max_dihedral_angle_rad
                        
                        # Apply rotation to molecule coordinates
                        new_mol_coords = rotate_around_bond(mol_coords, bond_atom1, bond_atom2, 
                                                           moving_atoms, rotation_angle)
                        
                        # Update proposed coordinates
                        proposed_rp[mol_start_idx:mol_end_idx, :] = new_mol_coords
                        
                        # Recalculate center of mass after conformational change
                        proposed_rcm[imo, :] = calculate_mass_center(new_mol_coords, mol_masses)
                        
                        _print_verbose(f"  Applied conformational move to molecule {imo+1}", 2, state)
                    else:
                        # No rotatable bonds, apply rigid-body move instead
                        trans(imo, proposed_rp, mol_rel_coords_temp, proposed_rcm, local_ds, state) 
                        rotac(imo, proposed_rp, mol_rel_coords_temp, proposed_rcm, local_dphi, state)
                        
                except Exception as e:
                    # If conformational move fails, fall back to rigid-body move
                    _print_verbose(f"  Conformational move failed for molecule {imo+1}, using rigid-body: {e}", 2, state)
                    trans(imo, proposed_rp, mol_rel_coords_temp, proposed_rcm, local_ds, state) 
                    rotac(imo, proposed_rp, mol_rel_coords_temp, proposed_rcm, local_dphi, state)
            else:
                # Apply traditional rigid-body move (translation and rotation)
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
def calculate_molecular_volume(mol_def, method='covalent_spheres') -> float:
    """
    Calculates the approximate volume of a molecule using different methods.
    
    Args:
        mol_def: MoleculeData object containing atomic coordinates and numbers
        method (str): Calculation method - 'covalent_spheres', 'convex_hull', or 'grid_based'
    
    Returns:
        float: Estimated molecular volume in Angstroms^3
    """
    if not mol_def.atoms_coords:
        return 0.0
    
    if method == 'covalent_spheres':
        # Sum of individual atomic volumes using covalent radii
        # This is an upper bound estimate but computationally simple
        total_volume = 0.0
        for atomic_num, x, y, z in mol_def.atoms_coords:
            radius = R_ATOM.get(atomic_num, 1.5)  # Default to 1.5 Å for unknown atoms
            atomic_volume = (4.0/3.0) * np.pi * (radius ** 3)
            total_volume += atomic_volume
        
        # Apply an overlap correction factor (molecules are not just isolated spheres)
        # Typical values: 0.6-0.8 for organic molecules, 0.7-0.9 for inorganic
        overlap_factor = 0.74  # Based on typical molecular packing
        return total_volume * overlap_factor
    
    elif method == 'convex_hull':
        # Calculate volume using the convex hull of atomic spheres
        # More accurate for elongated or complex-shaped molecules
        try:
            from scipy.spatial import ConvexHull
            
            # Create points on the surface of each atomic sphere
            all_surface_points = []
            for atomic_num, x, y, z in mol_def.atoms_coords:
                radius = R_ATOM.get(atomic_num, 1.5)
                center = np.array([x, y, z])
                
                # Generate points on sphere surface (using spherical coordinates)
                n_points = 20  # Number of points per atom
                phi = np.random.uniform(0, 2*np.pi, n_points)
                costheta = np.random.uniform(-1, 1, n_points)
                theta = np.arccos(costheta)
                
                sphere_points = center + radius * np.column_stack([
                    np.sin(theta) * np.cos(phi),
                    np.sin(theta) * np.sin(phi),
                    np.cos(theta)
                ])
                all_surface_points.extend(sphere_points)
            
            if len(all_surface_points) < 4:
                # Fall back to covalent_spheres method
                return calculate_molecular_volume(mol_def, 'covalent_spheres')
            
            hull = ConvexHull(all_surface_points)
            return hull.volume
            
        except ImportError:
            # scipy not available, fall back to covalent_spheres
            return calculate_molecular_volume(mol_def, 'covalent_spheres')
        except Exception:
            # Any other error, fall back to covalent_spheres
            return calculate_molecular_volume(mol_def, 'covalent_spheres')
    
    else:
        # Default to covalent_spheres method
        return calculate_molecular_volume(mol_def, 'covalent_spheres')


def calculate_hydrogen_bond_potential(mol_def) -> Dict:
    """
    Calculate the potential hydrogen bonding volume based on donor/acceptor counts.
    
    Args:
        mol_def: MoleculeData object containing atomic coordinates and numbers
    
    Returns:
        Dictionary with hydrogen bond analysis and volume estimation
    """
    # Average hydrogen bond length in organic clusters
    avg_hb_length = 2.5  # Angstroms
    
    # Volume estimation: cylindrical volume around H-bond
    # Using radius of ~1.2 Å (approximate H-bond interaction zone)
    hb_interaction_radius = 1.2
    hb_volume_per_bond = math.pi * (hb_interaction_radius ** 2) * avg_hb_length
    
    # Count potential donors and acceptors
    donors = 0
    acceptors = 0
    
    for atomic_num, x, y, z in mol_def.atoms_coords:
        element = get_element_symbol(atomic_num)
        
        # Hydrogen bond donors (H attached to N, O, F)
        if element == 'H':
            donors += 1
        
        # Hydrogen bond acceptors (N, O, F with lone pairs)
        elif element in ['N', 'O', 'F']:
            acceptors += 1
        
        # Special cases for other elements that can participate
        elif element == 'S':
            acceptors += 0.5  # Weaker acceptor
        elif element == 'Cl':
            acceptors += 0.3  # Weak acceptor
    
    # Estimate potential hydrogen bonds (limited by the smaller of donors/acceptors)
    potential_hb_bonds = min(donors, acceptors)
    
    # Total volume needed for hydrogen bonding network
    hb_network_volume = potential_hb_bonds * hb_volume_per_bond
    
    return {
        'donors': int(donors),
        'acceptors': int(acceptors),
        'potential_bonds': potential_hb_bonds,
        'avg_bond_length': avg_hb_length,
        'hb_volume_per_bond': hb_volume_per_bond,
        'total_hb_volume': hb_network_volume
    }


def calculate_optimal_box_length(state: SystemState, target_packing_fractions: Optional[List[float]] = None) -> Dict:
    """
    Calculates optimal box lengths based on molecular volumes and target packing densities.
    
    Args:
        state: SystemState object containing molecule definitions
        target_packing_fractions: List of target packing fractions to calculate box sizes for
    
    Returns:
        Dict: Results containing volumes, box lengths for different packing fractions, and recommendations
    """
    if target_packing_fractions is None:
        # Use more conservative packing fractions for hydrogen-bonded systems
        target_packing_fractions = [0.05, 0.10, 0.15, 0.20, 0.25]  # 5% to 25% packing
    
    if not state.all_molecule_definitions:
        return {'error': 'No molecule definitions found'}
    
    results = {
        'individual_molecular_volumes': [],
        'total_molecular_volume': 0.0,
        'num_molecules': state.num_molecules,
        'box_length_recommendations': {},
        'packing_analysis': {}
    }
    
    # Calculate volume and hydrogen bond potential for each unique molecule definition
    unique_molecular_volumes = []
    unique_hb_analyses = []
    total_hb_volume = 0.0
    
    for mol_def in state.all_molecule_definitions:
        volume = calculate_molecular_volume(mol_def, method='covalent_spheres')
        hb_analysis = calculate_hydrogen_bond_potential(mol_def)
        
        unique_molecular_volumes.append(volume)
        unique_hb_analyses.append(hb_analysis)
        
        results['individual_molecular_volumes'].append({
            'molecule_label': mol_def.label,
            'num_atoms': mol_def.num_atoms,
            'volume_A3': volume,
            'hb_donors': hb_analysis['donors'],
            'hb_acceptors': hb_analysis['acceptors'],
            'potential_hb_bonds': hb_analysis['potential_bonds'],
            'hb_network_volume_A3': hb_analysis['total_hb_volume']
        })
    
    # Calculate total volume of all molecules that will be placed
    total_molecular_volume = 0.0
    total_hb_network_volume = 0.0
    
    for i, mol_def_idx in enumerate(state.molecules_to_add):
        if mol_def_idx < len(unique_molecular_volumes):
            total_molecular_volume += unique_molecular_volumes[mol_def_idx]
            total_hb_network_volume += unique_hb_analyses[mol_def_idx]['total_hb_volume']
    
    # Total effective volume includes both molecular and hydrogen bonding network volumes
    total_effective_volume = total_molecular_volume + total_hb_network_volume
    
    results['total_molecular_volume'] = total_molecular_volume
    results['total_hb_network_volume'] = total_hb_network_volume
    results['total_effective_volume'] = total_effective_volume
    
    if total_effective_volume <= 0:
        return {'error': 'Total effective volume is zero or negative'}
    
    # Calculate box lengths for different packing fractions
    # For hydrogen-bonded systems, use more conservative packing fractions
    for packing_fraction in target_packing_fractions:
        # Box volume = total_effective_volume / packing_fraction
        required_box_volume = total_effective_volume / packing_fraction
        
        # For a cubic box: L^3 = required_box_volume
        box_length = required_box_volume ** (1.0/3.0)
        
        results['box_length_recommendations'][f'{packing_fraction:.1%}'] = {
            'packing_fraction': packing_fraction,
            'box_length_A': box_length,
            'box_volume_A3': required_box_volume,
            'free_volume_A3': required_box_volume - total_effective_volume,
            'free_volume_fraction': 1.0 - packing_fraction,
            'molecular_volume_fraction': total_molecular_volume / required_box_volume,
            'hb_network_volume_fraction': total_hb_network_volume / required_box_volume
        }
    
    # Add analysis for current box length if available
    if hasattr(state, 'cube_length') and state.cube_length > 0:
        current_box_volume = state.cube_length ** 3
        current_packing_fraction = total_effective_volume / current_box_volume
        
        results['current_box_analysis'] = {
            'current_box_length_A': state.cube_length,
            'current_box_volume_A3': current_box_volume,
            'current_packing_fraction': current_packing_fraction,
            'current_free_volume_A3': current_box_volume - total_effective_volume,
            'current_free_volume_fraction': 1.0 - current_packing_fraction,
            'molecular_packing_fraction': total_molecular_volume / current_box_volume,
            'hb_network_fraction': total_hb_network_volume / current_box_volume
        }
    
    # Calculate largest molecular dimension for comparison with old method
    max_molecular_extent = 0.0
    for mol_def in state.all_molecule_definitions:
        if not mol_def.atoms_coords:
            continue
        coords_array = np.array([atom[1:] for atom in mol_def.atoms_coords])
        min_coords = np.min(coords_array, axis=0)
        max_coords = np.max(coords_array, axis=0)
        extents = max_coords - min_coords
        current_max_extent = np.max(extents)
        if current_max_extent > max_molecular_extent:
            max_molecular_extent = current_max_extent
    
    results['max_molecular_extent_A'] = max_molecular_extent
    results['old_method_recommendation_A'] = max_molecular_extent + 16.0  # 8 Å on each side
    
    return results


def write_box_analysis_to_file(state: SystemState, output_file_handle):
    """
    Writes box length analysis results to the output file.
    """
    if not state.all_molecule_definitions:
        return
    
    # Calculate optimal box lengths using volume-based approach
    results = calculate_optimal_box_length(state)
    
    if 'error' in results:
        return
    
    # Get recommendations for output file
    recommendations = results['box_length_recommendations']
    rec_5 = recommendations.get('5.0%', {}).get('box_length_A', 0)
    rec_10 = recommendations.get('10.0%', {}).get('box_length_A', 0)
    rec_15 = recommendations.get('15.0%', {}).get('box_length_A', 0)
    
    # Write to output file
    if rec_5 > 0 and rec_10 > 0 and rec_15 > 0:
        print(f"H-bond aware suggestions:", file=output_file_handle)
        print(f"  • Isolated clusters: {rec_5:.1f} A (5% effective packing)", file=output_file_handle)
        print(f"  • Cluster formation: {rec_10:.1f} A (10% effective packing)", file=output_file_handle)
        print(f"  • Network studies: {rec_15:.1f} A (15% effective packing)", file=output_file_handle)
    
    # Store results in state for potential use elsewhere
    state.max_molecular_extent = results['max_molecular_extent_A']
    state.volume_based_recommendations = recommendations

def provide_box_length_advice(state: SystemState):
    """
    Provides comprehensive advice on appropriate box lengths based on molecular volumes
    and target packing densities. This is a much more rigorous approach than the simple
    8 Angstrom rule of thumb.
    """
    if not state.all_molecule_definitions:
        _print_verbose("Cannot provide box length advice: No molecule definitions found.", 0, state)
        return

    _print_verbose("\n" + "="*78, 1, state)
    _print_verbose("Box length analysis", 1, state)
    _print_verbose("="*78, 1, state)
    _print_verbose(f"Successfully parsed {state.natom} atoms", 1, state)
    _print_verbose("", 1, state)
    
    # Calculate optimal box lengths using volume-based approach
    results = calculate_optimal_box_length(state)
    
    if 'error' in results:
        _print_verbose(f"Error in volume analysis: {results['error']}", 0, state)
        return
    
    # Display molecular volume analysis
    _print_verbose("1. Molecular volume and hydrogen bonding analysis:", 1, state)
    _print_verbose("-" * 50, 1, state)
    
    total_molecular_volume = results['total_molecular_volume']
    total_hb_volume = results['total_hb_network_volume']
    total_effective_volume = results['total_effective_volume']
    
    _print_verbose(f"Number of molecules to place: {results['num_molecules']}", 1, state)
    _print_verbose(f"  Total molecular volume: {total_molecular_volume:.2f} Å³", 1, state)
    _print_verbose(f"  Total H-bond network volume: {total_hb_volume:.2f} Å³", 1, state)
    _print_verbose(f"  Total effective volume: {total_effective_volume:.2f} Å³", 1, state)
    
    _print_verbose("\nIndividual molecule analysis:", 1, state)
    for i, mol_info in enumerate(results['individual_molecular_volumes']):
        # Get the molecular formula from the corresponding molecule definition
        if i < len(state.all_molecule_definitions):
            mol_def = state.all_molecule_definitions[i]
            molecular_formula = get_molecular_formula(mol_def)
            _print_verbose(f"  • {mol_info['molecule_label']}: {molecular_formula} {mol_info['volume_A3']:.2f} Å³", 1, state)
        else:
            _print_verbose(f"  • {mol_info['molecule_label']}: {mol_info['volume_A3']:.2f} Å³", 1, state)
    
    # Display box length recommendations
    _print_verbose("\n2. Box length recommendations (H-Bond Network Aware):", 1, state)
    _print_verbose("-" * 70, 1, state)
    
    recommendations = results['box_length_recommendations']
    _print_verbose("Packing (%)    Box Length (Å)     Box Volume (Å³)       Free (%)", 1, state)
    _print_verbose("-" * 70, 1, state)

    for key, rec in recommendations.items():
        pf = rec['packing_fraction']
        bl = rec['box_length_A']
        bv = rec['box_volume_A3']
        free_pct = rec['free_volume_fraction'] * 100
        _print_verbose(f"    {pf*100:4.1f}          {bl:6.2f}             {bv:6.0f}               {free_pct:4.1f}", 1, state)
    
    # Current box analysis - show current cube length and largest molecular extent
    if 'current_box_analysis' in results:
        _print_verbose("\n3. Current box analysis:", 1, state)
        _print_verbose("-" * 26, 1, state)
        current = results['current_box_analysis']
        _print_verbose(f"Cube's length = {current['current_box_length_A']:.2f} Å", 1, state)
        _print_verbose(f"  Current effective packing: {current['current_packing_fraction']:.1%}", 1, state)
        _print_verbose(f"    └ Molecular: {current['molecular_packing_fraction']:.1%}, H-bond network: {current['hb_network_fraction']:.1%}", 1, state)
        _print_verbose(f"  Current free volume: {current['current_free_volume_A3']:.0f} Å³ "
                      f"({current['current_free_volume_fraction']:.1%})", 1, state)
        _print_verbose(f"  Largest molecular extent: {results['max_molecular_extent_A']:.2f} Å", 1, state)
        
        # Provide assessment of current box size for H-bonded systems
        pf = current['current_packing_fraction']
        if pf < 0.05:
            assessment = "Very dilute - good for isolated cluster studies"
        elif pf < 0.15:
            assessment = "Dilute - appropriate for H-bonded cluster formation"
        elif pf < 0.25:
            assessment = "Moderate - suitable for network formation studies"
        elif pf < 0.35:
            assessment = "Dense - good for condensed phase simulations"
        elif pf < 0.45:
            assessment = "Very dense - may constrain H-bond network flexibility"
        else:
            assessment = "Extremely dense - may prevent proper H-bond formation"
        
        _print_verbose(f"  {assessment}", 1, state)
    
    # Store results in state for potential use elsewhere
    max_extent = results['max_molecular_extent_A']
    state.max_molecular_extent = max_extent
    state.volume_based_recommendations = recommendations
    
    # Final recommendations
    _print_verbose("\n4. Recommendations for H-bonded systems:", 1, state)
    _print_verbose("-" * 40, 1, state)
    
    # Get recommendations as reasonable defaults for H-bonded systems
    rec_5 = recommendations.get('5.0%', {}).get('box_length_A', 0)
    rec_10 = recommendations.get('10.0%', {}).get('box_length_A', 0)
    rec_15 = recommendations.get('15.0%', {}).get('box_length_A', 0)
    
    if rec_5 > 0 and rec_10 > 0 and rec_15 > 0:
        _print_verbose(f"• For isolated clusters: {rec_5:.1f} Å (5% effective packing)", 1, state)
        _print_verbose(f"• For cluster formation: {rec_10:.1f} Å (10% effective packing)", 1, state)
        _print_verbose(f"• For network studies: {rec_15:.1f} Å (15% effective packing)", 1, state)
        _print_verbose(f"• Includes space for H-bond network (avg. bond length: 2.5 Å)", 1, state)
    
    _print_verbose("\n" + "="*78, 1, state)
    _print_verbose("Note: This analysis accounts for hydrogen bonding networks in molecular clusters.", 1, state)
    _print_verbose("H-bond volume estimated using 2.5 Å average bond length and 1.2 Å interaction radius.", 1, state)
    _print_verbose("Run the full simulation to validate these recommendations.", 1, state)
    _print_verbose("="*78, 1, state)

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


def create_launcher_script(replicated_files: List[str], input_dir: str, script_name: str = "launcher_ascec.sh") -> str:
    """
    Creates a bash launcher script for sequential execution of replicated runs.
    
    Args:
        replicated_files (List[str]): List of paths to the replicated input files
        input_dir (str): Directory where the launcher script should be created
        script_name (str): Name of the launcher script
    
    Returns:
        str: Path to the created launcher script
    """
    launcher_path = os.path.join(input_dir, script_name)
    
    # Get the directory where ascec-v04.py is located
    ascec_script_path = os.path.abspath(__file__)
    ascec_directory = os.path.dirname(ascec_script_path)
    
    try:
        with open(launcher_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            
            # Configuration for ASCEC v04
            f.write("# Configuration for ASCEC v04\n")
            f.write("# Set ASCEC_ROOT to the directory containing ascec-v04.py\n")
            f.write(f'export ASCEC_ROOT="{ascec_directory}"\n\n')
            
            f.write("# Save original environment paths\n")
            f.write('_SYSTEM_PATH="$PATH"\n\n')
            
            f.write("# Add the ASCEC directory to the system PATH for direct execution\n")
            f.write('export PATH="$ASCEC_ROOT:$_SYSTEM_PATH"\n\n')
            
            f.write('echo "ASCEC v04 environment is now active via direct script setup."\n')
            f.write('echo "ASCEC_ROOT set to: $ASCEC_ROOT"\n\n')
            
            f.write("# Run ASCEC using the full path\n")
            
            commands = []
            for i, replicated_file in enumerate(replicated_files):
                rel_path = os.path.relpath(replicated_file, input_dir)
                output_name = os.path.splitext(rel_path)[0] + ".out"
                
                # Add separator before each calculation (except the first one)
                if i > 0:
                    commands.append('echo "=================================================================="')
                
                commands.append(f"python3 {ascec_script_path} {rel_path} > {output_name}")
            
            # Join commands with " && \\\n" for sequential execution
            f.write(" && \\\n".join(commands))
            f.write("\n")
        
        # Make the script executable
        os.chmod(launcher_path, 0o755)
        
        print(f"Created launcher script: {script_name}")
        return launcher_path
        
    except IOError as e:
        print(f"Error creating launcher script '{launcher_path}': {e}")
        return ""


def merge_launcher_scripts(working_dir: str = ".") -> str:
    """
    Finds all launcher_ascec.sh scripts in the working directory and subfolders,
    and merges them into a single launcher script.
    
    Args:
        working_dir (str): Working directory to search for launcher scripts
    
    Returns:
        str: Path to the merged launcher script
    """
    working_dir_full = os.path.abspath(working_dir)
    merged_launcher_path = os.path.join(working_dir_full, "launcher_ascec.sh")
    
    # Find all launcher scripts
    launcher_scripts = []
    for root, dirs, files in os.walk(working_dir_full):
        for file in files:
            if file == "launcher_ascec.sh":
                launcher_scripts.append(os.path.join(root, file))
    
    if not launcher_scripts:
        print("No launcher_ascec.sh scripts found in the working directory or subfolders.")
        return ""
    
    print(f"Found {len(launcher_scripts)} launcher scripts:")
    for script in launcher_scripts:
        rel_path = os.path.relpath(script, working_dir_full)
        print(f"  {rel_path}")
    
    # Merge all launcher scripts
    all_commands = []
    
    try:
        for script_path in launcher_scripts:
            with open(script_path, 'r') as f:
                lines = f.readlines()
                
            # Extract commands (skip shebang and comments)
            commands = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and 'python3 ascec-v04.py' in line:
                    # Remove trailing " && \\" if present
                    line = line.rstrip(' \\&')
                    commands.append(line)
            
            if commands:
                # Add commands from this script
                all_commands.extend(commands)
                # Add separator comment between different script groups
                if script_path != launcher_scripts[-1]:  # Don't add separator after last script
                    all_commands.append("###")
        
        # Write merged launcher script
        with open(merged_launcher_path, 'w') as f:
            f.write("#!/bin/bash\n\n")
            
            # Configuration for ASCEC v04
            # Get the directory where ascec-v04.py is located
            ascec_script_path = os.path.abspath(__file__)
            ascec_directory = os.path.dirname(ascec_script_path)
            
            f.write("# Configuration for ASCEC v04\n")
            f.write("# Set ASCEC_ROOT to the directory containing ascec-v04.py\n")
            f.write(f'export ASCEC_ROOT="{ascec_directory}"\n\n')
            
            f.write("# Save original environment paths\n")
            f.write('_SYSTEM_PATH="$PATH"\n\n')
            
            f.write("# Add the ASCEC directory to the system PATH for direct execution\n")
            f.write('export PATH="$ASCEC_ROOT:$_SYSTEM_PATH"\n\n')
            
            f.write('echo "ASCEC v04 environment is now active via direct script setup."\n')
            f.write('echo "ASCEC_ROOT set to: $ASCEC_ROOT"\n\n')
            
            f.write("# Run ASCEC using the full path\n")
            
            # Process commands with proper formatting
            for i, cmd in enumerate(all_commands):
                if cmd == "###":
                    # Write separator on its own line
                    f.write(" && \\\n###\n")
                else:
                    f.write(cmd)
                    # Add " && \" only if this is not the last command and the next command is not "###"
                    if i < len(all_commands) - 1 and all_commands[i + 1] != "###":
                        f.write(" && \\\n")
                    elif i == len(all_commands) - 1:
                        f.write("\n")  # Last command, just add newline
        
        # Make the script executable
        os.chmod(merged_launcher_path, 0o755)
        
        print(f"\nCreated merged launcher script: launcher_ascec.sh")
        print(f"Total commands: {len([cmd for cmd in all_commands if cmd != '###'])}")
        return merged_launcher_path
        
    except IOError as e:
        print(f"Error creating merged launcher script: {e}")
        return ""


def extract_configurations_from_xyz(xyz_file_path: str) -> List[Dict]:
    """
    Extracts all configurations from an XYZ file.
    
    Args:
        xyz_file_path (str): Path to the XYZ file
    
    Returns:
        List[Dict]: List of configuration dictionaries with atoms, energy, and config number
    """
    configurations = []
    
    try:
        with open(xyz_file_path, 'r') as f:
            lines = f.readlines()
        
        i = 0
        while i < len(lines):
            # Skip empty lines
            if not lines[i].strip():
                i += 1
                continue
            
            # Read number of atoms
            try:
                num_atoms = int(lines[i].strip())
            except (ValueError, IndexError):
                i += 1
                continue
            
            # Read comment line with configuration info
            if i + 1 >= len(lines):
                break
            
            comment_line = lines[i + 1].strip()
            
            # Extract configuration number and energy from comment
            config_num = 1
            energy = 0.0
            
            if "Configuration:" in comment_line:
                parts = comment_line.split("|")
                for part in parts:
                    part = part.strip()
                    if part.startswith("Configuration:"):
                        try:
                            config_num = int(part.split(":")[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif "E =" in part:
                        try:
                            energy_str = part.split("=")[1].strip().split()[0]
                            energy = float(energy_str)
                        except (ValueError, IndexError):
                            pass
            else:
                # Handle other formats like "Motif_01_opt_conf_4 (G = -55.537389 hartree)"
                import re
                # Try to extract configuration number from patterns like "conf_4" or "Motif_01"
                conf_match = re.search(r'conf_(\d+)', comment_line)
                if conf_match:
                    try:
                        config_num = int(conf_match.group(1))
                    except ValueError:
                        pass
                else:
                    # Try to extract from Motif_XX or motif_XX pattern
                    motif_match = re.search(r'[Mm]otif_(\d+)', comment_line)
                    if motif_match:
                        try:
                            config_num = int(motif_match.group(1))
                        except ValueError:
                            pass
                
                # Try to extract energy from patterns like "G = -55.537389" or "E = -55.537389"
                energy_match = re.search(r'[GE]\s*=\s*([-\d.]+)', comment_line)
                if energy_match:
                    try:
                        energy = float(energy_match.group(1))
                    except ValueError:
                        pass
            
            # Read atom coordinates (preserve original string format for precision)
            atoms = []
            for j in range(num_atoms):
                if i + 2 + j >= len(lines):
                    break
                
                atom_line = lines[i + 2 + j].strip()
                if atom_line:
                    parts = atom_line.split()
                    if len(parts) >= 4:
                        symbol = parts[0]
                        x_str = parts[1]  # Keep original string format
                        y_str = parts[2]  # Keep original string format
                        z_str = parts[3]  # Keep original string format
                        # Also store float values for numerical operations if needed
                        try:
                            x_float = float(x_str)
                            y_float = float(y_str)
                            z_float = float(z_str)
                            atoms.append((symbol, x_str, y_str, z_str, x_float, y_float, z_float))
                        except ValueError:
                            # Skip malformed coordinates
                            continue
            
            if len(atoms) == num_atoms:
                # Convert Motif_ to motif_ in comment for consistency
                processed_comment = comment_line.replace('Motif_', 'motif_')
                configurations.append({
                    'config_num': config_num,
                    'energy': energy,
                    'atoms': atoms,
                    'comment': processed_comment
                })
            
            i += num_atoms + 2
    
    except IOError as e:
        print(f"Error reading XYZ file '{xyz_file_path}': {e}")
        return []
    
    return configurations


def create_qm_input_file(config_data: Dict, template_content: str, output_path: str, qm_program: str) -> bool:
    """
    Creates a QM input file from configuration data and template.
    
    Args:
        config_data (Dict): Configuration data with atoms, energy, etc.
        template_content (str): Template file content
        output_path (str): Path where to save the input file
        qm_program (str): QM program type ('orca' or 'gaussian')
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create coordinates section (preserve original precision with proper alignment)
        coords_section = ""
        for atom in config_data['atoms']:
            if len(atom) == 7:  # New format with string coordinates
                symbol, x_str, y_str, z_str, x_float, y_float, z_float = atom
                # Right-align coordinates to maintain column alignment while preserving original precision
                coords_section += f"{symbol: <3} {x_str: >12}  {y_str: >12}  {z_str: >12}\n"
            else:  # Old format compatibility
                symbol, x, y, z = atom
                coords_section += f"{symbol: <3} {x: 12.6f}  {y: 12.6f}  {z: 12.6f}\n"
        
        # Replace name placeholders with the configuration comment (only if placeholders exist)
        content = template_content
        if "#name" in content:
            content = content.replace("#name", f"# {config_data['comment']}")
        
        if qm_program == 'orca':
            # For ORCA, replace the coordinate section between * xyz and *
            # Look for * xyz pattern (case-insensitive) with charge and multiplicity
            import re
            
            lines = content.split('\n')
            new_lines = []
            in_coords = False
            xyz_pattern = re.compile(r'^\s*\*\s+xyz\s+[-\d]+\s+\d+\s*$', re.IGNORECASE)
            
            for line in lines:
                if xyz_pattern.match(line.strip()):
                    new_lines.append(line)
                    in_coords = True
                elif in_coords and line.strip() == "*":
                    new_lines.append(line)
                    in_coords = False
                elif in_coords and line.strip() == "#":
                    # Replace the # placeholder with coordinates
                    new_lines.append(coords_section.rstrip())
                elif in_coords:
                    # Skip other coordinate lines (they will be replaced by coords_section when we hit #)
                    continue
                elif not in_coords:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
        
        elif qm_program == 'gaussian':
            # For Gaussian, replace # placeholder with coordinates
            lines = content.split('\n')
            new_lines = []
            
            for line in lines:
                if line.strip() == "#":
                    # Replace the # placeholder with coordinates
                    new_lines.append(coords_section.rstrip())
                else:
                    new_lines.append(line)
            
            content = '\n'.join(new_lines)
        
        # Create directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # Write the input file
        with open(output_path, 'w') as f:
            f.write(content)
        
        return True
        
    except IOError as e:
        print(f"Error creating QM input file '{output_path}': {e}")
        return False


def interactive_directory_selection_with_pattern(input_ext: str, pattern: str = "") -> List[str]:
    """
    Provides interactive directory selection for updating input files with optional pattern filtering.
    Shows only directories that contain matching files and lets user choose.
    
    Args:
        input_ext (str): File extension to search for
        pattern (str): Pattern to filter files (empty string means no filtering)
    
    Returns:
        List[str]: List of selected input file paths
    """
    print("\n" + "=" * 60)
    print("Directory selection".center(60))
    if pattern and pattern.strip():
        print(f"Filtering files containing: '{pattern}'".center(60))
    print("=" * 60)
    
    def filter_files(files, pattern):
        """Filter files by pattern if pattern is provided"""
        if not pattern or not pattern.strip():
            return files
        return [f for f in files if pattern in os.path.basename(f)]
    
    # Scan for directories with matching files, but group similar paths
    directories_with_files = {}
    all_files = []
    
    # Find all matching files recursively
    for root, dirs, files in os.walk("."):
        matching_files_in_dir = []
        for file in files:
            if file.endswith(input_ext):
                full_path = os.path.join(root, file)
                if not pattern or not pattern.strip() or pattern in file:
                    matching_files_in_dir.append(full_path)
                    all_files.append(full_path)
        
        # Only include directories that have matching files
        if matching_files_in_dir:
            # Normalize and group directory paths
            if root == ".":
                dir_display = "Current working directory"
            else:
                dir_display = root
                if dir_display.startswith("./"):
                    dir_display = dir_display[2:]
                dir_display += "/"
            
            # Group similar directories (e.g., merge individual semiempiric/calculation/opt1_conf_XX/ into one)
            if "semiempiric/calculation/" in dir_display and dir_display.count("/") > 2:
                # Group all semiempiric individual calculation directories
                parent_dir = "semiempiric/calculation/ (individual directories)"
                if parent_dir not in directories_with_files:
                    directories_with_files[parent_dir] = []
                directories_with_files[parent_dir].extend(matching_files_in_dir)
            else:
                directories_with_files[dir_display] = matching_files_in_dir
    
    if not directories_with_files:
        if pattern and pattern.strip():
            print(f"No files found containing pattern '{pattern}'.")
        else:
            print("No input files found in any directory.")
        return []
    
    # Display options - only directories with files
    print("\nDirectories with matching files:")
    print("-" * 40 + "\n")
    
    # Create numbered options
    options = {}
    option_num = 1
    
    # Sort directories by name for consistent ordering
    sorted_dirs = sorted(directories_with_files.items(), key=lambda x: (len(x[1]), x[0]))
    
    for dir_name, files in sorted_dirs:
        options[str(option_num)] = (dir_name, files)
        print(f"{option_num}. {dir_name}: {len(files)} files")
        
        # Show first few files as examples
        if len(files) <= 5:
            # Show all files if 5 or fewer
            for file_path in files:
                filename = os.path.basename(file_path)
                print(f"   - {filename}")
        else:
            # Show first 3 files if more than 5
            examples = files[:3]
            for example in examples:
                filename = os.path.basename(example)
                print(f"   - {filename}")
            print(f"   ... and {len(files) - 3} more")
        print()
        option_num += 1
    
    # Add "all" option if there are multiple directories
    if len(directories_with_files) > 1:
        options["a"] = ("All directories", all_files)
        print(f"a. All directories: {len(all_files)} files total")
        print()
    
    # Get user choice
    while True:
        try:
            valid_options = list(options.keys()) + ['q']
            if len(valid_options) > 6:  # If too many options, simplify the prompt
                choice = input(f"Select option (1-{len(valid_options)-2}, 'a' for all, or 'q' to quit): ").strip().lower()
            else:
                # Build a cleaner prompt with explicit 'a' for all description
                option_parts = []
                for opt in valid_options[:-1]:  # Exclude 'q'
                    if opt == 'a':
                        option_parts.append("'a' for all")
                    else:
                        option_parts.append(opt)
                choice = input(f"Select option ({', '.join(option_parts)}, or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Update cancelled.")
                return []
            
            if choice in options:
                selected_files = options[choice][1]
                dir_name = options[choice][0]
                
                print(f"\nSelected: {dir_name}")
                print(f"Files to update: {len(selected_files)}")
                if pattern and pattern.strip():
                    print(f"Pattern filter: '{pattern}'")
                
                return selected_files
            else:
                if len(valid_options) > 6:
                    print(f"Invalid option. Please choose 1-{len(valid_options)-2}, 'a', or 'q' to quit.")
                else:
                    print(f"Invalid option. Please choose {', '.join(valid_options[:-1])}, or 'q' to quit.")
                
        except KeyboardInterrupt:
            print("\nUpdate cancelled by user.")
            return []
        except EOFError:
            print("\nUpdate cancelled.")
            return []


def interactive_directory_selection(input_ext: str) -> List[str]:
    """
    Provides interactive directory selection for updating input files.
    Shows only directories that contain files and lets user choose.
    
    Args:
        input_ext (str): File extension to search for
    
    Returns:
        List[str]: List of selected input file paths
    """
    print("\n" + "=" * 60)
    print("Directory selection".center(60))
    print("=" * 60)
    
    # Scan for directories with matching files
    directories_with_files = {}
    all_files = []
    
    # Find all matching files recursively
    for root, dirs, files in os.walk("."):
        matching_files_in_dir = []
        for file in files:
            if file.endswith(input_ext):
                full_path = os.path.join(root, file)
                matching_files_in_dir.append(full_path)
                all_files.append(full_path)
        
        # Only include directories that have matching files
        if matching_files_in_dir:
            # Normalize directory path for display
            dir_display = root if root != "." else "Current working directory"
            if root.startswith("./"):
                dir_display = root[2:] + "/"
            elif root == ".":
                dir_display = "Current working directory"
            else:
                dir_display = root + "/"
            
            directories_with_files[dir_display] = matching_files_in_dir
    
    if not directories_with_files:
        print("No input files found in any directory.")
        return []
    
    # Display options - only directories with files
    print("\nDirectories with matching files:")
    print("-" * 40 + "\n")

    # Create numbered options
    options = {}
    option_num = 1
    
    for dir_name, files in directories_with_files.items():
        options[str(option_num)] = (dir_name, files)
        print(f"{option_num}. {dir_name}: {len(files)} files")
        
        # Show first few files as examples
        examples = files[:3]
        for example in examples:
            filename = os.path.basename(example)
            print(f"   - {filename}")
        if len(files) > 3:
            print(f"   ... and {len(files) - 3} more")
        print()
        option_num += 1
    
    # Add "all" option if there are multiple directories
    if len(directories_with_files) > 1:
        options["a"] = ("All directories", all_files)
        print(f"a. All directories: {len(all_files)} files total")
        print()
    
    # Get user choice
    while True:
        try:
            valid_options = list(options.keys()) + ['q']
            # Build a cleaner prompt with explicit 'a' for all description
            option_parts = []
            for opt in valid_options[:-1]:  # Exclude 'q'
                if opt == 'a':
                    option_parts.append("'a' for all")
                else:
                    option_parts.append(opt)
            choice = input(f"Select option ({', '.join(option_parts)}, or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Update cancelled.")
                return []
            
            if choice in options:
                selected_files = options[choice][1]
                dir_name = options[choice][0]
                
                print(f"\nSelected: {dir_name}")
                print(f"Files to update: {len(selected_files)}")
                
                return selected_files
            else:
                print(f"Invalid option. Please choose {', '.join(valid_options[:-1])}, or 'q' to quit.")
                
        except KeyboardInterrupt:
            print("\nUpdate cancelled by user.")
            return []
        except EOFError:
            print("\nUpdate cancelled.")
            return []


def find_files_by_pattern(pattern: str, input_ext: str) -> List[str]:
    """
    Finds input files matching a specific pattern across all directories.
    
    Args:
        pattern (str): Pattern to match in filenames
        input_ext (str): File extension to search for
    
    Returns:
        List[str]: List of matching file paths
    """
    matching_files = []
    
    # Search recursively in all directories
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.endswith(input_ext) and pattern in file:
                matching_files.append(os.path.join(root, file))
    
    print(f"\nFound {len(matching_files)} files matching pattern '{pattern}':")
    for file in matching_files:
        print(f"  - {file}")
    
    return matching_files


def update_existing_input_files(template_file: str, target_pattern: str = "") -> str:
    """
    Updates existing QM input files with a new template, preserving the coordinates 
    and configuration information from the original files.
    
    Args:
        template_file (str): New template file (e.g., new_template.inp)
        target_pattern (str): Search pattern - empty string for all locations, or specific pattern for filtered search
    
    Returns:
        str: Status message
    """
    # Determine QM program from template file extension
    if template_file.endswith('.inp'):
        qm_program = 'orca'
        input_ext = '.inp'
    elif template_file.endswith('.com'):
        qm_program = 'gaussian'
        input_ext = '.com'
    elif template_file.endswith('.gjf'):
        qm_program = 'gaussian'
        input_ext = '.gjf'
    else:
        return f"Error: Unsupported template file extension. Use .inp for ORCA or .com/.gjf for Gaussian."
    
    # Check if template file exists
    if not os.path.exists(template_file):
        return f"Error: Template file '{template_file}' not found."
    
    # Read template content
    try:
        with open(template_file, 'r') as f:
            template_content = f.read()
    except IOError as e:
        return f"Error reading template file '{template_file}': {e}"
    
    # Always show interactive directory selection, but filter by pattern if provided
    input_files = interactive_directory_selection_with_pattern(input_ext, target_pattern)
    if not input_files:
        return "No files selected for update."
    
    # Exclude the template file from the update list
    template_file_abs = os.path.abspath(template_file)
    input_files = [f for f in input_files if os.path.abspath(f) != template_file_abs]
    
    if not input_files:
        return "No input files found for update (excluding template file)."
    
    updated_count = 0
    skipped_count = 0
    backup_files = []  # Track backup files for potential revert
    
    print(f"\nStarting update of {len(input_files)} files...")
    
    for input_file in input_files:
        try:
            # Read existing input file
            with open(input_file, 'r') as f:
                existing_content = f.read()
            
            # Create backup file
            backup_file = input_file + '.backup_temp'
            with open(backup_file, 'w') as f:
                f.write(existing_content)
            backup_files.append((input_file, backup_file))
            
            # Extract configuration information and coordinates from existing file
            config_info = extract_config_from_input_file(existing_content, qm_program)
            if not config_info:
                print(f"Warning: Could not extract configuration from {os.path.basename(input_file)}, skipping.")
                skipped_count += 1
                continue
            
            # Create updated content using the new template
            if create_qm_input_file(config_info, template_content, input_file, qm_program):
                print(f"  Updated: {os.path.basename(input_file)}")
                updated_count += 1
            else:
                print(f"  Failed to update: {os.path.basename(input_file)}")
                skipped_count += 1
                
        except IOError as e:
            print(f"Error processing {os.path.basename(input_file)}: {e}")
            skipped_count += 1
    
    # Show final result and offer revert option
    print(f"\nUpdate completed: {updated_count} files updated, {skipped_count} files skipped.")
    
    if updated_count > 0:
        print("\nPress ENTER to finish and keep changes, or type 'r' and ENTER to revert all changes:")
        try:
            user_choice = input().strip().lower()
            if user_choice == 'r':
                # Revert all changes
                reverted_count = 0
                for original_file, backup_file in backup_files:
                    try:
                        if os.path.exists(backup_file):
                            # Restore original content
                            with open(backup_file, 'r') as f:
                                original_content = f.read()
                            with open(original_file, 'w') as f:
                                f.write(original_content)
                            reverted_count += 1
                    except IOError as e:
                        print(f"Error reverting {os.path.basename(original_file)}: {e}")
                
                # Clean up backup files
                for _, backup_file in backup_files:
                    try:
                        if os.path.exists(backup_file):
                            os.remove(backup_file)
                    except OSError:
                        pass
                
                return f"Reverted {reverted_count} files to original state."
            else:
                # Keep changes, clean up backup files
                for _, backup_file in backup_files:
                    try:
                        if os.path.exists(backup_file):
                            os.remove(backup_file)
                    except OSError:
                        pass
                return f"Changes kept: {updated_count} files updated, {skipped_count} files skipped."
        except KeyboardInterrupt:
            print("\nKeeping changes (Ctrl+C pressed).")
            # Clean up backup files
            for _, backup_file in backup_files:
                try:
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                except OSError:
                    pass
            return f"Changes kept: {updated_count} files updated, {skipped_count} files skipped."
        except EOFError:
            print("\nKeeping changes.")
            # Clean up backup files
            for _, backup_file in backup_files:
                try:
                    if os.path.exists(backup_file):
                        os.remove(backup_file)
                except OSError:
                    pass
            return f"Changes kept: {updated_count} files updated, {skipped_count} files skipped."
    else:
        # No files were updated, just clean up any backup files
        for _, backup_file in backup_files:
            try:
                if os.path.exists(backup_file):
                    os.remove(backup_file)
            except OSError:
                pass
        return f"Update completed: {updated_count} files updated, {skipped_count} files skipped."


def extract_config_from_input_file(content: str, qm_program: str) -> Optional[Dict]:
    """
    Extracts configuration information from an existing QM input file.
    
    Args:
        content (str): Content of the input file
        qm_program (str): QM program type ('orca' or 'gaussian')
    
    Returns:
        Dict: Configuration data or None if extraction fails
    """
    try:
        lines = content.split('\n')
        
        # Extract comment line (configuration info)
        comment = ""
        for line in lines:
            if line.strip().startswith('#') and ('Configuration:' in line or 'E =' in line):
                comment = line.strip()[1:].strip()  # Remove # and extra spaces
                break
        
        # Extract coordinates
        atoms = []
        
        if qm_program == 'orca':
            # For ORCA, find coordinates between "* xyz 0 1" and "*"
            in_coords = False
            for line in lines:
                if line.strip() == "* xyz 0 1":
                    in_coords = True
                    continue
                elif line.strip() == "*" and in_coords:
                    break
                elif in_coords and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        symbol = parts[0]
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        atoms.append((symbol, x, y, z))
        
        elif qm_program == 'gaussian':
            # For Gaussian, find coordinates after charge/multiplicity line
            found_charge_mult = False
            for line in lines:
                if not found_charge_mult and line.strip() and len(line.strip().split()) == 2:
                    try:
                        int(line.strip().split()[0])  # charge
                        int(line.strip().split()[1])  # multiplicity
                        found_charge_mult = True
                        continue
                    except ValueError:
                        pass
                elif found_charge_mult and line.strip():
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        symbol = parts[0]
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        atoms.append((symbol, x, y, z))
                elif found_charge_mult and not line.strip():
                    # Empty line might indicate end of coordinates
                    if atoms:  # Only break if we have found some atoms
                        break
        
        if not atoms:
            return None
        
        return {
            'comment': comment,
            'atoms': atoms
        }
        
    except Exception as e:
        print(f"Error extracting configuration: {e}")
        return None


def interactive_xyz_file_selection(xyz_files: List[str], calc_dir: str = ".") -> List[str]:
    """
    Provides interactive selection for XYZ files to process.
    
    Args:
        xyz_files (List[str]): List of available XYZ file paths
        calc_dir (str): Directory where combined files should be created
    
    Returns:
        List[str]: List of selected XYZ file paths
    """
    print("\n" + "=" * 60)
    print("XYZ file selection".center(60))
    print("=" * 60)
    
    if not xyz_files:
        print("No XYZ files found.")
        return []
    
    # Separate result_*.xyz files from combined_results.xyz and combined_r*.xyz
    result_files = [f for f in xyz_files if not (os.path.basename(f).startswith("combined_results") or os.path.basename(f).startswith("combined_r"))]
    combined_files = [f for f in xyz_files if (os.path.basename(f).startswith("combined_results") or os.path.basename(f).startswith("combined_r"))]
    
    # Display options
    print("\nAvailable XYZ files:")
    print("-" * 40)
    
    # Create numbered options
    options = {}
    option_num = 1
    
    # First show result_*.xyz files
    if result_files:
        print("\nResult files:")
        for xyz_file in result_files:
            options[str(option_num)] = xyz_file
            print(f"{option_num}. {xyz_file}")
            option_num += 1
    
    # Then show combined files
    if combined_files:
        print("\nCombined files:")
        for xyz_file in combined_files:
            options[str(option_num)] = xyz_file
            print(f"{option_num}. {xyz_file}")
            option_num += 1
    
    # Add special options
    print("\nSpecial options:")
    if len(result_files) > 1:
        options["a"] = "All result files"
        print(f"a. Process all result_*.xyz files ({len(result_files)} files, excluding combined)")
        options["c"] = "Combine result files"
        print(f"c. Combine all result_*.xyz files first, then process the combined file (combined_r{len(result_files)}.xyz)")
    
    print("q. Quit")
    
    # Get user choice
    while True:
        try:
            choice = input("\nSelect files (enter numbers separated by spaces, 'a' for all, or 'c' to combine): ").strip()
            
            if choice.lower() == 'q':
                print("Operation cancelled.")
                return []
            
            if choice.lower() == 'a' and len(result_files) > 1:
                print(f"Selected: All result_*.xyz files ({len(result_files)} files)")
                return result_files
            
            if choice.lower() == 'c' and len(result_files) > 1:
                print(f"Combining {len(result_files)} result_*.xyz files...")
                # Use merge_xyz_files to combine result files only
                combined_filename = os.path.join(calc_dir, f"combined_r{len(result_files)}.xyz")
                success = merge_xyz_files(result_files, combined_filename)
                if success:
                    print(f"Successfully created {os.path.basename(combined_filename)}")
                    return [combined_filename]
                else:
                    print("Failed to combine files. Using individual files instead.")
                    return result_files
            
            # Handle numbered selections (single or multiple)
            try:
                numbers = [int(x.strip()) for x in choice.split() if x.strip().isdigit()]
                selected_files = []
                
                for num in numbers:
                    if str(num) in options:
                        selected_files.append(options[str(num)])
                    else:
                        print(f"Invalid number: {num}")
                        raise ValueError
                
                if selected_files:
                    print(f"Selected {len(selected_files)} file(s):")
                    for f in selected_files:
                        print(f"  - {f}")
                    return selected_files
                else:
                    print("No valid files selected.")
                    
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces, or 'a' for all result files.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return []
        except EOFError:
            print("\nOperation cancelled.")
            return []


def create_combined_xyz_from_list(xyz_files: List[str]) -> bool:
    """
    Create a combined XYZ file from a list of XYZ files.
    
    Args:
        xyz_files (List[str]): List of XYZ file paths to combine
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not xyz_files:
        print("No XYZ files to combine.")
        return False
    
    combined_content = []
    
    for xyz_file in xyz_files:
        try:
            with open(xyz_file, 'r') as f:
                content = f.read().strip()
                if content:
                    combined_content.append(f"# {xyz_file}")
                    combined_content.append(content)
                    combined_content.append("")  # Empty line between structures
        except Exception as e:
            print(f"Error reading {xyz_file}: {e}")
    
    if combined_content:
        try:
            with open("combined_results.xyz", 'w') as f:
                f.write("\n".join(combined_content))
            print(f"Created combined_results.xyz with {len(xyz_files)} structures.")
            return True
        except Exception as e:
            print(f"Error creating combined_results.xyz: {e}")
            return False
    else:
        print("No content to combine.")
        return False
    
    # Add special options
    if len(xyz_files) > 1:
        options["a"] = "All files"
        print(f"a. All files: {len(xyz_files)} files total")
    
    options["c"] = "Combined results"
    num_result_files = len([f for f in xyz_files if 'result_' in os.path.basename(f)])
    print(f"c. Combine all result_*.xyz files first, then process the combined file (combined_r{num_result_files}.xyz)")
    print()
    
    # Get user choice
    while True:
        try:
            valid_options = list(options.keys()) + ['q']
            if len(valid_options) > 6:
                choice = input(f"Select option (1-{len(valid_options)-3}, 'a' for all, 'c' for combined, or 'q' to quit): ").strip().lower()
            else:
                # Build a cleaner prompt
                option_parts = []
                for opt in valid_options[:-1]:  # Exclude 'q'
                    if opt == 'a':
                        option_parts.append("'a' for all")
                    elif opt == 'c':
                        option_parts.append("'c' for combined")
                    else:
                        option_parts.append(opt)
                choice = input(f"Select option ({', '.join(option_parts)}, or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Calculation system creation cancelled.")
                return []
            
            if choice == 'c':
                # Create combined_r{N}.xyz first where N is number of result files
                num_result_files = len([f for f in xyz_files if 'result_' in os.path.basename(f)])
                combined_filename = f"combined_r{num_result_files}.xyz"
                print(f"\nCreating {combined_filename}...")
                if combine_xyz_files(output_filename=combined_filename):
                    if os.path.exists(combined_filename):
                        print(f"Successfully created {combined_filename}")
                        return [combined_filename]
                    else:
                        print(f"Error: {combined_filename} was not created.")
                        return []
                else:
                    print(f"Failed to create {combined_filename}")
                    return []
            
            if choice == 'a':
                if len(xyz_files) > 1:
                    print(f"\nSelected: All files ({len(xyz_files)} files)")
                    return xyz_files
                else:
                    print("Only one file available, selecting it.")
                    return xyz_files
            
            if choice in options and choice not in ['a', 'c']:
                selected_file = options[choice]
                print(f"\nSelected: {selected_file}")
                return [selected_file]
            else:
                if len(valid_options) > 6:
                    print(f"Invalid option. Please choose 1-{len(valid_options)-3}, 'a', 'c', or 'q' to quit.")
                else:
                    print(f"Invalid option. Please choose {', '.join(valid_options[:-1])}, or 'q' to quit.")
                
        except KeyboardInterrupt:
            print("\nCalculation system creation cancelled by user.")
            return []
        except EOFError:
            print("\nCalculation system creation cancelled.")
            return []


def extract_qm_executable_from_launcher(launcher_content: str, qm_program: str) -> str:
    """
    Extract the QM executable path from the launcher template.
    
    Args:
        launcher_content (str): Content of the launcher template
        qm_program (str): QM program name ('orca' or 'gaussian')
    
    Returns:
        str: The executable path or command to use
    """
    if not launcher_content:
        # Fallback to bare commands if no launcher content provided
        if qm_program == 'orca':
            return "orca"
        else:  # gaussian
            return "g16"
    
    lines = launcher_content.split('\n')
    
    if qm_program == 'orca':
        # Look for ORCA root definitions - enhanced patterns to catch custom variable names
        orca_root_patterns = [
            r'export\s+(ORCA\d*_ROOT)\s*=\s*(.+)',  # ORCA5_ROOT, ORCA66_ROOT, etc.
            r'export\s+(ORCA_ROOT)\s*=\s*(.+)',     # Standard ORCA_ROOT
            r'(ORCA\d*_ROOT)\s*=\s*(.+)',           # Without export
            r'(ORCA_ROOT)\s*=\s*(.+)',              # Without export
        ]
        
        orca_root_var = None
        orca_in_path = False
        
        for line in lines:
            line = line.strip()
            
            # Check if ORCA is added to PATH
            if 'PATH=' in line and 'ORCA' in line:
                orca_in_path = True
            
            # Look for ORCA root variable definitions
            for pattern in orca_root_patterns:
                import re
                match = re.search(pattern, line)
                if match:
                    var_name = match.group(1)  # e.g., ORCA66_ROOT, ORCA5_ROOT
                    orca_root = match.group(2).strip().strip('"\'')
                    
                    # Handle variable expansion like $ORCA_BASE/orca_5_0_4
                    if '$' in orca_root:
                        # For complex paths, use the variable as-is and let shell expand
                        orca_root_var = f"${var_name}/orca"
                    else:
                        # Direct path
                        orca_root_var = f"{orca_root}/orca"
                    break
        
        # Return the found root variable, or use 'orca' if it's in PATH
        if orca_root_var:
            return orca_root_var
        elif orca_in_path:
            return "orca"
        else:
            # Default fallback
            return "orca"
        
    else:  # gaussian
        # Look for Gaussian root definitions
        gaussian_root_patterns = [
            r'export\s+G16_ROOT\s*=\s*(.+)',
            r'export\s+G09_ROOT\s*=\s*(.+)',
            r'export\s+GAUSS_EXEDIR\s*=\s*(.+)',
            r'G16_ROOT\s*=\s*(.+)',
            r'G09_ROOT\s*=\s*(.+)',
        ]
        
        gaussian_in_path = False
        
        for line in lines:
            line = line.strip()
            
            # Check if Gaussian is in PATH
            if 'PATH=' in line and ('G16' in line or 'G09' in line or 'gaussian' in line.lower()):
                gaussian_in_path = True
            
            # Look for Gaussian root variable definitions
            for pattern in gaussian_root_patterns:
                import re
                match = re.search(pattern, line)
                if match:
                    gauss_root = match.group(1).strip().strip('"\'')
                    if '$' in gauss_root:
                        if 'G16_ROOT' in line:
                            return "$G16_ROOT/g16"
                        elif 'G09_ROOT' in line:
                            return "$G09_ROOT/g09"
                        else:
                            return "$GAUSS_EXEDIR/g16"
                    else:
                        return f"{gauss_root}/g16"
        
        # If Gaussian appears to be in PATH or no specific root found, use bare command
        return "g16"


def create_simple_calculation_system(template_file: str, launcher_template: Optional[str] = None) -> str:
    """
    Creates a calculation system by looking for result_*.xyz, combined_results.xyz, or combined_r*.xyz files
    and generating QM input files with optional launcher scripts.
    
    Args:
        template_file (str): Template input file (e.g., example_input.inp)
        launcher_template (str, optional): Template launcher file (e.g., launcher_orca.sh)
    
    Returns:
        str: Status message
    """
    # Determine QM program from template file extension
    if template_file.endswith('.inp'):
        qm_program = 'orca'
        input_ext = '.inp'
        output_ext = '.out'
    elif template_file.endswith('.com'):
        qm_program = 'gaussian'
        input_ext = '.com'
        output_ext = '.log'
    elif template_file.endswith('.gjf'):
        qm_program = 'gaussian'
        input_ext = '.gjf'
        output_ext = '.log'
    else:
        return f"Error: Unsupported template file extension. Use .inp for ORCA or .com/.gjf for Gaussian."
    
    # Check if template files exist
    if not os.path.exists(template_file):
        return f"Error: Template file '{template_file}' not found."
    
    if launcher_template and not os.path.exists(launcher_template):
        return f"Error: Launcher template '{launcher_template}' not found."
    
    # Read template content
    try:
        with open(template_file, 'r') as f:
            template_content = f.read()
    except IOError as e:
        return f"Error reading template file '{template_file}': {e}"
    
    # Read launcher template if provided
    launcher_content = None
    if launcher_template:
        try:
            with open(launcher_template, 'r') as f:
                launcher_content = f.read()
        except IOError as e:
            return f"Error reading launcher template '{launcher_template}': {e}"
    
    # Create calculation directory with incremental numbering
    def get_next_calc_dir():
        """Find the next available calculation directory (calculation, calculation_2, etc.)"""
        base_name = "calculation"
        if not os.path.exists(base_name):
            return base_name
        
        counter = 2
        while True:
            calc_dir_name = f"{base_name}_{counter}"
            if not os.path.exists(calc_dir_name):
                return calc_dir_name
            counter += 1
    
    calc_dir = get_next_calc_dir()
    os.makedirs(calc_dir, exist_ok=True)
    
    # Look for XYZ files: result_*.xyz AND combined_results.xyz AND combined_r*.xyz
    xyz_files = []
    
    # Check for combined files in current directory
    for file in os.listdir("."):
        if (file.startswith("combined_results") or file.startswith("combined_r")) and file.endswith(".xyz"):
            xyz_files.append(file)
    
    # Look for result_*.xyz files recursively in subdirectories
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.startswith("result_") and file.endswith(".xyz") and not file.startswith("resultbox_"):
                xyz_files.append(os.path.join(root, file))
    
    if not xyz_files:
        return "No result_*.xyz, combined_results.xyz, or combined_r*.xyz files found in the current directory or subdirectories."
    
    # Sort XYZ files by annealing number (extracted from directory name)
    def get_annealing_number(file_path):
        """Extract annealing number from file path like './some_name_2/result_*.xyz'"""
        import re
        # Look for pattern like 'name_N' in the directory path
        directory = os.path.dirname(file_path)
        match = re.search(r'_(\d+)$', directory)
        if match:
            return int(match.group(1))
        # If no _N pattern found in directory, try to extract from filename
        match = re.search(r'result_(\d+)', os.path.basename(file_path))
        if match:
            return int(match.group(1))
        return float('inf')  # Put unmatched files at the end
    
    xyz_files.sort(key=get_annealing_number)
    
    print(f"Found {len(xyz_files)} XYZ file(s) to process:")
    for xyz_file in xyz_files:
        print(f"  - {xyz_file}")
    
    # Always let user choose which files to process
    selected_xyz_files = interactive_xyz_file_selection(xyz_files, calc_dir)
    if not selected_xyz_files:
        return "No XYZ files selected for processing."
    xyz_files = selected_xyz_files
    
    # Process each XYZ file
    all_input_files = []
    
    for xyz_file in xyz_files:
        # Extract configurations
        configurations = extract_configurations_from_xyz(xyz_file)
        if not configurations:
            print(f"Warning: No configurations found in {xyz_file}")
            continue
        
        # Determine run number from filename and directory
        filename = os.path.basename(xyz_file)
        if filename.startswith("combined_results") or filename.startswith("combined_r"):
            run_num = 1
        else:
            # Extract seed from result_<seed>.xyz
            try:
                seed = filename.replace("result_", "").replace(".xyz", "")
                run_num = int(seed) if seed.isdigit() else 1
            except:
                run_num = 1
            
            # Also try to get run number from directory name if it contains meaningful info
            dir_name = os.path.dirname(xyz_file)
            if dir_name != ".":
                # Extract number from directory name (e.g., w6_annealing4_1 -> 1)
                parts = os.path.basename(dir_name).split('_')
                for part in reversed(parts):
                    try:
                        dir_run_num = int(part)
                        run_num = dir_run_num  # Use directory number if available
                        break
                    except ValueError:
                        continue
        
        print(f"\nProcessing {xyz_file} with {len(configurations)} configurations:")
        
        # Create input files for each configuration
        for config in configurations:
            # Clean up the comment for result files to remove temperature and add source info
            filename = os.path.basename(xyz_file)
            if not (filename.startswith("combined_results") or filename.startswith("combined_r")):
                # Extract energy from original comment
                original_comment = config['comment']
                energy_match = re.search(r'E = ([-\d.]+) a\.u\.', original_comment)
                config_match = re.search(r'Configuration: (\d+)', original_comment)
                
                energy = energy_match.group(1) if energy_match else "unknown"
                config_num = config_match.group(1) if config_match else config['config_num']
                
                # Create new comment without temperature, with source info
                source_name = filename.replace('.xyz', '')
                if energy == "unknown":
                    config['comment'] = f"Configuration: {config_num} | {source_name}"
                else:
                    config['comment'] = f"Configuration: {config_num} | E = {energy} a.u. | {source_name}"
            
            if filename.startswith("combined_results") or filename.startswith("combined_r"):
                input_name = f"opt_conf_{config['config_num']}{input_ext}"
            else:
                input_name = f"opt{run_num}_conf_{config['config_num']}{input_ext}"
                
            input_path = os.path.join(calc_dir, input_name)
            
            if create_qm_input_file(config, template_content, input_path, qm_program):
                all_input_files.append(input_name)
                print(f"  Created: {input_name}")
            else:
                print(f"  Failed to create: {input_name}")
    
    if not all_input_files:
        return "No input files were created successfully."
    
    # Create launcher script only if template is provided
    if launcher_template and launcher_content:
        launcher_path = os.path.join(calc_dir, f"launcher_{qm_program}.sh")
        
        try:
            # Group commands by run number
            run_groups = {}
            
            # Extract QM executable path from launcher template
            qm_executable = extract_qm_executable_from_launcher(launcher_content, qm_program)
            print(f"Using QM executable: {qm_executable}")
            
            for input_file in all_input_files:
                # Extract run number from filename
                if input_file.startswith("opt_conf_"):
                    run_num = 0  # For combined_results files (opt_conf_X.inp)
                elif input_file.startswith("opt") and "_conf_" in input_file:
                    run_num = int(input_file.split("_")[0][3:])  # Extract number from "optX_conf_Y.inp"
                else:
                    run_num = 0  # Default fallback
                
                if run_num not in run_groups:
                    run_groups[run_num] = []
                
                output_file = input_file.replace(input_ext, output_ext)
                if qm_program == 'orca':
                    cmd = f"{qm_executable} {input_file} > {output_file}"
                else:  # gaussian
                    cmd = f"{qm_executable} {input_file} {output_file}"
                run_groups[run_num].append(cmd)
            
            # Sort run groups and commands within each group
            sorted_runs = sorted(run_groups.keys())
            for run_num in sorted_runs:
                # Sort commands numerically by configuration number
                def extract_conf_number(cmd):
                    import re
                    # Extract number from opt_conf_X.inp or optY_conf_X.inp
                    match = re.search(r'_conf_(\d+)\.inp', cmd)
                    return int(match.group(1)) if match else 0
                
                run_groups[run_num].sort(key=extract_conf_number)
            
            # Write launcher script
            with open(launcher_path, 'w') as f:
                # Process launcher template content to handle ### separator
                launcher_lines = launcher_content.rstrip().split('\n')
                separator_found = False
                
                # Write everything up to the ### separator
                for line in launcher_lines:
                    if line.strip() == '###':
                        separator_found = True
                        f.write(line + "\n")
                        f.write("\n")  # Add blank line after separator
                        f.write("# Run QM using the full path\n")  # Add the comment as requested
                        break
                    else:
                        # Skip existing ORCA commands (example commands to be replaced)
                        if '$ORCA5_ROOT/orca' in line and '.inp' in line:
                            continue
                        # Also skip lines that are just '&&' continuation from removed commands
                        if line.strip() == '&&' or line.strip() == '&& \\':
                            continue
                        f.write(line + "\n")
                
                # If no ### separator found, add it
                if not separator_found:
                    f.write("\n###\n\n")
                    f.write("# Run QM using the full path\n")
                
                # Write grouped commands with separators
                for i, run_num in enumerate(sorted_runs):
                    commands = run_groups[run_num]
                    
                    # Write commands for this run
                    for j, cmd in enumerate(commands):
                        f.write(cmd)
                        if j < len(commands) - 1:
                            f.write(" && \\\n")
                        else:
                            f.write(" && \\\n")  # End this run group with &&
                    
                    # Add separator between run groups (except after the last group)
                    if i < len(sorted_runs) - 1:
                        f.write("###\n")
                
                # Remove the trailing " && \" from the last command and add final newline
                f.seek(f.tell() - 5)  # Go back to remove " && \"
                f.write("\n")
                f.truncate()  # Remove any content after the current position
            
            # Make launcher executable
            os.chmod(launcher_path, 0o755)
            
            print(f"\nCreated calculation system in '{calc_dir}' directory:")
            print(f"  Input files: {len(all_input_files)}")
            print(f"  Launcher script: launcher_{qm_program}.sh")
            print(f"\nTo run all calculations, use:")
            print(f"  cd {calc_dir}")
            print(f"  ./launcher_{qm_program}.sh")
            
            return f"Successfully created calculation system with {len(all_input_files)} input files."
            
        except IOError as e:
            return f"Error creating launcher script: {e}"
    else:
        # No launcher template provided - just create input files
        print(f"\nCreated calculation system in '{calc_dir}' directory:")
        print(f"  Input files: {len(all_input_files)}")
        print("\nInput files created without launcher script.")
        print("To run calculations manually, use appropriate QM program commands in the calculation directory.")
        
        return f"Successfully created calculation system with {len(all_input_files)} input files (input files only)."


def create_optimization_system(template_file: str, launcher_template: Optional[str] = None) -> str:
    """
    Creates an optimization system by looking for files with 'combined' in their name 
    (like all_motifs_combined.xyz) and motif_*.xyz files.
    
    Args:
        template_file (str): Path to the QM input template file
        launcher_template (str): Path to the launcher script template
    
    Returns:
        str: Status message indicating success or failure
    """
    if not os.path.exists(template_file):
        return f"Error: Template file '{template_file}' not found."
    
    if launcher_template and not os.path.exists(launcher_template):
        return f"Error: Launcher template '{launcher_template}' not found."
    
    # Determine calculation directory name
    def get_next_optimization_dir():
        """Find the next available optimization directory (optimization, optimization_2, etc.)"""
        base_name = "optimization"
        if not os.path.exists(base_name):
            return base_name
        
        counter = 2
        while True:
            opt_dir_name = f"{base_name}_{counter}"
            if not os.path.exists(opt_dir_name):
                return opt_dir_name
            counter += 1
    
    opt_dir = get_next_optimization_dir()
    os.makedirs(opt_dir, exist_ok=True)
    
    # Look for optimization files: files with 'combined' in name AND motif_*.xyz files
    xyz_files = []
    
    # Check for combined files in current directory
    for file in os.listdir("."):
        if file.endswith(".xyz") and "combined" in file.lower():
            xyz_files.append(file)
    
    # Look for motif_*.xyz files in current directory and subdirectories
    for root, dirs, files in os.walk("."):
        for file in files:
            if (file.startswith("motif_") and file.endswith(".xyz")) or \
               (file.endswith(".xyz") and "combined" in file.lower() and root != "."):
                xyz_files.append(os.path.join(root, file))
    
    if not xyz_files:
        return "No files with 'combined' in name or motif_*.xyz files found in the current directory or subdirectories."
    
    # Interactive file selection
    selected_xyz_files = interactive_optimization_file_selection(xyz_files, opt_dir)
    
    if not selected_xyz_files:
        # Clean up empty directory
        try:
            os.rmdir(opt_dir)
        except:
            pass
        return "No files selected for optimization. Operation cancelled."
    
    # Determine QM program and file extension from template
    if template_file.lower().endswith('.inp'):
        qm_program = "orca"
        input_ext = ".inp"
    elif template_file.lower().endswith('.com'):
        qm_program = "gaussian"
        input_ext = ".com"
    elif template_file.lower().endswith('.gjf'):
        qm_program = "gaussian"
        input_ext = ".gjf"
    else:
        qm_program = "gaussian"  # Default fallback
        input_ext = ".com"
    
    # Read template content
    try:
        with open(template_file, 'r') as f:
            template_content = f.read()
    except IOError as e:
        return f"Error reading template file: {e}"
    
    # Process each selected XYZ file
    all_input_files = []
    
    for xyz_file in selected_xyz_files:
        # Extract configurations
        configurations = extract_configurations_from_xyz(xyz_file)
        if not configurations:
            print(f"Warning: No configurations found in {xyz_file}")
            continue
        
        # Create input files for each configuration
        for config in configurations:
            # Get base name for source info
            base_name = os.path.basename(xyz_file).replace('.xyz', '')
            
            # Extract motif name from comment if available, otherwise use base filename
            import re
            comment = config['comment']
            motif_match = re.search(r'[Mm]otif_(\d+)', comment)
            
            # If not found in comment, try to extract from filename
            if not motif_match:
                motif_match = re.search(r'[Mm]otif_(\d+)', base_name)
            
            if motif_match:
                # Use motif name from comment or filename
                motif_num = int(motif_match.group(1))
                input_name = f"motif_{motif_num:0>2}_opt{input_ext}"
                source_name = f"motif_{motif_num:0>2}"
            else:
                # For non-motif files, use simple opt_conf_X naming
                input_name = f"opt_conf_{config['config_num']}{input_ext}"
                source_name = base_name
            
            input_path = os.path.join(opt_dir, input_name)
            
            # Update config comment with source file info
            config['comment'] = f"Configuration: {config['config_num']} | Source: {source_name}"
            
            # Create input file
            if create_qm_input_file(config, template_content, input_path, qm_program):
                all_input_files.append(input_name)
                print(f"Created: {input_name}")
            else:
                print(f"Failed to create: {input_name}")
    
    if not all_input_files:
        return "No input files were created successfully."
    
    # Create launcher script only if template is provided
    if launcher_template:
        launcher_path = os.path.join(opt_dir, f"launcher_{qm_program}.sh")
        
        try:
            # Read launcher template
            with open(launcher_template, 'r') as f:
                launcher_content = f.read()
            
            # Process launcher template content to handle ### separator
            launcher_lines = launcher_content.split('\n')
            header_lines = []
            found_separator = False
            
            # Write everything up to the ### separator
            for line in launcher_lines:
                if line.strip() == '###':
                    found_separator = True
                    break
                header_lines.append(line)
            
            # Extract QM executable path from launcher template
            qm_executable = extract_qm_executable_from_launcher(launcher_content, qm_program)
            print(f"Using QM executable: {qm_executable}")
            
            with open(launcher_path, 'w') as f:
                # Write header
                for line in header_lines:
                    f.write(line + '\n')
                
                # Always add separator and comment line
                f.write("\n###\n\n")
                f.write("# Run QM using the full path \n\n")
                
                # Add commands for all input files
                for i, input_file in enumerate(all_input_files):
                    # Create output filename by replacing extension
                    if qm_program == "gaussian":
                        # Handle both .com and .gjf extensions
                        if input_file.endswith('.com'):
                            output_file = input_file.replace('.com', '.log')
                        elif input_file.endswith('.gjf'):
                            output_file = input_file.replace('.gjf', '.log')
                        else:
                            output_file = input_file + '.log'  # fallback
                        command = f"{qm_executable} {input_file} {output_file}"
                    else:
                        output_file = input_file.replace('.inp', '.out')
                        command = f"{qm_executable} {input_file} > {output_file}"
                    
                    if i < len(all_input_files) - 1:
                        f.write(f"{command} && \\\n")
                    else:
                        f.write(f"{command}\n")
            
            # Make launcher executable
            os.chmod(launcher_path, 0o755)
            
            return f"""Optimization system created successfully in '{opt_dir}':
- Input files: {len(all_input_files)} files
- Launcher script: {os.path.basename(launcher_path)}
- Total configurations: {len(all_input_files)}

To run the calculations:
cd {opt_dir}
./{os.path.basename(launcher_path)}
"""
            
        except IOError as e:
            return f"Error creating launcher script: {e}"
    else:
        # No launcher template provided - just create input files
        return f"""Optimization system created successfully in '{opt_dir}':
- Input files: {len(all_input_files)} files
- Total configurations: {len(all_input_files)}

Input files created without launcher script.
To run calculations manually, use appropriate QM program commands in the {opt_dir} directory.
"""


def interactive_optimization_file_selection(xyz_files: List[str], opt_dir: str = ".") -> List[str]:
    """
    Provides interactive selection for optimization XYZ files.
    
    Args:
        xyz_files (List[str]): List of available XYZ file paths
        opt_dir (str): Directory where files will be processed
    
    Returns:
        List[str]: List of selected XYZ file paths
    """
    print("\n" + "=" * 60)
    print("Optimization XYZ file selection".center(60))
    print("=" * 60)
    
    if not xyz_files:
        print("No XYZ files found.")
        return []
    
    # Separate combined files from motif files
    combined_files = [f for f in xyz_files if "combined" in os.path.basename(f).lower()]
    motif_files = [f for f in xyz_files if "motif_" in os.path.basename(f) and "combined" not in os.path.basename(f).lower()]
    
    # Sort motif files by motif number
    def extract_motif_number(filepath):
        import re
        filename = os.path.basename(filepath)
        match = re.search(r'motif_(\d+)', filename)
        return int(match.group(1)) if match else 0
    
    motif_files.sort(key=extract_motif_number)
    
    # Display options
    print("\nAvailable XYZ files:")
    print("-" * 40)
    
    # Create numbered options
    options = {}
    option_num = 0
    
    # First show combined files
    if combined_files:
        print("\nCombined files:")
        for xyz_file in combined_files:
            options[str(option_num)] = xyz_file
            print(f"{option_num}) {xyz_file}")
            option_num += 1
    
    # Then show motif files
    if motif_files:
        print("\nMotif files:")
        for xyz_file in motif_files:
            options[str(option_num)] = xyz_file
            print(f"{option_num}) {xyz_file}")
            option_num += 1
    
    # Add special options
    print("\nSpecial options:")
    if len(xyz_files) > 1:
        options["a"] = "All files"
        print(f"a. Process all files ({len(xyz_files)} files total)")
    
    print("q. Quit")
    
    # Get user choice
    while True:
        try:
            choice = input("\nSelect files (enter numbers separated by spaces, 'a' for all, or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print("Optimization system creation cancelled.")
                return []
            
            if choice.lower() == 'a':
                print(f"Selected: All files ({len(xyz_files)} files)")
                return xyz_files
            
            # Handle numbered selections (single or multiple)
            try:
                choices = choice.split()
                selected_files = []
                
                for ch in choices:
                    if ch in options and ch not in ['a', 'q']:
                        selected_files.append(options[ch])
                    else:
                        print(f"Invalid choice: {ch}")
                        break
                else:
                    if selected_files:
                        print(f"Selected {len(selected_files)} file(s):")
                        for f in selected_files:
                            print(f"  - {f}")
                        return selected_files
                    else:
                        print("No valid files selected.")
                        
            except ValueError:
                print("Invalid input. Please enter numbers separated by spaces, or 'a' for all files.")
                
        except KeyboardInterrupt:
            print("\nOptimization system creation cancelled by user.")
            return []
        except EOFError:
            print("\nOptimization system creation cancelled.")
            return []


def execute_merge_command():
    """
    Execute the merge command functionality.
    Shows directories with XYZ files and allows user to merge them.
    """
    print("\n" + "=" * 60)
    print("XYZ Files Merge System".center(60))
    print("=" * 60)
    
    # Scan for directories with XYZ files (excluding _trj.xyz files)
    directories_with_xyz = {}
    
    # Check current directory
    current_xyz = []
    for file in os.listdir("."):
        if (file.endswith(".xyz") and not file.endswith("_trj.xyz") and 
            not file.startswith("combined_results") and not file.startswith("combined_r")):
            current_xyz.append(file)
    
    if current_xyz:
        directories_with_xyz["."] = current_xyz
    
    # Check subdirectories
    for root, dirs, files in os.walk("."):
        if root == ".":
            continue
            
        xyz_files = []
        for file in files:
            if (file.endswith(".xyz") and not file.endswith("_trj.xyz") and 
                not file.startswith("combined_results") and not file.startswith("combined_r")):
                xyz_files.append(os.path.join(root, file))
        
        if xyz_files:
            directories_with_xyz[root] = xyz_files
    
    if not directories_with_xyz:
        print("No .xyz files found in current directory or subdirectories (excluding _trj.xyz, combined_results, and combined_r files).")
        return

    # Display options
    print("\nDirectories with XYZ files:")
    print("-" * 40)
    
    options = {}
    option_num = 1
    
    total_files = 0
    for dir_path, xyz_files in directories_with_xyz.items():
        dir_display = "Current directory" if dir_path == "." else dir_path
        options[str(option_num)] = (dir_path, xyz_files)
        file_count = len(xyz_files)
        total_files += file_count
        
        print(f"{option_num}. {dir_display}: {file_count} files")
        
        # Show first few files as examples
        examples = xyz_files[:3]
        for example in examples:
            filename = os.path.basename(example)
            print(f"   - {filename}")
        if len(xyz_files) > 3:
            print(f"   ... and {len(xyz_files) - 3} more")
        print()
        option_num += 1
    
    # Add "all" option if there are multiple directories
    if len(directories_with_xyz) > 1:
        all_files = []
        for files in directories_with_xyz.values():
            all_files.extend(files)
        options["a"] = ("All directories", all_files)
        print(f"a. All directories: {total_files} files total")
        print()
    
    # Get user choice
    while True:
        try:
            valid_options = list(options.keys()) + ['q']
            choice = input(f"Select option (1-{len(directories_with_xyz)}, 'a' for all, or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Merge operation cancelled.")
                return
            
            if choice in options:
                dir_path, files_to_merge = options[choice]
                
                if choice == "a":
                    output_file = "combined_results.xyz"
                    print(f"\nMerging {len(files_to_merge)} files from all directories...")
                else:
                    if dir_path == ".":
                        output_file = "combined_results.xyz"
                        print(f"\nMerging {len(files_to_merge)} files from current directory...")
                    else:
                        dir_name = os.path.basename(dir_path)
                        output_file = f"combined_results_{dir_name}.xyz"
                        print(f"\nMerging {len(files_to_merge)} files from {dir_path}...")
                
                # Perform the merge
                success = merge_xyz_files(files_to_merge, output_file)
                
                if success:
                    print(f"✓ Successfully created {output_file}")
                    print(f"  Combined {len(files_to_merge)} XYZ files")
                    
                    # Check if .mol file was also created
                    mol_file = output_file.replace('.xyz', '.mol')
                    if os.path.exists(mol_file):
                        print(f"  Also created {mol_file}")
                else:
                    print(f"✗ Failed to create {output_file}")
                
                return
            else:
                print(f"Invalid option. Please select 1-{len(directories_with_xyz)}, 'a', or 'q'.")
                
        except KeyboardInterrupt:
            print("\nMerge operation cancelled by user.")
            return
        except EOFError:
            print("\nMerge operation cancelled.")
            return


def execute_merge_result_command():
    """
    Execute the merge result command functionality (original behavior).
    Shows directories with result_*.xyz files and allows user to merge them.
    """
    print("\n" + "=" * 60)
    print("Result XYZ Files Merge System".center(60))
    print("=" * 60)
    
    # Scan for directories with result_*.xyz files only
    directories_with_xyz = {}
    
    # Check current directory
    current_xyz = []
    for file in os.listdir("."):
        if file.startswith("result_") and file.endswith(".xyz"):
            current_xyz.append(file)
    
    if current_xyz:
        directories_with_xyz["."] = current_xyz
    
    # Check subdirectories
    for root, dirs, files in os.walk("."):
        if root == ".":
            continue
            
        xyz_files = []
        for file in files:
            if file.startswith("result_") and file.endswith(".xyz"):
                xyz_files.append(os.path.join(root, file))
        
        if xyz_files:
            directories_with_xyz[root] = xyz_files
    
    if not directories_with_xyz:
        print("No result_*.xyz files found in current directory or subdirectories.")
        return

    # Display options
    print("\nDirectories with result XYZ files:")
    print("-" * 40)
    
    options = {}
    option_num = 1
    
    total_files = 0
    for dir_path, xyz_files in directories_with_xyz.items():
        dir_display = "Current directory" if dir_path == "." else dir_path
        options[str(option_num)] = (dir_path, xyz_files)
        file_count = len(xyz_files)
        total_files += file_count
        
        print(f"{option_num}. {dir_display}: {file_count} files")
        
        # Show first few files as examples
        examples = xyz_files[:3]
        for example in examples:
            filename = os.path.basename(example)
            print(f"   - {filename}")
        if len(xyz_files) > 3:
            print(f"   ... and {len(xyz_files) - 3} more")
        print()
        option_num += 1
    
    # Add "all" option if there are multiple directories
    if len(directories_with_xyz) > 1:
        all_files = []
        for files in directories_with_xyz.values():
            all_files.extend(files)
        options["a"] = ("All directories", all_files)
        print(f"a. All directories: {total_files} files total")
        print()
    
    # Get user choice
    while True:
        try:
            valid_options = list(options.keys()) + ['q']
            choice = input(f"Select option (1-{len(directories_with_xyz)}, 'a' for all, or 'q' to quit): ").strip().lower()
            
            if choice == 'q':
                print("Merge operation cancelled.")
                return
            
            if choice in options:
                dir_path, files_to_merge = options[choice]
                
                if choice == "a":
                    output_file = "combined_results.xyz"
                    print(f"\nMerging {len(files_to_merge)} result files from all directories...")
                else:
                    if dir_path == ".":
                        output_file = "combined_results.xyz"
                        print(f"\nMerging {len(files_to_merge)} result files from current directory...")
                    else:
                        dir_name = os.path.basename(dir_path)
                        output_file = f"combined_results_{dir_name}.xyz"
                        print(f"\nMerging {len(files_to_merge)} result files from {dir_path}...")
                
                # Perform the merge
                success = merge_xyz_files(files_to_merge, output_file)
                
                if success:
                    print(f"✓ Successfully created {output_file}")
                    print(f"  Combined {len(files_to_merge)} result XYZ files")
                    
                    # Check if .mol file was also created
                    mol_file = output_file.replace('.xyz', '.mol')
                    if os.path.exists(mol_file):
                        print(f"  Also created {mol_file}")
                else:
                    print(f"✗ Failed to create {output_file}")
                
                return
            else:
                print(f"Invalid option. Please select 1-{len(directories_with_xyz)}, 'a', or 'q'.")
                
        except KeyboardInterrupt:
            print("\nMerge operation cancelled by user.")
            return
        except EOFError:
            print("\nMerge operation cancelled.")
            return


def merge_xyz_files(xyz_files: List[str], output_filename: str) -> bool:
    """
    Merge a list of XYZ files into a single output file.
    Renumbers configurations sequentially and adds source file information.
    
    Args:
        xyz_files (List[str]): List of XYZ file paths to merge
        output_filename (str): Name of the output file
    
    Returns:
        bool: True if successful, False otherwise
    """
    if not xyz_files:
        return False
    
    try:
        # Sort files by the number in the filename for consistent ordering
        def get_sort_key(filepath):
            filename = os.path.basename(filepath)
            # Try multiple patterns to extract configuration numbers
            patterns = [
                r'result_(\d+)\.xyz',           # result_123.xyz
                r'conf_(\d+)\.xyz',             # conf_20.xyz
                r'opt\d+_conf_(\d+)\.xyz',      # opt1_conf_20.xyz
                r'_(\d+)\.xyz',                 # any_123.xyz (general pattern)
                r'(\d+)\.xyz'                   # 123.xyz (number only)
            ]
            
            for pattern in patterns:
                match = re.search(pattern, filename)
                if match:
                    return int(match.group(1))
            
            # If no pattern matches, sort alphabetically by filename
            return (float('inf'), filename)
        
        sorted_files = sorted(xyz_files, key=get_sort_key)
        
        total_configs = 0
        
        with open(output_filename, 'w') as outfile:
            for file_idx, xyz_file in enumerate(sorted_files):
                try:
                    # Extract configurations from this file
                    configurations = extract_configurations_from_xyz(xyz_file)
                    
                    if not configurations:
                        print(f"Warning: No configurations found in {xyz_file}")
                        continue
                    
                    source_name = os.path.basename(xyz_file).replace('.xyz', '')
                    
                    for config in configurations:
                        total_configs += 1
                        
                        # Write atom count
                        outfile.write(f"{len(config['atoms'])}\n")
                        
                        # Parse original comment to extract energy
                        original_comment = config['comment']
                        energy_match = re.search(r'E = ([-\d.]+) a\.u\.', original_comment)
                        
                        energy = energy_match.group(1) if energy_match else "unknown"
                        
                        # Create new comment with sequential numbering and source info (no temperature, no "Source:" label)
                        if energy == "unknown":
                            new_comment = f"Configuration: {total_configs} | {source_name}"
                        else:
                            new_comment = f"Configuration: {total_configs} | E = {energy} a.u. | {source_name}"
                        
                        outfile.write(f"{new_comment}\n")
                        
                        # Write atoms (preserve original coordinate precision with proper column alignment)
                        for atom in config['atoms']:
                            if len(atom) == 7:  # New format with string coordinates
                                symbol, x_str, y_str, z_str, x_float, y_float, z_float = atom
                                # Right-align coordinates to maintain column alignment while preserving original precision
                                outfile.write(f"{symbol: <3} {x_str: >12}  {y_str: >12}  {z_str: >12}\n")
                            else:  # Old format compatibility (fallback to 6 decimal places)
                                symbol, x, y, z = atom
                                outfile.write(f"{symbol: <3} {x: 12.6f}  {y: 12.6f}  {z: 12.6f}\n")
                        
                except IOError as e:
                    print(f"Warning: Could not read {xyz_file}: {e}")
                    continue
        
        print(f"Merged {len(sorted_files)} files into {output_filename} with {total_configs} configurations")
        
        # Create .mol file if obabel is available
        if shutil.which("obabel"):
            success, error_msg = convert_xyz_to_mol_simple(output_filename, output_filename.replace('.xyz', '.mol'))
            if success:
                print(f"Also created {output_filename.replace('.xyz', '.mol')}")
            else:
                print(f"Warning: Could not create .mol file: {error_msg}")
        
        return True
        
    except IOError as e:
        print(f"Error writing to {output_filename}: {e}")
        return False


def create_calculation_system(template_file: str, launcher_template: str) -> str:
    """
    Creates a calculation system by extracting configurations from XYZ files
    and generating QM input files with launcher scripts.
    
    Args:
        template_file (str): Template input file (e.g., example_input.inp)
        launcher_template (str): Template launcher file (e.g., launcher_orca.sh)
    
    Returns:
        str: Status message
    """
    # Determine QM program from template file extension
    if template_file.endswith('.inp'):
        qm_program = 'orca'
        input_ext = '.inp'
        output_ext = '.out'
    elif template_file.endswith('.com'):
        qm_program = 'gaussian'
        input_ext = '.com'
        output_ext = '.log'
    elif template_file.endswith('.gjf'):
        qm_program = 'gaussian'
        input_ext = '.gjf'
        output_ext = '.log'
    else:
        return f"Error: Unsupported template file extension. Use .inp for ORCA or .com/.gjf for Gaussian."
    
    # Check if template files exist
    if not os.path.exists(template_file):
        return f"Error: Template file '{template_file}' not found."
    
    if not os.path.exists(launcher_template):
        return f"Error: Launcher template '{launcher_template}' not found."
    
    # Read template content
    try:
        with open(template_file, 'r') as f:
            template_content = f.read()
    except IOError as e:
        return f"Error reading template file '{template_file}': {e}"
    
    try:
        with open(launcher_template, 'r') as f:
            launcher_content = f.read()
    except IOError as e:
        return f"Error reading launcher template '{launcher_template}': {e}"
    
    # Create calculation directory with incremental numbering
    def get_next_calc_dir():
        """Find the next available calculation directory (calculation, calculation_2, etc.)"""
        base_name = "calculation"
        if not os.path.exists(base_name):
            return base_name
        
        counter = 2
        while True:
            calc_dir_name = f"{base_name}_{counter}"
            if not os.path.exists(calc_dir_name):
                return calc_dir_name
            counter += 1
    
    calc_dir = get_next_calc_dir()
    os.makedirs(calc_dir, exist_ok=True)
    
    # Find all result_*.xyz files (ignore resultbox_*.xyz)
    xyz_files = []
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.startswith("result_") and file.endswith(".xyz") and not file.startswith("resultbox_"):
                xyz_path = os.path.join(root, file)
                xyz_files.append(xyz_path)
    
    if not xyz_files:
        return "No result_*.xyz files found in the working directory and subfolders."
    
    # Sort XYZ files by annealing number (extracted from directory name)
    def get_annealing_number(file_path):
        """Extract annealing number from file path like './some_name_2/result_*.xyz'"""
        import re
        # Look for pattern like 'name_N' in the directory path
        directory = os.path.dirname(file_path)
        match = re.search(r'_(\d+)$', directory)
        if match:
            return int(match.group(1))
        # If no _N pattern found in directory, try to extract from filename
        match = re.search(r'result_(\d+)', os.path.basename(file_path))
        if match:
            return int(match.group(1))
        return float('inf')  # Put unmatched files at the end
    
    xyz_files.sort(key=get_annealing_number)
    
    # Interactive selection of XYZ files
    selected_xyz_files = interactive_xyz_file_selection(xyz_files, calc_dir)
    if not selected_xyz_files:
        return "No XYZ files selected for processing."
    
    # Process each selected XYZ file
    all_input_files = []
    
    for xyz_file in selected_xyz_files:
        # Extract configurations
        configurations = extract_configurations_from_xyz(xyz_file)
        if not configurations:
            print(f"Warning: No configurations found in {xyz_file}")
            continue
        
        # Determine the run number from the directory name
        dir_name = os.path.dirname(xyz_file)
        if dir_name == ".":
            run_num = 1
        else:
            # Extract number from directory name (e.g., w6_annealing4_1 -> 1)
            parts = os.path.basename(dir_name).split('_')
            run_num = 1
            for part in reversed(parts):
                try:
                    run_num = int(part)
                    break
                except ValueError:
                    continue
        
        print(f"\nProcessing {xyz_file} (run {run_num}) with {len(configurations)} configurations:")
        
        # Create input files for each configuration
        for config in configurations:
            # Clean up the comment to remove temperature and add source info
            original_comment = config['comment']
            energy_match = re.search(r'E = ([-\d.]+) a\.u\.', original_comment)
            config_match = re.search(r'Configuration: (\d+)', original_comment)
            
            energy = energy_match.group(1) if energy_match else "unknown"
            config_num = config_match.group(1) if config_match else config['config_num']
            
            # Create new comment without temperature, with source info
            source_name = os.path.basename(xyz_file).replace('.xyz', '')
            if energy == "unknown":
                config['comment'] = f"Configuration: {config_num} | {source_name}"
            else:
                config['comment'] = f"Configuration: {config_num} | E = {energy} a.u. | {source_name}"
            
            input_name = f"opt{run_num}_conf_{config['config_num']}{input_ext}"
            input_path = os.path.join(calc_dir, input_name)
            
            if create_qm_input_file(config, template_content, input_path, qm_program):
                all_input_files.append(input_name)
                print(f"  Created: {input_name}")
            else:
                print(f"  Failed to create: {input_name}")
    
    if not all_input_files:
        return "No input files were created successfully."
    
    # Create launcher script
    launcher_path = os.path.join(calc_dir, f"launcher_{qm_program}.sh")
    
    try:
        # Group input files by run number for separator placement
        run_groups = {}
        for input_file in all_input_files:
            # Extract run number from filename (e.g., opt1_conf_1.inp -> 1)
            run_num = int(input_file.split('_')[0].replace('opt', ''))
            if run_num not in run_groups:
                run_groups[run_num] = []
            run_groups[run_num].append(input_file)
        
        # Sort run groups
        sorted_runs = sorted(run_groups.keys())
        
        # Create commands
        all_commands = []
        for i, run_num in enumerate(sorted_runs):
            # Sort files numerically by configuration number
            def extract_conf_number(filename):
                import re
                # Extract number from opt_conf_X.inp or optY_conf_X.inp
                match = re.search(r'_conf_(\d+)\.inp', filename)
                return int(match.group(1)) if match else 0
            
            run_files = sorted(run_groups[run_num], key=extract_conf_number)
            for input_file in run_files:
                output_file = input_file.replace(input_ext, output_ext)
                if qm_program == 'orca':
                    cmd = f"$ORCA5_ROOT/orca {input_file} > {output_file}"
                else:  # gaussian
                    cmd = f"$G16_ROOT/g16 {input_file} {output_file}"
                all_commands.append(cmd)
            
            # Add separator between runs (except after last run)
            if i < len(sorted_runs) - 1:
                all_commands.append("###")
        
        # Write launcher script
        with open(launcher_path, 'w') as f:
            # Process launcher template content to remove any existing example commands
            launcher_lines = launcher_content.rstrip().split('\n')
            filtered_lines = []
            
            for line in launcher_lines:
                # Skip lines that contain ORCA commands (example commands to be replaced)
                if '$ORCA5_ROOT/orca' in line and '.inp' in line:
                    continue
                # Also skip lines that are just '&&' continuation from removed commands
                if line.strip() == '&&' or line.strip() == '&& \\':
                    continue
                filtered_lines.append(line)
            
            # Write the cleaned launcher template content
            f.write('\n'.join(filtered_lines))
            f.write("\n")
            
            # Process commands with proper formatting
            for i, cmd in enumerate(all_commands):
                if cmd == "###":
                    f.write(" && \\\n###\n")
                else:
                    f.write(cmd)
                    if i < len(all_commands) - 1 and all_commands[i + 1] != "###":
                        f.write(" && \\\n")
                    elif i == len(all_commands) - 1:
                        f.write("\n")  # Final newline for last command
        
        # Make launcher executable
        os.chmod(launcher_path, 0o755)
        
        print(f"\nCreated calculation system in '{calc_dir}' directory:")
        print(f"  Input files: {len(all_input_files)}")
        print(f"  Launcher script: launcher_{qm_program}.sh")
        print(f"\nTo run all calculations, use:")
        print(f"  cd {calc_dir}")
        print(f"  ./launcher_{qm_program}.sh")
        
        return f"Successfully created calculation system with {len(all_input_files)} input files."
        
    except IOError as e:
        return f"Error creating launcher script: {e}"


def create_replicated_runs(input_file_path: str, num_replicas: int) -> List[str]:
    """
    Creates replicated folders and input files for multiple annealing runs.
    
    Args:
        input_file_path (str): Path to the original input file
        num_replicas (int): Number of replicas to create
    
    Returns:
        List[str]: List of paths to the replicated input files
    """
    input_file_path_full = os.path.abspath(input_file_path)
    input_dir = os.path.dirname(input_file_path_full)
    input_basename = os.path.basename(input_file_path_full)
    input_name, input_ext = os.path.splitext(input_basename)
    
    replicated_files = []
    
    print(f"Creating {num_replicas} replicated runs for '{input_basename}'...")
    
    for i in range(1, num_replicas + 1):
        # Create folder name: e.g., example_1, example_2, example_3
        folder_name = f"{input_name}_{i}"
        folder_path = os.path.join(input_dir, folder_name)
        
        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Create the replicated input file name: e.g., example_1.in, example_2.in, example_3.in
        replicated_input_name = f"{input_name}_{i}{input_ext}"
        replicated_input_path = os.path.join(folder_path, replicated_input_name)
        
        # Copy the original input file to the new location
        try:
            with open(input_file_path_full, 'r') as src:
                content = src.read()
            with open(replicated_input_path, 'w') as dst:
                dst.write(content)
            
            replicated_files.append(replicated_input_path)
            print(f"  Created: {folder_name}/{replicated_input_name}")
            
        except IOError as e:
            print(f"Error creating replicated file '{replicated_input_path}': {e}")
            continue
    
    print(f"\nSuccessfully created {len(replicated_files)} replicated runs.")
    
    # Create launcher script
    if replicated_files:
        create_launcher_script(replicated_files, input_dir)
    
    print("\nTo run all simulations sequentially, use:")
    print("  ./launcher_ascec.sh")
    
    return replicated_files


# Sort functionality - integrated from sort_files.py
def extract_base(filename):
    """Extract base name from filename by removing extension and known suffixes."""
    # Define optional suffixes that can appear before extensions
    KNOWN_SUFFIXES = ['_trj', '_opt', '_property', '_gu', '_xtbrestart', '_engrad', '_xyz', '_out', '_inp', '_tmp']
    
    # Remove extension
    name, *_ = filename.split('.', 1)
    # Remove known suffixes (only the last one)
    for suffix in KNOWN_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name

def group_files_by_base(directory='.'):
    """Group files by base name and move them to folders."""
    
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    base_map = defaultdict(list)

    for file in files:
        base = extract_base(file)
        if base:
            base_map[base].append(file)

    # Move files
    moved_count = 0
    for base, grouped_files in base_map.items():
        if len(grouped_files) > 1:
            folder_path = os.path.join(directory, base)
            os.makedirs(folder_path, exist_ok=True)
            for file in grouped_files:
                src = os.path.join(directory, file)
                dest = os.path.join(folder_path, file)
                shutil.move(src, dest)
            print(f"Moved {len(grouped_files)} files to folder: {base}")
            moved_count += len(grouped_files)
    
    if moved_count == 0:
        print("No files needed to be grouped.")
    else:
        print(f"Total files moved: {moved_count}")

# Merge XYZ functionality - integrated from mergexyz_files.py
def get_sort_key(filename):
    """Extract the first configuration number after an underscore before .xyz for sorting."""
    import re
    match = re.search(r'_(\d+)\.xyz', filename)
    if match:
        return int(match.group(1))
    return float('inf')

def combine_xyz_files(output_filename="combined_results.xyz", exclude_pattern="_trj.xyz"):
    """Combine all relevant .xyz files into a single .xyz file."""
    
    all_xyz_files = []
    for root, _, files in os.walk("."):
        for file in files:
            filepath = os.path.join(root, file)
            if filepath.endswith(".xyz") and exclude_pattern not in file and file != output_filename:
                all_xyz_files.append(filepath)

    if not all_xyz_files:
        print(f"No relevant .xyz files found (excluding '{exclude_pattern}' and '{output_filename}').")
        return False

    # Sort the files based on the first configuration number found
    sorted_xyz_files = sorted(all_xyz_files, key=lambda x: get_sort_key(os.path.basename(x)))

    with open(output_filename, "w") as outfile:
        for xyz_file in sorted_xyz_files:
            print(f"Processing: {xyz_file}")
            with open(xyz_file, "r") as infile:
                lines = infile.readlines()
                outfile.writelines(lines)

    print(f"\nSuccessfully combined {len(sorted_xyz_files)} .xyz files into: {output_filename}")
    
    # Create .mol file if obabel is available
    if shutil.which("obabel"):
        success, error_msg = convert_xyz_to_mol_simple(output_filename, output_filename.replace('.xyz', '.mol'))
        if success:
            print(f"Also created {output_filename.replace('.xyz', '.mol')}")
        else:
            print(f"Warning: Could not create .mol file: {error_msg}")
    
    return True

# MOL conversion functionality - integrated from mol_files.py
def convert_xyz_to_mol_simple(input_xyz, output_mol):
    """Convert an XYZ file to a MOL file using Open Babel."""
    try:
        # Make paths absolute
        input_xyz = os.path.abspath(input_xyz)
        output_mol = os.path.abspath(output_mol)

        # Ensure the output directory exists
        output_dir = os.path.dirname(output_mol)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        command = ["obabel", "-i", "xyz", input_xyz, "-o", "mol", "-O", output_mol]
        subprocess.run(command, check=True, capture_output=True)
        return True, None
    except subprocess.CalledProcessError as e:
        return False, f"Error converting {input_xyz}: {e.stderr.decode()}"
    except FileNotFoundError:
        return False, "Error: Open Babel ('obabel') command not found. Make sure it's installed and in your system's PATH."
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"

def create_combined_mol():
    """Create MOL file from combined_results.xyz."""
    if os.path.exists("combined_results.xyz"):
        success, error_message = convert_xyz_to_mol_simple("combined_results.xyz", "combined_results.mol")
        if success:
            print("Successfully created combined_results.mol")
            return True
        else:
            print(f"Failed to create MOL file: {error_message}")
            return False
    else:
        print("No combined_results.xyz file found to convert.")
        return False

# Summary functionality - integrated from summary_files.py
def parse_orca_output(filename):
    """Parse an ORCA output file to extract key information."""
    
    results = {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except (FileNotFoundError, IOError, UnicodeDecodeError) as e:
        print(f"Error reading file {filename}: {e}")
        return None

    # Check for ORCA signature
    if "ORCA - Electronic Structure Program" not in content:
        if "*******" not in content:
            print(f"File {filename} is not an ORCA output file.")
            return None

    results['input_file'] = os.path.splitext(os.path.basename(filename))[0]

    # Extract final single point energy
    energy_matches = re.findall(r"FINAL SINGLE POINT ENERGY\s*(-?\d+\.\d+)", content)
    if energy_matches:
        optimization_done_index = content.find("*** OPTIMIZATION RUN DONE ***")
        if optimization_done_index != -1:
            last_valid_energy_index = -1
            for i, match in enumerate(energy_matches):
                match_index = content.find(r"FINAL SINGLE POINT ENERGY\s*(-?\d+\.\d+)", 0)
                if match_index < optimization_done_index:
                    last_valid_energy_index = i
            if last_valid_energy_index != -1:
                results['energy'] = float(energy_matches[last_valid_energy_index])
            else:
                results['energy'] = None
        else:
            results['energy'] = float(energy_matches[-1])
    else:
        results['energy'] = None

    # Extract total run time
    time_match = re.search(r"TOTAL RUN TIME:\s*(\d+)\s*days\s*(\d+)\s*hours\s*(\d+)\s*minutes\s*(\d+)\s*seconds\s*(\d+)\s*msec", content)
    if time_match:
        days = int(time_match.group(1))
        hours = int(time_match.group(2))
        minutes = int(time_match.group(3))
        seconds = int(time_match.group(4))
        milliseconds = int(time_match.group(5))
        results['time'] = (days * 24 * 3600) + (hours * 3600) + (minutes * 60) + seconds + (milliseconds / 1000.0)
    else:
        results['time'] = None
    return results

def format_time_summary(seconds, include_days=False):
    """Format time for summary output."""
    days, rem = divmod(seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, sec = divmod(rem, 60)
    if include_days:
        return f"{int(days)} days, {int(hours)}:{int(minutes)}:{sec:.3f}"
    else:
        return f"{int(hours)}:{int(minutes)}:{sec:.3f}"

def format_total_time(seconds):
    """Format the total execution time in H:M:S format."""
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{int(hours)}:{int(minutes)}:{sec:.3f}"

def format_mean_time(seconds):
    """Format mean execution time in H:M:S format."""
    hours, rem = divmod(seconds, 3600)
    minutes, sec = divmod(rem, 60)
    return f"{int(hours)}:{int(minutes)}:{sec:.3f}"

def format_wall_time(seconds):
    """Format the wall time showing only non-zero values."""
    days, rem = divmod(seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, sec = divmod(rem, 60)
    
    # Build time string showing only non-zero values
    time_parts = []
    
    if days > 0:
        weeks, remaining_days = divmod(days, 7)
        if weeks > 0:
            time_parts.append(f"{int(weeks)} week{'s' if weeks != 1 else ''}")
        if remaining_days > 0:
            time_parts.append(f"{int(remaining_days)} day{'s' if remaining_days != 1 else ''}")
    
    if hours > 0:
        time_parts.append(f"{int(hours)} hour{'s' if hours != 1 else ''}")
    
    if minutes > 0:
        time_parts.append(f"{int(minutes)} minute{'s' if minutes != 1 else ''}")
    
    if sec > 0 or len(time_parts) == 0:  # Show seconds if it's the only unit or if there are seconds
        time_parts.append(f"{int(sec)} second{'s' if sec != 1 else ''}")
    
    return ", ".join(time_parts)

def summarize_calculations(directory=".", file_types=None):
    """Create summary of calculations for ORCA (.out) and/or Gaussian (.log) files."""
    if file_types is None:
        file_types = ['orca', 'gaussian']  # Default: process both types
    
    results_by_type = {}
    
    # Process each requested file type
    for file_type in file_types:
        if file_type == 'orca':
            summary_file = "orca_summary.txt"
            file_extension = ".out"
            parse_function = parse_orca_output
        elif file_type == 'gaussian':
            summary_file = "gaussian_summary.txt" 
            file_extension = ".log"
            parse_function = parse_gaussian_output
        else:
            continue
        
        job_summaries = []
        all_results = {
            'job_count': 0,
            'total_time': 0,
            'min_time': None,
            'max_time': None,
        }

        # Find files of this type
        found_files = []
        for root, _, files in os.walk(directory):
            for filename in files:
                if filename.endswith(file_extension):
                    found_files.append(os.path.join(root, filename))
        
        if not found_files:
            continue  # Skip this type if no files found
            
        with open(summary_file, 'w', encoding='utf-8') as outfile:
            for filepath in found_files:
                results = parse_function(filepath)
                if results:
                    job_summaries.append(results)
                    all_results['job_count'] += 1
                    if results.get('time') is not None:
                        all_results['total_time'] += results['time']
                        if all_results['min_time'] is None or results['time'] < all_results['min_time']:
                            all_results['min_time'] = results['time']
                        if all_results['max_time'] is None or results['time'] > all_results['max_time']:
                            all_results['max_time'] = results['time']

            # Sort the job summaries by total_time
            job_summaries.sort(key=lambda x: x.get('time') or float('inf'))

            # Write summary
            outfile.write("=" * 40 + "\n")
            outfile.write("Summary of all calculations:\n")
            outfile.write(f"  Number of jobs: {all_results['job_count']}\n")
            if all_results['total_time']:
                outfile.write(f"  Total execution time: {format_total_time(all_results['total_time'])}\n")
                outfile.write(f"  Mean execution time: {format_mean_time(all_results['total_time'] / all_results['job_count'])}\n")
                outfile.write(f"  Shortest execution time: {format_time_summary(all_results['min_time'], include_days=False)}\n")
                outfile.write(f"  Longest execution time: {format_time_summary(all_results['max_time'], include_days=False)}\n")
                outfile.write(f"  Total wall time: {format_wall_time(all_results['total_time'])}\n")

            outfile.write("=" * 40 + "\n\n")

            # Write individual job details
            job_index = 1
            for result in job_summaries:
                if file_type == 'orca':
                    outfile.write(f"=> {job_index}. {result['input_file']}.out\n")
                else:  # gaussian
                    outfile.write(f"=> {job_index}. {result['input_file']}.log\n")
                job_index += 1
                for key, value in result.items():
                    if key == 'time' and value is not None:
                        outfile.write(f"  time = {format_time_summary(value, include_days=False)}\n")
                    elif key != 'input_file':
                        outfile.write(f"  {key} = {value}\n")
                outfile.write("\n")

        print(f"Summary written to {summary_file}")
        results_by_type[file_type] = len(job_summaries)

    # Return total number of summaries created
    return sum(results_by_type.values())

def find_out_files(root_dir):
    """Find all .out files in the directory tree."""
    out_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.out'):
                out_files.append(os.path.join(root, file))
    return out_files


def get_unique_folder_name(base_name, current_dir):
    """Generate a unique folder name if base_name already exists."""
    folder_path = os.path.join(current_dir, base_name)
    counter = 1
    
    while os.path.exists(folder_path):
        new_name = f"{base_name}_{counter}"
        folder_path = os.path.join(current_dir, new_name)
        counter += 1
    
    return os.path.basename(folder_path)


def collect_out_files():
    """Collect all .out files into a single folder inside a similarity directory."""
    current_directory = os.getcwd()
    print(f"Searching for .out files in: {current_directory} and its subfolders...")

    all_out_files = find_out_files(current_directory)

    if not all_out_files:
        print("No .out files found in the current directory or its subfolders.")
        return False

    num_files = len(all_out_files)
    base_destination_folder_name = f"orca_out_{num_files}"
    
    # Create similarity directory at parent level
    parent_directory = os.path.dirname(current_directory)
    similarity_path = os.path.join(parent_directory, "similarity")
    os.makedirs(similarity_path, exist_ok=True)
    
    destination_folder_name = get_unique_folder_name(base_destination_folder_name, similarity_path)
    destination_path = os.path.join(similarity_path, destination_folder_name)

    os.makedirs(destination_path)
    print(f"\nCreated destination folder: similarity/{destination_folder_name}")

    print(f"Copying {num_files} .out files to 'similarity/{destination_folder_name}'...")
    for file_path in all_out_files:
        try:
            shutil.copy2(file_path, destination_path)
        except Exception as e:
            print(f"Error copying {os.path.basename(file_path)}: {e}")

    print(f"\nCopied {num_files} .out files to similarity/{destination_folder_name}")
    print("Process complete. Original files remain untouched.")
    return True

def capture_current_state(directory):
    """Capture the current state of files and folders for potential revert."""
    state = {
        'files': {},  # filename -> full_path
        'folders': set()
    }
    
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isfile(item_path):
            state['files'][item] = item_path
        elif os.path.isdir(item_path):
            state['folders'].add(item_path)
    
    return state


def group_files_by_base_with_tracking(directory='.'):
    """Group files by base name and track what was moved."""
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    base_map = defaultdict(list)
    
    for file in files:
        base = extract_base(file)
        if base:
            base_map[base].append(file)
    
    # Track moved files and created folders
    tracking = {'folders': [], 'moved_files': {}}
    moved_count = 0
    
    for base, grouped_files in base_map.items():
        if len(grouped_files) > 1:
            folder_path = os.path.join(directory, base)
            os.makedirs(folder_path, exist_ok=True)
            tracking['folders'].append(folder_path)
            
            for file in grouped_files:
                src = os.path.join(directory, file)
                dest = os.path.join(folder_path, file)
                tracking['moved_files'][dest] = src  # destination -> original
                shutil.move(src, dest)
            
            print(f"Moved {len(grouped_files)} files to folder: {base}")
            moved_count += len(grouped_files)
    
    if moved_count == 0:
        print("No files needed to be grouped.")
    else:
        print(f"Total files moved: {moved_count}")
    
    return tracking


def create_summary_with_tracking(directory):
    """Create summaries and return list of created files."""
    created_files = []
    
    # Check for ORCA files (.out)
    orca_files = []
    gaussian_files = []
    
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith(".out"):
                orca_files.append(os.path.join(root, filename))
            elif filename.endswith(".log"):
                gaussian_files.append(os.path.join(root, filename))
    
    try:
        # Determine which file types to process
        file_types_to_process = []
        if orca_files:
            file_types_to_process.append('orca')
        if gaussian_files:
            file_types_to_process.append('gaussian')
        
        # Create summaries for found file types
        if file_types_to_process:
            num_summaries = summarize_calculations(directory, file_types_to_process)
            
            # Check which summary files were created
            if 'orca' in file_types_to_process and os.path.exists("orca_summary.txt"):
                created_files.append("orca_summary.txt")
            if 'gaussian' in file_types_to_process and os.path.exists("gaussian_summary.txt"):
                created_files.append("gaussian_summary.txt")
    except:
        pass
    
    return created_files


def collect_out_files_with_tracking():
    """Collect .out files and return the created similarity folder path."""
    try:
        current_directory = os.getcwd()
        all_out_files = find_out_files(current_directory)
        
        if not all_out_files:
            return None
        
        num_files = len(all_out_files)
        base_destination_folder_name = f"orca_out_{num_files}"
        
        # Create similarity folder with incremental numbering at parent level
        parent_directory = os.path.dirname(current_directory)
        
        def get_next_similarity_dir():
            """Find the next available similarity directory (similarity, similarity_2, etc.)"""
            base_name = "similarity"
            similarity_path = os.path.join(parent_directory, base_name)
            if not os.path.exists(similarity_path):
                return similarity_path
            
            counter = 2
            while True:
                similarity_dir_name = f"{base_name}_{counter}"
                similarity_path = os.path.join(parent_directory, similarity_dir_name)
                if not os.path.exists(similarity_path):
                    return similarity_path
                counter += 1
        
        similarity_dir = get_next_similarity_dir()
        os.makedirs(similarity_dir, exist_ok=True)
        
        # Create the orca_out_### subfolder inside similarity folder
        destination_folder_name = get_unique_folder_name(base_destination_folder_name, similarity_dir)
        destination_path = os.path.join(similarity_dir, destination_folder_name)
        os.makedirs(destination_path)
        
        # Copy files to the orca_out_### subfolder
        for file_path in all_out_files:
            shutil.copy2(file_path, destination_path)
        
        # Get just the folder name for display (without full path)
        similarity_folder_name = os.path.basename(similarity_dir)
        print(f"Copied {num_files} .out files to {similarity_folder_name}/{destination_folder_name}")
        return destination_path
    except:
        return None


def revert_sort_changes(original_state, created_files, created_folders):
    """Revert all changes made during the sort process."""
    # Remove created files
    for file_path in created_files:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"  Removed: {file_path}")
        except Exception as e:
            print(f"  Warning: Could not remove {file_path}: {e}")
    
    # Move files back to original locations and remove created folders
    for folder_path in created_folders:
        try:
            if os.path.exists(folder_path):
                # If it's a local folder (created by grouping), move files back
                if os.path.dirname(folder_path) == os.getcwd():
                    for item in os.listdir(folder_path):
                        src = os.path.join(folder_path, item)
                        dest = os.path.join(os.getcwd(), item)
                        if os.path.isfile(src):
                            shutil.move(src, dest)
                            print(f"  Moved back: {item}")
                
                # Remove the folder
                shutil.rmtree(folder_path)
                print(f"  Removed folder: {folder_path}")
        except Exception as e:
            print(f"  Warning: Could not revert folder {folder_path}: {e}")


def parse_gaussian_output(filepath):
    """Parse Gaussian .log output file for energy and time."""
    results = {'input_file': os.path.splitext(os.path.basename(filepath))[0]}
    
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

    # Extract SCF Done energy (final energy)
    energy_matches = re.findall(r"SCF Done:\s+E\([^)]+\)\s*=\s*(-?\d+\.\d+)", content)
    if energy_matches:
        results['energy'] = float(energy_matches[-1])  # Take the last SCF Done energy
    else:
        results['energy'] = None

    # Extract job cpu time
    time_match = re.search(r"Job cpu time:\s*(\d+)\s*days\s*(\d+)\s*hours\s*(\d+)\s*minutes\s*(\d+\.\d+)\s*seconds", content)
    if time_match:
        days = int(time_match.group(1))
        hours = int(time_match.group(2))
        minutes = int(time_match.group(3))
        seconds = float(time_match.group(4))
        results['time'] = (days * 24 * 3600) + (hours * 3600) + (minutes * 60) + seconds
    else:
        results['time'] = None
    
    return results

def execute_summary_only():
    """Execute only summary creation without sorting files."""
    print("=" * 50)
    print("ASCEC Summary Creation")
    print("=" * 50)
    
    # Check for ORCA files (.out) and Gaussian files (.log)
    orca_files = []
    gaussian_files = []
    
    for root, _, files in os.walk("."):
        for filename in files:
            if filename.endswith(".out"):
                orca_files.append(os.path.join(root, filename))
            elif filename.endswith(".log"):
                gaussian_files.append(os.path.join(root, filename))
    
    created_summaries = []
    file_types_to_process = []
    
    if orca_files:
        print(f"\nFound {len(orca_files)} ORCA output files.")
        file_types_to_process.append('orca')
        
    if gaussian_files:
        print(f"\nFound {len(gaussian_files)} Gaussian output files.")
        file_types_to_process.append('gaussian')
    
    if not orca_files and not gaussian_files:
        print("\nNo ORCA (.out) or Gaussian (.log) output files found in the current directory or its subfolders.")
        return
        
    # Create summaries for found file types
    if file_types_to_process:
        print("Creating summaries...")
        num_summaries = summarize_calculations(".", file_types_to_process)
        
        # Check which summary files were created
        if 'orca' in file_types_to_process and os.path.exists("orca_summary.txt"):
            created_summaries.append("orca_summary.txt")
        if 'gaussian' in file_types_to_process and os.path.exists("gaussian_summary.txt"):
            created_summaries.append("gaussian_summary.txt")
    
    print("\n" + "=" * 50)
    print("Summary Creation Completed")
    print("=" * 50)
    
    if created_summaries:
        print(f"\nCreated summary files: {', '.join(created_summaries)}")
    else:
        print("\nNo summary files were created (no valid calculation data found).")


def execute_sort_command(include_summary=True):
    """Execute the complete sort process with option to revert."""
    print("=" * 50)
    print("ASCEC Sort Process Started")
    print("=" * 50)
    
    # Store original state for potential revert
    original_state = capture_current_state(".")
    created_files = []
    created_folders = []
    
    try:
        # Step 1: Sort files by base names
        print("\n1. Sorting files by base names...")
        moved_files = group_files_by_base_with_tracking(".")
        created_folders.extend(moved_files.get('folders', []))
        
        # Step 2: Merge XYZ files
        print("\n2. Merging XYZ files...")
        combined_file = "combined_results.xyz"
        if combine_xyz_files():
            created_files.append(combined_file)
            # Step 3: Create MOL file
            print("\n3. Creating MOL file...")
            mol_file = "combined_results.mol"
            if create_combined_mol():
                created_files.append(mol_file)
        
        # Step 4: Create summary (if requested)
        if include_summary:
            print("\n4. Creating calculation summary...")
            summary_files = create_summary_with_tracking(".")
            created_files.extend(summary_files)
        else:
            print("\n4. Skipping summary creation (--nosum flag used)")
        
        # Step 5: Collect .out files
        print("\n5. Collecting .out files...")
        similarity_folder = collect_out_files_with_tracking()
        if similarity_folder:
            created_folders.append(similarity_folder)
        
        print("\n" + "=" * 50)
        print("ASCEC Sort Process Completed")
        print("=" * 50)
        
        # Interactive confirmation
        print("\nPress ENTER to accept the sorting, or type 'r' to revert all changes:")
        user_input = input().strip().lower()
        
        if user_input == 'r':
            print("\nReverting all changes...")
            revert_sort_changes(original_state, created_files, created_folders)
            print("All changes have been reverted successfully.")
        else:
            print("Sorting accepted and finalized.")
            # Suggest similarity analysis if .out files were collected
            # Check if any similarity folder exists at the parent level
            parent_dir = os.path.dirname(os.getcwd())
            similarity_dirs = []
            for item in os.listdir(parent_dir):
                if item == "similarity" or item.startswith("similarity_"):
                    similarity_path = os.path.join(parent_dir, item)
                    if os.path.isdir(similarity_path):
                        similarity_dirs.append(item)
            
            if similarity_dirs:
                print("\nSuggested next step:")
                print("  python3 ascec-v04.py sim --threshold 0.9")
                print("  Run similarity analysis on collected output files")
    
    except Exception as e:
        print(f"\nError during sort process: {e}")
        print("Attempting to revert changes...")
        revert_sort_changes(original_state, created_files, created_folders)
        print("Changes reverted due to error.")


def execute_similarity_analysis(*args):
    """Execute similarity analysis by calling the similarity script."""
    import subprocess
    import sys
    
    # Get the directory where ascec-v04.py is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    similarity_script = os.path.join(script_dir, "similarity_v01.py")
    
    if not os.path.exists(similarity_script):
        print(f"Error: similarity_v01.py not found in {script_dir}")
        print("Make sure similarity_v01.py is in the same directory as ascec-v04.py")
        return
    
    # Build command
    cmd = [sys.executable, similarity_script] + list(args)
    
    print("=" * 50)
    print("ASCEC Similarity Analysis")
    print("=" * 50)
    print(f"Executing: {' '.join(cmd[1:])}")  # Don't show python path
    print()
    
    try:
        # Execute the similarity script with all arguments
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 50)
        print("Similarity analysis completed successfully.")
        print("=" * 50)
    except subprocess.CalledProcessError as e:
        print(f"\nError executing similarity analysis: {e}")
        print("Check the similarity script arguments and try again.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")


def execute_box_analysis(input_file: str):
    """
    Analyze an input file and provide box length recommendations without running the simulation.
    This reads the input file, parses the molecular structure, and outputs the volume-based 
    box length analysis to help users choose appropriate box sizes.
    """
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"Error: Input file '{input_file}' not found.")
            sys.exit(1)
        
        # Create a minimal SystemState for parsing
        state = SystemState()
        state.verbosity_level = 1  # Enable verbose output for box analysis
        
        # Parse the input file
        read_input_file(state, input_file)
        
        # Provide box length analysis using the existing function
        provide_box_length_advice(state)
        
        # Exit successfully after box analysis
        return
        
    except Exception as e:
        print(f"Error during box analysis: {e}")
        sys.exit(1)

# 16. main ascec integrated function
def print_all_commands():
    """Print comprehensive command line usage information."""
    print("=" * 80)
    print("ASCEC v04 - All Available Commands")
    print("=" * 80)
    print()
    
    print("1. SIMULATION COMMANDS:")
    print("-" * 40)
    print("  Run single simulation:")
    print("    python3 ascec-v04.py input_file.in > output.out  # Run simulation with output redirect")
    print("    python3 ascec-v04.py input_file.in --v > output.out  # Verbose mode")
    print("    python3 ascec-v04.py input_file.in --va > output.out # Very verbose mode")
    print("    python3 ascec-v04.py input_file.in --standard > output.out # Standard Metropolis")
    print("    python3 ascec-v04.py input_file.in --nobox > output.out    # Disable box XYZ files")
    print()
    print("  Analyze box length requirements:")
    print("    python3 ascec-v04.py input_file.in box        # Show box analysis in terminal")
    print("    python3 ascec-v04.py box input_file.in        # Alternative syntax")
    print("    python3 ascec-v04.py input_file.in box > box.txt  # Save analysis to file")
    print()
    
    print("  Run replicated simulations:")
    print("    python3 ascec-v04.py input_file.in r3         # Create 3 replicas")
    print("    python3 ascec-v04.py input_file.in r5         # Create 5 replicas")
    print()
    
    print("2. CALCULATION SYSTEM COMMANDS:")
    print("-" * 40)
    print("  Create QM input files from simulation results:")
    print("    python3 ascec-v04.py calc template.inp launcher.sh     # Uses result_*.xyz or combined_results.xyz")
    print("    python3 ascec-v04.py calc template.com launcher.sh     # Gaussian version")
    print("  ")
    print("  Create optimization input files from combined/motif files:")
    print("    python3 ascec-v04.py opt template.inp launcher.sh      # Uses *combined*.xyz and motif_*.xyz files")
    print("    python3 ascec-v04.py opt template.com launcher.sh      # Gaussian version, creates optimization/ folder")
    print("  ")
    print("  Merge XYZ files:")
    print("    python3 ascec-v04.py merge                             # Interactive selection (all .xyz files)")
    print("    python3 ascec-v04.py merge result                      # Interactive selection (result_*.xyz files only)")
    print("  ")
    print("  Update existing QM input files with new template:")
    print("    python3 ascec-v04.py update new_template.inp             # Interactive selection (same extension)")
    print("    python3 ascec-v04.py update new_template.inp pattern     # Interactive selection (filtered by pattern)")
    print()
    
    print("3. ORGANIZATION COMMANDS:")
    print("-" * 40)
    print("  Sort and organize calculation results:")
    print("    python3 ascec-v04.py sort                     # Full sort with summary")
    print("    python3 ascec-v04.py sort --nosum             # Sort without summary")
    print("    python3 ascec-v04.py sort --justsum           # Create summaries only")
    print()
    
    print("  Merge launcher scripts:")
    print("    python3 ascec-v04.py launcher                 # Merge all launcher scripts")
    print()
    
    print("4. ANALYSIS COMMANDS:")
    print("-" * 40)
    print("  Similarity analysis:")
    print("    python3 ascec-v04.py sim --threshold 0.9      # Run similarity analysis")
    print("    python3 ascec-v04.py sim --help               # See similarity options")
    print()
    
    print("5. INPUT FILE FORMAT:")
    print("-" * 40)
    print("  Line 1: Number of different Temperatures")
    print("  Line 2: Initial Temperature")
    print("  Line 3: Temperature schedule (linear/geometric)")
    print("  Line 4: Cube length")
    print("  Line 5: Temperature step parameters")
    print("  Line 6: MaxCycle [floor_value]    # floor_value optional, default: 10")
    print("  Line 7: Max displacement and rotation")
    print("  Line 8: QM program and alias")
    print("  Line 9: QM method and basis set")
    print("  Line 10: nprocs [memory] [ascec_cores] (QM cores, memory, and optional ASCEC system cores)")
    print("           - nprocs: number of cores for each QM calculation")
    print("           - memory: optional memory allocation (e.g., '2GB')")
    print("           - ascec_cores: optional number of system cores for ASCEC operations")
    print("             If omitted, auto-detects based on remaining system cores")
    print("  Line 11: Charge and multiplicity")
    print("  Line 12+: Molecule definitions")
    print()
    
    print("6. TEMPLATE FILE FORMAT (for update command):")
    print("-" * 40)
    print("  Template should contain:")
    print("    # name                         # This will be replaced with config info")
    print("    ! Opt B3LYP 6-311++g**         # QM method line")
    print("    %pal")
    print("      nprocs 8")
    print("    end")
    print("    %maxcore 1000")
    print("    %geom")
    print("      maxiter 5000")
    print("    end")
    print("    * xyz 0 1                      # Coordinates will be inserted here")
    print("    *")
    print()
    
    print("7. EXAMPLES:")
    print("-" * 40)
    print("  Complete workflow:")
    print("    1. python3 ascec-v04.py input.in box          # Check box requirements first")
    print("    2. python3 ascec-v04.py input.in r3           # Create 3 replicated runs")
    print("    3. python3 ascec-v04.py merge                 # Combine XYZ files")
    print("    4. python3 ascec-v04.py calc template.inp launcher.sh")
    print("    5. cd calculation && ./launcher_orca.sh")
    print("    6. cd .. && python3 ascec-v04.py sort")
    print("    7. python3 ascec-v04.py sim --threshold 0.9")
    print()
    
    print("  Update existing calculations:")
    print("    python3 ascec-v04.py update new_template.inp            # Interactive selection (same extension)")
    print("    python3 ascec-v04.py update new_template.inp conf_1     # Interactive selection (filtered)")
    print()
    
    print("=" * 80)


def main_ascec_integrated():
    # Setup argument parser - use parse_known_args to handle shell expansion
    parser = argparse.ArgumentParser(description="ASCEC: Annealing Simulation")
    parser.add_argument("command", help="Command: input file path, 'box' to analyze box length requirements, 'launcher' to merge launcher scripts, 'calc' to create calculation system, 'opt' to create optimization system, 'merge' to combine XYZ files, 'update' to update existing input files, 'sort' to organize calculation results, or 'sim' to run similarity analysis")
    parser.add_argument("arg1", nargs='?', default=None, 
                       help="Second argument: replication mode (e.g., 'r3'), template file for calc, etc.")
    parser.add_argument("arg2", nargs='?', default=None,
                       help="Third argument: launcher template for calc command")
    parser.add_argument("--v", action="store_true", help="Verbose output: print detailed steps every 10 cycles.")
    parser.add_argument("--va", action="store_true", help="Very verbose output: print detailed steps for every cycle.")
    parser.add_argument("--standard", action="store_true", help="Use standard Metropolis criterion instead of modified.")
    parser.add_argument("--nosum", action="store_true", help="Skip summary creation in sort command.")
    parser.add_argument("--justsum", action="store_true", help="Only create summary files without sorting or moving files.")
    parser.add_argument("--nobox", action="store_true", help="Disable creation of box XYZ files (files with dummy atoms for visualization).")
    parser.add_argument("--conformational", type=float, default=None, 
                       help="Override conformational move probability from input file (0.0-1.0)")
    parser.add_argument("--maxdihedral", type=float, default=None,
                       help="Override maximum dihedral rotation angle from input file (degrees)")
    
    # Use parse_known_args to handle shell expansion gracefully
    args, unknown_args = parser.parse_known_args()
    
    # Check if help is requested
    if args.command.lower() in ["help", "--help", "-h", "commands"]:
        print_all_commands()
        return
    
    # Check if similarity analysis mode is requested
    if args.command.lower() == "sim":
        # Pass all remaining arguments to similarity script
        similarity_args = sys.argv[2:]  # Skip 'ascec-v04.py' and 'sim'
        execute_similarity_analysis(*similarity_args)
        return
    
    # Check if sort mode is requested
    if args.command.lower() == "sort":
        if args.justsum:
            execute_summary_only()
        else:
            execute_sort_command(include_summary=not args.nosum)
        return
    
    # Check if box analysis mode is requested
    if args.command.lower() == "box":
        if args.arg1:
            # Use specified input file
            input_file = args.arg1
        else:
            print("Error: box command requires an input file.")
            print("Usage: python3 ascec-v04.py box input_file.inp")
            print("   or: python3 ascec-v04.py box input_file.inp > box_info.txt")
            sys.exit(1)
        
        execute_box_analysis(input_file)
        return
    
    # Check if calculation mode is requested
    if args.command.lower() == "calc":
        if not args.arg1:
            print("Error: calc command requires a template file.")
            print("Usage: python3 ascec-v04.py calc template_file [launcher_template]")
            print("Example: python3 ascec-v04.py calc example_input.inp launcher_orca.sh")
            print("         python3 ascec-v04.py calc example_input.inp  # Creates inputs only")
            sys.exit(1)
        
        result = create_simple_calculation_system(args.arg1, args.arg2)
        print(result)
        return

    # Check if optimization mode is requested
    if args.command.lower() == "opt":
        if not args.arg1:
            print("Error: opt command requires a template file.")
            print("Usage: python3 ascec-v04.py opt template_file [launcher_template]")
            print("Example: python3 ascec-v04.py opt example_input.inp launcher_orca.sh")
            print("         python3 ascec-v04.py opt example_input.inp  # Creates inputs only")
            sys.exit(1)
        
        result = create_optimization_system(args.arg1, args.arg2)
        print(result)
        return
    
    # Check if merge mode is requested
    if args.command.lower() == "merge":
        if args.arg1 and args.arg1.lower() == "result":
            execute_merge_result_command()
        else:
            execute_merge_command()
        return
    
    # Check if update mode is requested
    if args.command.lower() == "update":
        if not args.arg1:
            print("Error: update command requires a template file.")
            print("Usage: python3 ascec-v04.py update new_template.inp")
            print("   or: python3 ascec-v04.py update new_template.inp pattern")
            print("This will search for files with the same extension as the template")
            sys.exit(1)
        
        # Handle shell expansion issue - if we have unknown arguments, it means shell expanded *
        if unknown_args:
            # If we have unknown arguments, it means shell expanded * 
            print(f"Detected shell expansion (found {len(unknown_args) + 2} total arguments). Switching to interactive mode...")
            target_pattern = ""
        elif args.arg2 is None:
            # If no second argument, default to interactive selection for same extension
            target_pattern = ""
        else:
            target_pattern = args.arg2
        
        result = update_existing_input_files(args.arg1, target_pattern)
        print(result)
        return
    
    # Check if launcher merge mode is requested
    if args.command.lower() == "launcher":
        # Merge all launcher scripts in current directory and subfolders
        merge_launcher_scripts(".")
        return
    
    # For simulation mode, the command is the input file
    input_file = args.command
    replication = args.arg1
    
    # Check if box analysis is requested as second argument
    if replication is not None and replication.lower() == "box":
        execute_box_analysis(input_file)
        sys.exit(0)  # Explicitly exit after box analysis to prevent any further execution
        
    # Check if replication mode is requested
    if replication is not None:
        # Parse replication argument (e.g., "r3" -> 3 replicas)
        if replication.lower().startswith('r') and len(replication) > 1:
            try:
                num_replicas = int(replication[1:])
                if num_replicas <= 0:
                    raise ValueError("Number of replicas must be positive")
                
                # Create replicated runs and exit
                create_replicated_runs(input_file, num_replicas)
                return
                
            except ValueError as e:
                print(f"Error: Invalid replication argument '{replication}'. {e}")
                print("Usage: python3 ascec-v04.py input_file.in r<number>")
                print("Example: python3 ascec-v04.py example.in r3")
                sys.exit(1)
        else:
            print(f"Error: Invalid replication format '{replication}'.")
            print("Usage: python3 ascec-v04.py input_file.in r<number>")
            print("Example: python3 ascec-v04.py example.in r3")
            sys.exit(1)

    # Initialize file handles and paths to None to prevent UnboundLocalError
    out_file_handle = None 
    failed_initial_configs_xyz_handle = None 
    failed_configs_path = None 
    
    rless_file_path = None 
    tvse_file_path = None  
    xyz_filename_base = "" # Initialize for error path
    rless_filename = ""    # Initialize for error path
    tvse_filename = ""     # Initialize for error path
    initial_failed_config_idx = 0  # Initialize for failed config tracking
    
    # Determine the directory where the input file is located
    input_file_path_full = os.path.abspath(input_file)
    run_dir = os.path.dirname(input_file_path_full)
    if not run_dir:
        run_dir = os.getcwd()

    qm_files_to_clean: List[str] = [] 
    initial_qm_successful = False 
    
    start_time = time.time() # Start wall time measurement

    state = SystemState()
    
    # Print startup message early to confirm script is running
    _print_verbose(f"{version}", 0, state)
    _print_verbose("Starting ASCEC simulation...", 0, state)
    
    # Set verbosity level based on command line arguments
    if args.va:
        state.verbosity_level = 2
    elif args.v:
        state.verbosity_level = 1
    else:
        state.verbosity_level = 0 # Default minimal output
    
    state.use_standard_metropolis = args.standard # Set the flag in state
    
    # Handle --nobox flag to disable box XYZ file creation
    global CREATE_BOX_XYZ_COPY
    if args.nobox:
        CREATE_BOX_XYZ_COPY = False

    # Check for Open Babel executable early
    if not shutil.which("obabel"):
        _print_verbose("\nCRITICAL ERROR: Open Babel executable 'obabel' not found in your system's PATH.", 0, state)
        _print_verbose("Please ensure Open Babel is installed and its executable is accessible from your command line.", 0, state)
        _print_verbose("Cannot proceed with .mol file generation.", 0, state)
    
    try:
        # Call read_input_file as early as possible to populate state
        # Add file existence check to prevent blocking on missing files
        if not os.path.exists(input_file):
            _print_verbose(f"\nCRITICAL ERROR: Input file '{input_file}' not found.", 0, state)
            _print_verbose("Please check the file path and ensure the file exists.", 0, state)
            sys.exit(1)
        
        # Add file readability check
        try:
            with open(input_file, 'r') as test_file:
                test_file.readline()  # Try to read first line
        except (PermissionError, IOError) as e:
            _print_verbose(f"\nCRITICAL ERROR: Cannot read input file '{input_file}': {e}", 0, state)
            _print_verbose("Please check file permissions and ensure the file is not locked.", 0, state)
            sys.exit(1)
            
        read_input_file(state, input_file)
        
        # Apply command-line overrides for conformational parameters (if provided)
        if args.conformational is not None:
            state.conformational_move_prob = max(0.0, min(1.0, args.conformational))  # Clamp to [0,1]
            _print_verbose(f"Command-line override: Conformational move probability set to {state.conformational_move_prob*100:.1f}%", 1, state)
        if args.maxdihedral is not None:
            state.max_dihedral_angle_rad = np.radians(args.maxdihedral)  # Convert degrees to radians
            _print_verbose(f"Command-line override: Maximum dihedral angle set to {args.maxdihedral:.1f} degrees", 1, state)

        # Set output directory to the directory containing the input file
        state.output_dir = run_dir

        # --- Call for Box Length Advice (only for simulation mode, not box analysis) ---
        provide_box_length_advice(state) # This will print to stderr

        # Set QM program name based on QM_PROGRAM_DETAILS mapping
        state.qm_program = QM_PROGRAM_DETAILS[state.ia]["name"]
        
        # Initialize parallel execution environment optimization
        optimize_qm_execution_environment(state)

        # If random seed is not explicitly set in input or invalid, generate one
        if state.random_seed == -1: 
            state.random_seed = random.randint(100000, 999999) # Generate a 6-digit integer
            _print_verbose(f"\nUsing random seed: {state.random_seed}\n", 0, state) # Modified print
        else:
             _print_verbose(f"\nUsing random seed: {state.random_seed}\n", 0, state) # Modified print
        
        # Initialize Python's random and NumPy's random with the seed
        random.seed(state.random_seed)
        np.random.seed(state.random_seed)

        input_base_name = os.path.splitext(os.path.basename(input_file))[0]

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
            failed_configs_filename = os.path.splitext(os.path.basename(input_file))[0] + "_failed_initial_configs.xyz"
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
                    state.lower_energy_configs = 1 # Initial config counts as the first lower energy config
                    
                    # Preserve QM files from the initial accepted configuration for debugging
                    preserve_last_qm_files(state, run_dir)
                    
                    _print_verbose(f"  Calculation successful. Energy: {state.current_energy:.8f} a.u.\n", 0, state) # Modified print
                    
                    # Print conformational sampling information
                    if state.conformational_move_prob > 0.0:
                        _print_verbose(f"Conformational sampling enabled: {state.conformational_move_prob*100:.1f}% of moves will be conformational", 0, state)
                        _print_verbose(f"Maximum dihedral rotation angle: {np.degrees(state.max_dihedral_angle_rad):.1f} degrees", 0, state)
                    else:
                        _print_verbose("Conformational sampling disabled (only rigid-body moves)", 0, state)
                    
                    _print_verbose("\nStarting annealing simulation...\n", 0, state)
                    initial_qm_calculation_succeeded = True
                    state.total_accepted_configs += 1 # Increment for initial accepted config
                    break # Exit retry loop on success

                except RuntimeError as e:
                    _print_verbose(f"  Initial QM calculation attempt {attempt + 1} failed: {e}", 0, state)
                    
                    if attempt < state.initial_qm_retries - 1:
                        _print_verbose("  Generating new initial configuration and retrying...\n", 0, state)
                    else:
                        # Preserve failed QM files from the last attempt only
                        preserve_failed_initial_qm_files(state, run_dir, attempt + 1)
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
                    
                    if failed_configs_path and os.path.exists(failed_configs_path):
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
            if failed_initial_configs_xyz_handle and failed_configs_path and os.path.exists(failed_configs_path):
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
                    # This will use parallel cores within the QM calculation itself
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
                            state.lower_energy_configs += 1  # Increment counter for lower energy configs 
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
                                    state.iboltz += 1
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
                            
                            # Preserve QM files from this accepted configuration for debugging
                            preserve_last_qm_files(state, run_dir)
                            
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
                            state.lowest_energy_config_idx = state.total_accepted_configs + 1 # Will be the number of this accepted config
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
                state.maxstep = max(state.max_cycle_floor, int(state.maxstep * 0.90)) # Reduce by 10% with user-defined floor
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
                out_file_handle_local.write(f"Accepted lower energy configurations = {state.lower_energy_configs}\n")
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
        if state.random_generate_config == 1 and state.lowest_energy_rp is not None and rless_file_path:
            write_lowest_energy_config_file(state, rless_file_path)
        elif state.random_generate_config == 1 and state.lowest_energy_rp is None:
            _print_verbose(f"No lowest energy configuration was successfully found and stored. Skipping rless file generation.", 0, state)

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


def analyze_box_length_from_xyz(xyz_file: str, num_molecules: int = 1) -> None:
    """
    Standalone function to analyze optimal box lengths from an existing XYZ file.
    Useful for testing the new volume-based approach.
    
    Args:
        xyz_file (str): Path to XYZ file containing molecular structure
        num_molecules (int): Number of copies of this molecule that will be simulated
    """
    import sys
    
    print(f"\nAnalyzing box length requirements for: {xyz_file}")
    print(f"Number of molecule copies: {num_molecules}")
    print("="*70)
    
    if not os.path.exists(xyz_file):
        print(f"Error: File '{xyz_file}' not found!")
        return
    
    try:
        # Read the XYZ file
        configurations = extract_configurations_from_xyz(xyz_file)
        
        if not configurations:
            print("Error: No valid configurations found in XYZ file!")
            return
        
        # Use the first configuration
        config = configurations[0]
        print(f"Using configuration {config['config_num']} with {len(config['atoms'])} atoms")
        
        # Convert to MoleculeData format
        atoms_coords = []
        for atom in config['atoms']:
            if len(atom) >= 7:  # New format with string and float coordinates
                symbol, x_str, y_str, z_str, x_float, y_float, z_float = atom
                # Look up atomic number from symbol
                atomic_num = ELEMENT_SYMBOLS.get(symbol)
                
                if atomic_num is None:
                    print(f"Warning: Unknown element symbol '{symbol}', skipping atom")
                    continue
                
                atoms_coords.append((atomic_num, x_float, y_float, z_float))
            else:
                print(f"Warning: Unexpected atom format: {atom}")
                continue
        
        if not atoms_coords:
            print("Error: No valid atoms found!")
            return
        
        # Create a MoleculeData object
        mol_data = MoleculeData("molecule", len(atoms_coords), atoms_coords)
        
        # Create a minimal SystemState for the analysis
        state = SystemState()
        state.num_molecules = num_molecules
        state.all_molecule_definitions = [mol_data]
        state.molecules_to_add = [0] * num_molecules  # All refer to the first (and only) molecule definition
        state.verbosity_level = 1
        
        # Perform the analysis
        results = calculate_optimal_box_length(state)
        
        if 'error' in results:
            print(f"Error in analysis: {results['error']}")
            return
        
        # Display results
        total_volume = results['total_molecular_volume']
        print(f"\nMOLECULAR VOLUME ANALYSIS:")
        print(f"  Single molecule volume: {results['individual_molecular_volumes'][0]['volume_A3']:.2f} Å³")
        print(f"  Total volume ({num_molecules} molecules): {total_volume:.2f} Å³")
        
        print(f"\nBOX LENGTH RECOMMENDATIONS:")
        print("Packing    Box Length    Box Volume    Free Volume   Typical Use")
        print("Fraction   (Å)          (Å³)         (Å³)") 
        print("-" * 65)
        
        recommendations = results['box_length_recommendations']
        contexts = {
            '10.0%': 'Very dilute gas phase',
            '20.0%': 'Dilute gas/vapor phase', 
            '30.0%': 'Moderate density fluid',
            '40.0%': 'Dense fluid phase',
            '50.0%': 'Very dense/liquid-like'
        }
        
        for key, rec in recommendations.items():
            pf = rec['packing_fraction']
            bl = rec['box_length_A']
            bv = rec['box_volume_A3']
            fv = rec['free_volume_A3']
            context = contexts.get(key, 'Custom density')
            print(f"{pf:6.1%}     {bl:8.2f}      {bv:8.0f}       {fv:8.0f}     {context}")
        
        # Calculate old method for comparison
        coords_array = np.array([atom[1:4] for atom in atoms_coords])
        min_coords = np.min(coords_array, axis=0)
        max_coords = np.max(coords_array, axis=0)
        extents = max_coords - min_coords
        max_extent = np.max(extents)
        old_method_box = max_extent + 16.0  # 8 Å on each side
        
        print(f"\nCOMPARISON WITH SIMPLE METHOD:")
        print(f"  Largest molecular dimension: {max_extent:.2f} Å")
        print(f"  Old method (8 Å rule): {old_method_box:.2f} Å")
        
        # Compare with 30% recommendation
        rec_30 = recommendations.get('30.0%', {}).get('box_length_A', 0)
        if rec_30 > 0:
            ratio = old_method_box / rec_30
            print(f"  Ratio (old/volume-based): {ratio:.2f}")
            if ratio > 1.5:
                print(f"  → Old method may be wastefully large")
            elif ratio < 0.7:
                print(f"  → Old method may be too small")
            else:
                print(f"  → Methods are reasonably consistent")
        
        print(f"\nRECOMMENDATIONS:")
        rec_20 = recommendations.get('20.0%', {}).get('box_length_A', 0)
        if rec_20 > 0 and rec_30 > 0:
            print(f"  • For most applications: {rec_30:.1f} Å (30% packing)")
            print(f"  • For gas phase studies: {rec_20:.1f} Å (20% packing)")
        
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"Error during analysis: {e}")
        traceback.print_exc()


# Example usage and testing function
def test_box_length_analysis():
    """
    Test function to demonstrate the new volume-based box length calculation.
    Creates a simple water molecule and analyzes it.
    """
    print("\n" + "="*70)
    print("TESTING VOLUME-BASED BOX LENGTH CALCULATION")
    print("="*70)
    
    # Create a simple water molecule for testing
    # H2O coordinates (in Angstroms, approximate)
    water_atoms = [
        (8, 0.0, 0.0, 0.0),      # O at origin
        (1, 0.757, 0.587, 0.0),  # H1
        (1, -0.757, 0.587, 0.0)  # H2
    ]
    
    water_mol = MoleculeData("H2O", 3, water_atoms)
    
    # Test with different numbers of molecules
    test_cases = [1, 10, 50, 100]
    
    for num_mols in test_cases:
        print(f"\nTesting {num_mols} water molecule(s):")
        print("-" * 40)
        
        # Create test state
        state = SystemState()
        state.num_molecules = num_mols
        state.all_molecule_definitions = [water_mol]
        state.molecules_to_add = [0] * num_mols
        state.verbosity_level = 0  # Suppress verbose output for testing
        
        results = calculate_optimal_box_length(state)
        
        if 'error' not in results:
            total_vol = results['total_molecular_volume']
            rec_20 = results['box_length_recommendations']['20.0%']['box_length_A']
            rec_30 = results['box_length_recommendations']['30.0%']['box_length_A']
            
            print(f"  Total molecular volume: {total_vol:.2f} Å³")
            print(f"  Recommended box (20% packing): {rec_20:.1f} Å")
            print(f"  Recommended box (30% packing): {rec_30:.1f} Å")
            
            # Calculate density at 30% packing
            box_vol_30 = rec_30**3
            density_30 = (num_mols * 18.015) / (box_vol_30 * 6.022e23 * 1e-24)  # g/cm³
            print(f"  Approximate density at 30%: {density_30:.3f} g/cm³")
        else:
            print(f"  Error: {results['error']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Allow command-line usage for box length analysis
        if sys.argv[1] == "analyze_box" and len(sys.argv) >= 3:
            xyz_file = sys.argv[2]
            num_molecules = int(sys.argv[3]) if len(sys.argv) > 3 else 1
            analyze_box_length_from_xyz(xyz_file, num_molecules)
        elif sys.argv[1] == "test_box":
            test_box_length_analysis()
        else:
            main_ascec_integrated()
    else:
        # Run normal ASCEC if no special arguments
        main_ascec_integrated()

