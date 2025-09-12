#!/bin/bash

# For ORCA 5.0.4:
export ORCA_BASE=/home/manuel/software
export ORCA5_ROOT=$ORCA_BASE/orca_5_0_4
export OPENMPI4_1_1_ROOT=$ORCA_BASE/openmpi-4.1.1

_SYSTEM_PATH="$PATH"
_SYSTEM_LD_LIBRARY_PATH="$LD_LIBRARY_PATH"

export PATH="$ORCA5_ROOT:$OPENMPI4_1_1_ROOT/bin:$_SYSTEM_PATH"
export LD_LIBRARY_PATH="$ORCA5_ROOT:$OPENMPI4_1_1_ROOT/lib:$_SYSTEM_LD_LIBRARY_PATH"

# Set up xTB environment variables
export XTB_PATH=/home/manuel/software/xtb-dist/share/xtb
export PATH=$PATH:/home/manuel/software/xtb-dist/bin

echo "ORCA 5.0.4 environment is now active via direct script setup."
mpirun --version

###
