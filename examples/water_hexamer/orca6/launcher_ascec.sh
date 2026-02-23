#!/bin/bash

# Configuration for ASCEC v04
# Set ASCEC_ROOT to the directory containing ascec-v04.py
export ASCEC_ROOT="/home/protio/software/ascec04"

# Save original environment paths
_SYSTEM_PATH="$PATH"

# Add the ASCEC directory to the system PATH for direct execution
export PATH="$ASCEC_ROOT:$_SYSTEM_PATH"

echo "ASCEC v04 environment is now active via direct script setup."
echo "ASCEC_ROOT set to: $ASCEC_ROOT"

# Run ASCEC using the full path
python /home/protio/software/ascec04/ascec-v04.py annealing/w6_1/w6_1.in > annealing/w6_1/w6_1.out && \
echo "==================================================================" && \
python /home/protio/software/ascec04/ascec-v04.py annealing/w6_2/w6_2.in > annealing/w6_2/w6_2.out && \
echo "==================================================================" && \
python /home/protio/software/ascec04/ascec-v04.py annealing/w6_3/w6_3.in > annealing/w6_3/w6_3.out
