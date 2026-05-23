"""Quantum-chemistry adapter layer.

Every annealing step asks a QM program for an energy. This package hides the
specifics of each program (xTB, Gaussian, ORCA, ORCA+OPI) behind one tiny
contract — :class:`QuantumChemistryAdapter` — so the annealing engine never
has to know which program is in use.

Importing this package also imports each concrete adapter module, which is how
their ``@register_adapter`` decorators populate the registry.

To add a new adapter ``foo`` (e.g. PySCF, NWChem, a remote service):

1. Create ``cosmic_ascec/quantum_chemistry/foo.py`` that subclasses
   :class:`QuantumChemistryAdapter` and decorates with ``@register_adapter("foo")``.
   Implement ``energy_function(qm_settings, run_dir, logger)`` to return a
   ``Cluster -> float`` callable (Hartree) — that is the entire contract.
2. Add ``from cosmic_ascec.quantum_chemistry import foo as _foo  # noqa: F401``
   below so the import side-effect fires at package load and the registry sees it.

No other file in the codebase changes — the CLI (``ascec``) and the annealing
loop pick the new adapter up by name from the ``.asc`` input file (Line 9).
"""

from cosmic_ascec.quantum_chemistry.base import QMResult, QuantumChemistryAdapter
from cosmic_ascec.quantum_chemistry.registry import (
    clear_registry,
    detect_orca_version,
    get_adapter,
    list_adapters,
    opi_importable,
    register_adapter,
    resolve_orca_adapter,
)
from cosmic_ascec.quantum_chemistry.runner import calculate_energy, make_energy_function

# Side-effect imports — each adapter module's @register_adapter call only
# fires when its module is imported, so we import every shipped adapter here.
# ``orca`` is imported before ``orca_opi`` because the OPI adapter subclasses
# ``ORCAAdapter``; the order keeps the dependency explicit.
from cosmic_ascec.quantum_chemistry import xtb as _xtb  # noqa: F401
from cosmic_ascec.quantum_chemistry import gaussian as _gaussian  # noqa: F401
from cosmic_ascec.quantum_chemistry import orca as _orca  # noqa: F401
from cosmic_ascec.quantum_chemistry import orca_opi as _orca_opi  # noqa: F401


__all__ = [
    "QMResult",
    "QuantumChemistryAdapter",
    "calculate_energy",
    "clear_registry",
    "detect_orca_version",
    "get_adapter",
    "list_adapters",
    "make_energy_function",
    "opi_importable",
    "register_adapter",
    "resolve_orca_adapter",
]
