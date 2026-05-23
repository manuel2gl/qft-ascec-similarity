"""Adapter registry — name -> QM-engine class lookup.

A QM package is a *name* (``"xtb"``, ``"orca"``, ``"gaussian"``) at the
CLI/asc-file layer; this module is the bridge from that name back to the
concrete :class:`QuantumChemistryAdapter` subclass that handles it. The
mapping is built at import time by ``@register_adapter`` decorators on each
adapter class — there is no central list to keep in sync.

Adding a QM package is a one-file affair: write a subclass of
:class:`QuantumChemistryAdapter`, decorate it with :func:`register_adapter`,
and add a side-effect import in ``quantum_chemistry/__init__.py`` so the
decorator fires at package load. Nothing else in the codebase needs editing.

ORCA is the one package with two adapters (``orca`` for 4.x/5.x via cclib,
``orca_opi`` for 6.1+ via the ORCA Python Interface). The ``.asc`` input only
says ``"orca"`` — the actual version is sniffed at runtime by
:func:`detect_orca_version` and dispatched by :func:`resolve_orca_adapter`.
"""

from __future__ import annotations

import importlib.util
import re
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Type, Union

from cosmic_ascec.exceptions import QMError
from cosmic_ascec.quantum_chemistry.base import QuantumChemistryAdapter


_REGISTRY: Dict[str, Type[QuantumChemistryAdapter]] = {}


def register_adapter(
    name: str,
    *aliases: str,
) -> "callable[[Type[QuantumChemistryAdapter]], Type[QuantumChemistryAdapter]]":
    """Decorator: register a :class:`QuantumChemistryAdapter` subclass under ``name``.

    Extra positional arguments are treated as alternate lookup keys (e.g. an
    xTB adapter can register itself as ``"xtb"`` and also ``"gfn2-xtb"``).
    Names are stored lowercased; lookup is case-insensitive.

    Example:

    .. code-block:: python

        @register_adapter("xtb")
        class XTBAdapter(QuantumChemistryAdapter):
            name = "xtb"
            ...
    """

    keys = (name, *aliases)
    lowered = tuple(k.lower() for k in keys)

    def _decorator(cls: Type[QuantumChemistryAdapter]) -> Type[QuantumChemistryAdapter]:
        if not issubclass(cls, QuantumChemistryAdapter):
            raise TypeError(
                f"register_adapter: {cls.__name__} is not a "
                f"QuantumChemistryAdapter subclass"
            )
        for key in lowered:
            existing = _REGISTRY.get(key)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"register_adapter: name {key!r} already registered to "
                    f"{existing.__name__}; cannot rebind to {cls.__name__}"
                )
            _REGISTRY[key] = cls
        # Make sure the class' canonical name attribute is set (subclasses
        # commonly do this themselves, but enforcing it here means the
        # decorator alone is enough metadata for new contributors).
        if not getattr(cls, "name", None):
            cls.name = lowered[0]
        return cls

    return _decorator


def get_adapter(name: str) -> QuantumChemistryAdapter:
    """Return a fresh instance of the adapter registered under ``name``.

    Raises :class:`QMError` if no adapter is registered. Lookup is
    case-insensitive.
    """
    cls = _REGISTRY.get(name.lower())
    if cls is None:
        available = ", ".join(sorted(set(c.name for c in _REGISTRY.values()))) or "<none>"
        raise QMError(
            f"No quantum-chemistry adapter registered under {name!r}. "
            f"Available: {available}."
        )
    return cls()


def list_adapters() -> Iterable[str]:
    """Return the sorted list of canonical adapter names (one entry per class)."""
    return sorted({cls.name for cls in _REGISTRY.values()})


def clear_registry() -> None:
    """Wipe the registry. Test fixture only; never call from production code."""
    _REGISTRY.clear()


# --------------------------------------------------------------------------- #
# ORCA version auto-detection                                                  #
# --------------------------------------------------------------------------- #
#
# ORCA 6.1 introduced a structured property JSON file (``.property.json``) that
# the ORCA Python Interface (``opi`` package) parses far more reliably than
# the cclib text-scrape. So we ship two ORCA adapters: ``orca`` (cclib, works
# for any version) and ``orca_opi`` (OPI, only ORCA 6.1+). The .asc input
# never records a version, so we sniff it at run time — either from an
# existing output file or by running ``orca --version``.

_ORCA_VERSION_RE = re.compile(r"Program Version\s+(\d+)\.(\d+)", re.IGNORECASE)
_ORCA_OPI_MIN_VERSION: Tuple[int, int] = (6, 1)
"""Minimum ORCA version that emits the property JSON consumed by OPI."""


def opi_importable() -> bool:
    """Return whether the ``opi`` package (ORCA Python Interface) is installed."""
    return importlib.util.find_spec("opi") is not None


def detect_orca_version(
    *,
    executable: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Optional[Tuple[int, int]]:
    """Detect an ORCA ``(major, minor)`` version, or ``None`` if undetectable.

    An ``output_path`` is consulted first — an ORCA ``.out`` file records the
    exact version that produced it in the header. Only the first 500 lines (or
    everything before ``INPUT FILE``) are scanned, which keeps this fast and
    robust to header growth across versions. Failing that, ``executable`` is
    run with ``--version``. Every failure mode collapses to ``None`` — the
    caller treats an unknown version as "assume the cclib adapter".
    """
    if output_path is not None:
        try:
            with open(output_path, "r", encoding="utf-8", errors="ignore") as handle:
                for index, line in enumerate(handle):
                    if "INPUT FILE" in line or index > 500:
                        break
                    match = _ORCA_VERSION_RE.search(line)
                    if match:
                        return (int(match.group(1)), int(match.group(2)))
        except OSError:
            pass

    if executable:
        try:
            completed = subprocess.run(
                [executable, "--version"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            blob = (completed.stdout or "") + (completed.stderr or "")
            match = _ORCA_VERSION_RE.search(blob)
            if match:
                return (int(match.group(1)), int(match.group(2)))
        except (OSError, subprocess.SubprocessError):
            pass

    return None


def resolve_orca_adapter(
    *,
    executable: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> QuantumChemistryAdapter:
    """Return the ORCA adapter matching the detected version.

    Decision table:

    * ORCA ≥ 6.1 *and* the ``opi`` package is importable → ``orca_opi`` adapter.
    * Anything else (older ORCA, unknown version, OPI missing) → ``orca``
      adapter (cclib), which still parses 6.x output correctly via text fallback.

    The cclib ``orca`` adapter also self-delegates to ``orca_opi`` inside
    :meth:`parse_result`, so callers that hardcode ``get_adapter("orca")``
    still get OPI-quality properties on a 6.1+ output file.
    """
    version = detect_orca_version(executable=executable, output_path=output_path)
    if version is not None and version >= _ORCA_OPI_MIN_VERSION and opi_importable():
        return get_adapter("orca_opi")
    return get_adapter("orca")


__all__ = [
    "clear_registry",
    "detect_orca_version",
    "get_adapter",
    "list_adapters",
    "opi_importable",
    "register_adapter",
    "resolve_orca_adapter",
]
