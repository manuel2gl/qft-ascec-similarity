"""ORCA 6.1+ adapter — parsing through the ORCA Python Interface (OPI).

cclib supports ORCA output only up to 5.0. ORCA 6.1 introduced a structured
``.property.json`` file that the ``opi`` package reads natively, giving us a
much more reliable property surface than text-scraping the ``.out``. This
adapter, registered under ``"orca_opi"``, is the 6.1+ counterpart of
:class:`~cosmic_ascec.quantum_chemistry.orca.ORCAAdapter`.

It is a thin subclass — input writing, execution, energy regex, termination
banner are all version-independent text operations, so they are inherited
unchanged. Only :meth:`parse_result` is overridden: OPI confirms termination
and supplies geometry; the property values are still read from the ``.out``
text (matching the layered approach in the original monolith).

OPI is an optional dependency. If it is not installed this adapter falls back
to pure text parsing, identical to the plain ``orca`` adapter. The registry's
:func:`~cosmic_ascec.quantum_chemistry.registry.resolve_orca_adapter` only
routes here when OPI *is* importable, so the degraded path is mostly a safety
net for callers that hardcode ``get_adapter("orca_opi")``.
"""

from __future__ import annotations

from pathlib import Path

from cosmic_ascec.quantum_chemistry.base import QMResult
from cosmic_ascec.quantum_chemistry.orca import ORCAAdapter
from cosmic_ascec.quantum_chemistry.registry import register_adapter

# OPI is optional — guard the import so the module loads without it.
try:  # pragma: no cover - import guard
    from opi.output.core import Output as _OPIOutput
except ImportError:  # pragma: no cover - exercised only without opi
    _OPIOutput = None


@register_adapter("orca_opi", "orca-opi")
class ORCAOPIAdapter(ORCAAdapter):
    """ORCA adapter for the 6.1+ era, parsing through OPI when available."""

    name = "orca_opi"

    def parse_result(self, output_path: Path) -> QMResult:
        """Full property block for ORCA 6.1+.

        The mandatory ``energy`` and the property ``extras`` come from the
        inherited text scan (it already handles every ORCA release). OPI is
        consulted on top of that for an authoritative ``terminated_normally``
        verdict; when OPI is missing or cannot read the directory, the text
        verdict stands.
        """
        # The inherited text parser raises QMError if no energy is present —
        # the same contract every adapter honours.
        base = self._parse_result_text(output_path)
        extras = dict(base.extras)

        converged = base.converged
        opi_verdict = _opi_terminated_normally(output_path)
        if opi_verdict is None:
            extras["parser"] = "opi_text_fallback"
        else:
            extras["parser"] = "opi"
            converged = opi_verdict

        return QMResult(energy=base.energy, converged=converged, extras=extras)


def _opi_terminated_normally(output_path: Path) -> bool | None:
    """Ask OPI whether the ORCA run terminated normally.

    Returns ``None`` — meaning "OPI could not tell, trust the text scan" — when
    ``opi`` is not installed or the directory lacks the files OPI needs. This
    mirrors cosmic-v01's defensive ``try/except`` around every OPI call
    (cosmic-v01.py lines 1602-1631).
    """
    if _OPIOutput is None:
        return None
    try:
        opi_output = _OPIOutput(
            basename=output_path.stem,
            working_dir=output_path.parent,
            version_check=False,
        )
        return bool(opi_output.terminated_normally())
    except Exception:  # noqa: BLE001 - OPI raises a broad family of errors
        return None


__all__ = ["ORCAOPIAdapter"]
