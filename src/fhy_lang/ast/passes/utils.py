"""Diagnostic-message formatting shared between validator passes.

Every validator in this package emits diagnostics through
:meth:`fhy_core.pass_infrastructure.CompilerPass.report`. The diagnostic
body follows a common shape: ``"<kind>: [<location>: ]<message>"`` where
``<kind>`` is a short human-readable tag ("structural error", "type error",
...) that lets users (and tests) distinguish diagnostic categories without
carrying a bespoke exception-class hierarchy.

This module is intentionally not re-exported from :mod:`fhy_lang.ast.passes`
-- each validator imports its helpers locally. That keeps diagnostic
formatting an implementation detail of the passes package rather than part
of the public surface.

"""

__all__ = [
    "format_diagnostic_message",
    "format_location_prefix",
]

from fhy_core import Provenance


def _format_location(provenance: Provenance | None) -> str | None:
    if provenance is None:
        return None
    elif provenance.span is not None:
        return str(provenance.span)
    elif provenance.origins:
        return str(provenance.origins[0])
    else:
        return None


def format_diagnostic_message(
    kind: str, message: str, provenance: Provenance | None
) -> str:
    """Build a diagnostic message tagged with a short category name.

    The output is shaped as ``"<kind>: <location>: <message>"`` (the
    location segment is omitted when no provenance is available), so the
    aggregated :class:`~fhy_core.pass_infrastructure.ValidationReport`
    preserves the diagnostic category in its textual rendering.

    Args:
        kind: Short human-readable category tag (e.g., ``"type error"``,
            ``"structural error"``). Callers are expected to use a small,
            stable set of tags so downstream tooling can grep for them.
        message: The human-readable diagnostic message body.
        provenance: Source provenance to surface, if known.

    Returns:
        The tagged, location-prefixed diagnostic string.

    """
    location = _format_location(provenance)
    if location is None:
        return f"{kind}: {message}"
    return f"{kind}: {location}: {message}"


def format_location_prefix(provenance: Provenance | None) -> str:
    """Return ``"<span>: "`` when a span is known and ``""`` otherwise.

    Used for WARNING diagnostics that do not need a category tag but still
    benefit from a visible source location.

    """
    if provenance is None or provenance.span is None:
        return ""
    return f"{provenance.span}: "
