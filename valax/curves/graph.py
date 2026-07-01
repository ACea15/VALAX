"""Curve graph: a runtime container for a set of related curves.

The :class:`CurveGraph` is the data structure that bootstrap instruments
price against.  Each instrument declares which curves (by string
identifier) it touches, and the graph provides lookup to those curves.

The :class:`CurveSpec` is the static, declarative description of a
single curve consumed by the joint multi-curve Newton solver
(:func:`valax.curves.bootstrap_graph.bootstrap_curve_graph`).  It
carries the curve identifier, currency, pillar dates, interpolation
mode, and day-count convention.  MC-Curves-2 ships ``log_linear_df``
as the only supported interpolation mode; monotone-convex / tension
spline variants are queued for MC-Curves-3.

Identifier alphabet (frozen in MC-Curves-2, see ``production.md`` §11.3)::

    <CCY>.<INDEX>.<TENOR>[.<QUALIFIER>]

where ``CCY`` is an ISO-4217 code, ``INDEX`` is the reference index
name (``SOFR``, ``ESTR``, ``EURIBOR``, ...), and ``TENOR`` is either
``OIS`` (discount role) or a numeric tenor label (``1M``, ``3M``,
``6M``, ``12M``, ``1Y``, ...).  The regex :data:`_CURVE_ID_RE` enforces
this on every :class:`CurveSpec` at construction time.  The sentinel
``_default_`` is grandfathered as an exception for the pre-MC-Curves-2
single-curve bootstrap path.

Examples:

* ``USD.SOFR.OIS``       — USD SOFR OIS discount curve
* ``USD.SOFR.3M``        — USD 3M-SOFR forward projection curve
* ``EUR.ESTR.OIS``       — EUR €STR OIS discount curve
* ``EUR.EURIBOR.6M``     — EUR 6M-EURIBOR forward projection curve

Partitioning contract (used by the workflow layer):

* ``TENOR == "OIS"`` → discount-curve role.
* ``TENOR`` is a numeric label → forward-projection-curve role.

The dict structure inside :class:`CurveGraph` is JAX-pytree native:
``jax.tree_util`` traverses the dict values and treats the string keys
as static metadata.  Curves can be replaced individually via
``eqx.tree_at`` without rebuilding the entire registry, which is
essential for scenario shocking and for the implicit-adjoint pass
through the bootstrap.
"""

import re

import equinox as eqx
from jaxtyping import Int
from jax import Array

from valax.curves.discount import DiscountCurve


# ── Curve-id alphabet (frozen MC-Curves-2) ───────────────────────────

#: The sentinel identifier emitted by the pre-MC-Curves-2 single-curve
#: bootstrap path.  Grandfathered as an exception to :data:`_CURVE_ID_RE`.
_DEFAULT_CURVE_ID = "_default_"

#: Regex enforcing the ``<CCY>.<INDEX>.<TENOR>[.<QUALIFIER>]`` alphabet
#: on every :class:`CurveSpec`.  See ``production.md`` §11.3.
_CURVE_ID_RE = re.compile(
    r"^[A-Z]{3}\.[A-Z][A-Z0-9_]*\.(OIS|\d+[DWMY])(\.[A-Za-z0-9_]+)?$"
)


def _validate_curve_id(curve_id: str) -> None:
    """Validate ``curve_id`` against the frozen alphabet.

    The sentinel ``_default_`` is accepted for backwards compatibility
    with the single-curve bootstrap path.  Any other string must match
    :data:`_CURVE_ID_RE`.  Raises :class:`ValueError` on mismatch.
    """
    if curve_id == _DEFAULT_CURVE_ID:
        return
    if not _CURVE_ID_RE.fullmatch(curve_id):
        raise ValueError(
            f"Invalid curve identifier {curve_id!r}. Expected the "
            f"pattern <CCY>.<INDEX>.<TENOR>[.<QUALIFIER>] (e.g. "
            f"'USD.SOFR.OIS', 'EUR.EURIBOR.6M'), or the sentinel "
            f"{_DEFAULT_CURVE_ID!r} for legacy single-curve paths."
        )


#: Supported interpolation modes on a :class:`CurveSpec`.  MC-Curves-2
#: ships ``log_linear_df`` only; ``monotone_convex`` / ``linear_zero`` /
#: ``tension_spline`` are queued for MC-Curves-3.
_SUPPORTED_INTERP = ("log_linear_df",)


class CurveSpec(eqx.Module):
    """Static, declarative description of one curve to be bootstrapped.

    Consumed by :func:`valax.curves.bootstrap_graph.bootstrap_curve_graph`
    together with a list of :class:`BootstrapInstrument` quotes to
    produce a calibrated :class:`CurveGraph`.

    Attributes:
        curve_id: Curve identifier, e.g. ``"USD.SOFR.OIS"``.  Enforced
            against the frozen alphabet (see module docstring).
        currency: ISO-4217 currency code, e.g. ``"USD"``.  Redundant
            with the prefix of ``curve_id`` in the standard case but
            exposed explicitly for downstream partitioning without
            having to parse the identifier at runtime.
        pillar_dates: Sorted ordinal dates for the curve's unknown
            pillars (i.e. **excluding** the reference date, at which
            ``DF == 1`` is imposed by construction).  Shape ``(n,)``.
        interp: Interpolation mode.  Currently only ``"log_linear_df"``
            is supported.  Static field.
        day_count: Day count convention for the emitted
            :class:`DiscountCurve`.  Static field.

    Note:
        ``pillar_dates`` is a differentiable-looking JAX array, but in
        practice it carries integer ordinals and enters no gradient
        path — the solver only differentiates through log-DFs.
    """

    curve_id: str = eqx.field(static=True)
    currency: str = eqx.field(static=True)
    pillar_dates: Int[Array, " n"]
    interp: str = eqx.field(static=True, default="log_linear_df")
    day_count: str = eqx.field(static=True, default="act_365")

    def __check_init__(self) -> None:
        """Validate ``curve_id`` alphabet and interpolation mode.

        Called by :mod:`equinox` after ``__init__`` on every construction
        (including pytree unflattening).  Fails fast at build time.
        """
        _validate_curve_id(self.curve_id)
        if self.interp not in _SUPPORTED_INTERP:
            raise ValueError(
                f"CurveSpec.interp={self.interp!r} not supported. "
                f"MC-Curves-2 supports {_SUPPORTED_INTERP!r}. "
                f"Additional variants are queued for MC-Curves-3."
            )


class CurveGraph(eqx.Module):
    """A flat, identifier-keyed registry of discount curves.

    Attributes:
        curves: Mapping from curve identifier to its
            :class:`DiscountCurve`.  All curves should share the same
            ``reference_date``; this invariant is the responsibility of
            the build layer (no runtime check inside the kernel).

    The graph itself is opinion-free about identifier alphabet: any
    hashable string key is accepted, so scenario shocking (e.g. renaming
    a curve for a stress) does not fail construction.  Alphabet
    enforcement lives on :class:`CurveSpec`, i.e. at the joint-solver
    entry point.
    """

    curves: dict

    def __getitem__(self, curve_id: str) -> DiscountCurve:
        """Look up a curve by identifier.  Raises ``KeyError`` if absent."""
        return self.curves[curve_id]

    def __contains__(self, curve_id: str) -> bool:
        return curve_id in self.curves

    def keys(self):
        """Return the iterable of curve identifiers in the graph."""
        return self.curves.keys()

    def values(self):
        return self.curves.values()

    def items(self):
        return self.curves.items()
