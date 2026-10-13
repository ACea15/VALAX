"""Multi-curve framework: separate discount and forward projection curves.

Post-crisis, OIS discounting + tenor-specific forward curves is the standard.
For example, SOFR OIS for discounting and separate 1M/3M SOFR curves for
forward rate projection.

Since MC-Curves-2b this module is a **thin view layer** over the joint
multi-curve Newton solver:
:func:`valax.curves.bootstrap_graph.bootstrap_curve_graph` does the work
and :class:`MultiCurveSet` is a two-tier *view* over the resulting
:class:`~valax.curves.graph.CurveGraph` (see :meth:`MultiCurveSet.from_graph`
and :meth:`MultiCurveSet.to_graph`).  The sequential OIS-then-tenor
pipeline that used to live here is gone.

The two-tier ``bootstrap_multi_curve`` builder has been removed: build with
:func:`valax.curves.bootstrap_graph.bootstrap_curve_graph` (which handles
every quote type — tenor-basis, cross-currency, futures, … — that the
two-tier container cannot express) and take a :class:`MultiCurveSet` view of
the resulting graph with :meth:`MultiCurveSet.from_graph` if a two-tier shape
is convenient.
"""

import equinox as eqx

from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph, _CURVE_ID_RE


# ── Synthetic curve identifiers for the legacy two-tier build ────────

#: Currency prefix used for the synthetic curve ids the legacy wrapper
#: feeds to the joint solver.  ``XXX`` is the ISO-4217 code for "no
#: currency", which is honest: a two-tier view carries no currency
#: information.  :class:`MultiCurveSet` is keyed by the user's own tenor
#: labels; these ids only appear when it is exported with
#: :meth:`MultiCurveSet.to_graph`.
_LEGACY_PREFIX = "XXX.LEGACY"

#: Discount (OIS) leg of the legacy two-tier build.
_LEGACY_DISCOUNT_ID = f"{_LEGACY_PREFIX}.OIS"


def _legacy_forward_id(tenor: str, position: int) -> str:
    """Synthetic curve id for the forward leg labelled ``tenor``.

    Uses ``XXX.LEGACY.<TENOR>`` when the (upper-cased) tenor label is
    conformant with the frozen curve-id alphabet — the ``"3M"`` /
    ``"6M"`` labels every real caller uses — and falls back to the
    qualifier form ``XXX.LEGACY.OIS.FWD<k>`` for free-form labels so
    that arbitrary legacy keys never fail validation.

    Args:
        tenor: The caller's tenor label, e.g. ``"3M"``.
        position: Index of this forward leg in the input dict, used
            only by the fallback id.

    Returns:
        A curve id accepted by :class:`CurveSpec`.
    """
    candidate = f"{_LEGACY_PREFIX}.{tenor.upper()}"
    if _CURVE_ID_RE.fullmatch(candidate):
        return candidate
    return f"{_LEGACY_DISCOUNT_ID}.FWD{position}"


class MultiCurveSet(eqx.Module):
    """Two-tier view over a :class:`~valax.curves.graph.CurveGraph`.

    A pedagogical container for the single-currency case: one OIS
    discount curve plus a tenor-keyed dict of forward projection
    curves.  It holds exactly the same :class:`DiscountCurve` objects
    that live in the graph — :meth:`from_graph` and :meth:`to_graph`
    convert between the two shapes without copying or re-solving.

    New code should prefer the :class:`CurveGraph` directly: it is
    n-currency, n-tenor, and carries the basis / cross-currency curves
    that this two-tier shape cannot name.

    Attributes:
        discount_curve: OIS discount curve used for present-value discounting.
        forward_curves: Tenor-keyed forward projection curves
            (e.g., ``{"3M": curve_3m, "6M": curve_6m}``).
    """

    discount_curve: DiscountCurve
    forward_curves: dict  # str -> DiscountCurve (keys are static in the pytree)

    @classmethod
    def from_graph(
        cls,
        graph: CurveGraph,
        discount_id: str,
        forward_ids: dict[str, str],
    ) -> "MultiCurveSet":
        """Build the two-tier view from a calibrated curve graph.

        Args:
            graph: Curve graph holding the curves (typically the output
                of :func:`valax.curves.bootstrap_curve_graph`).
            discount_id: Identifier of the discounting curve on the graph.
            forward_ids: Mapping from the tenor label you want in
                ``forward_curves`` to the curve identifier on the graph,
                e.g. ``{"3M": "USD.SOFR.3M"}``.

        Returns:
            A :class:`MultiCurveSet` sharing the graph's curve objects.

        Raises:
            KeyError: If any identifier is absent from the graph.
        """
        return cls(
            discount_curve=graph[discount_id],
            forward_curves={
                tenor: graph[curve_id]
                for tenor, curve_id in forward_ids.items()
            },
        )

    def to_graph(
        self,
        discount_id: str = _LEGACY_DISCOUNT_ID,
        forward_ids: dict[str, str] | None = None,
    ) -> CurveGraph:
        """Expose this set as a :class:`CurveGraph`.

        The inverse of :meth:`from_graph`.  Use it to feed legacy
        containers into the graph-aware pricers
        (:mod:`valax.pricing.analytic.graph_pricing`) and the
        ``CurveGraph`` shock primitives in :mod:`valax.risk.shocks`.

        Args:
            discount_id: Identifier to file the discount curve under.
                Defaults to the synthetic legacy id.
            forward_ids: Optional mapping from tenor label to the
                identifier to file that forward curve under.  Missing
                labels fall back to the synthetic legacy ids.

        Returns:
            A :class:`CurveGraph` holding the same curve objects.
        """
        forward_ids = forward_ids or {}
        curves: dict[str, DiscountCurve] = {discount_id: self.discount_curve}
        for position, (tenor, curve) in enumerate(self.forward_curves.items()):
            curve_id = forward_ids.get(
                tenor, _legacy_forward_id(tenor, position)
            )
            curves[curve_id] = curve
        return CurveGraph(curves=curves)
