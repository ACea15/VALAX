"""Curve graph: a runtime container for a set of related curves.

The :class:`CurveGraph` is the data structure that bootstrap instruments
price against.  Each instrument declares which curves (by string
identifier) it touches, and the graph provides lookup to those curves.

In MC-Curves-1, the graph is a thin identifier-keyed wrapper around a
``dict`` of :class:`~valax.curves.discount.DiscountCurve`.  MC-Curves-2
will introduce a static ``CurveSpec`` for declarative curve descriptions
and a joint Newton solver ``bootstrap_curve_graph`` that produces a
:class:`CurveGraph` from a list of bootstrap instruments and curve
specs.

The dict structure is JAX-pytree native: ``jax.tree_util`` traverses
the dict values and treats the string keys as static metadata.  Curves
can be replaced individually via ``eqx.tree_at`` without rebuilding the
entire registry, which is essential for scenario shocking and for the
implicit-adjoint pass through the bootstrap.

Identifier convention (see ``production.md`` §3.1)::

    <asset_class>.<issuer_or_pair>.<index_or_tenor>[.<qualifier>]

Examples:

* ``USD.SOFR.OIS``       — USD SOFR OIS discount curve
* ``USD.SOFR.3M``        — USD 3M-SOFR forward projection curve
* ``EUR.ESTR.OIS``       — EUR €STR OIS discount curve
* ``EUR.EURIBOR.6M``     — EUR 6M-EURIBOR forward projection curve

The graph itself is opinion-free about identifier alphabet: any
hashable string key is accepted.  Convention is enforced at the
:class:`~valax.market.state.MarketState` partitioning layer.
"""

import equinox as eqx

from valax.curves.discount import DiscountCurve


class CurveGraph(eqx.Module):
    """A flat, identifier-keyed registry of discount curves.

    Attributes:
        curves: Mapping from curve identifier to its
            :class:`DiscountCurve`.  All curves should share the same
            ``reference_date``; this invariant is the responsibility of
            the build layer (no runtime check inside the kernel).
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
