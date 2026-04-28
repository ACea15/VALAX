"""Convexity adjustment plug-ins for money-market futures.

A money-market future and an FRA over the same accrual period
:math:`[T_1, T_2]` are economically similar but pay differently:

* The FRA pays :math:`\\tau\\,(L - K)` at :math:`T_2` in a single
  cashflow.
* The future is daily margined; the position holder receives or posts
  cash equal to the daily change in the futures rate, and that cash
  earns the prevailing short rate.

These two payment styles are martingales under different measures
(:math:`Q^{T_2}` for the FRA, :math:`Q` for the future), so the
expected rate they fix on differs by the **convexity adjustment**:

.. math::
    F^{fut} - F^{FRA} = \\text{convexity adjustment}.

The size of the adjustment is a function of the term-structure model.
This module provides pluggable adjustment factories so that
:class:`~valax.curves.instruments.MoneyMarketFuture` is agnostic to
the choice.

Two variants ship with MC-Curves-1:

* :func:`no_convexity_adj` — returns 0 (use the futures rate as-is).
  Appropriate for short-dated futures where the adjustment is below
  noise (< 1bp at 1Y for typical USD parameters).

* :func:`constant_convexity_adj` — returns a fixed bps value.  The
  desk-supplied number, often produced by an external risk system
  and refreshed daily.

Reserved for a follow-up PR (gated on the short-rate-model
integration with the curve build):

* ``hull_white_convexity_adj(model)`` — derives the adjustment from a
  Hull-White one-factor short-rate model.  Closed-form derivation in
  ``theory.md`` §3.9; will plug in here without changing the
  :class:`MoneyMarketFuture` interface.

Convention: each plug-in returns the adjustment as a scalar in
**rate units** (e.g. ``0.0005`` for 5 bps), matching the units of
``futures_rate`` so that the residual computes
``futures_rate - adjustment - forward_rate(curve)``.
"""

from typing import Callable

import jax.numpy as jnp
from jaxtyping import Float, Int
from jax import Array

from valax.curves.graph import CurveGraph


# Type alias for a convexity adjustment plug-in.  Static field on
# MoneyMarketFuture; takes the curve graph and the period dates and
# returns the rate-unit adjustment to subtract from the futures rate.
ConvexityAdjFn = Callable[
    [CurveGraph, Int[Array, ""], Int[Array, ""]],
    Float[Array, ""],
]


def no_convexity_adj() -> ConvexityAdjFn:
    """Return a plug-in that always reports zero convexity adjustment.

    Use when the futures rate is to be treated as the forward rate
    directly — appropriate for short-dated futures, or when calibration
    is robust to the small front-end adjustment.

    Returns:
        A :data:`ConvexityAdjFn` that ignores its arguments and returns
        ``jnp.asarray(0.0)``.
    """
    zero = jnp.asarray(0.0)

    def adj(
        graph: CurveGraph,
        t0: Int[Array, ""],
        t1: Int[Array, ""],
    ) -> Float[Array, ""]:
        del graph, t0, t1  # unused
        return zero

    return adj


def constant_convexity_adj(bps: float) -> ConvexityAdjFn:
    """Return a plug-in that reports a fixed bps convexity adjustment.

    Desk practice for short-dated and mid-dated futures: produce the
    adjustment externally (often from a calibrated short-rate model
    or from market quote spreads against FRAs) and feed it as a
    constant value.

    Args:
        bps: Convexity adjustment in basis points (e.g. ``5.0`` for
            5 bps = 0.0005 in rate units).

    Returns:
        A :data:`ConvexityAdjFn` that returns ``bps * 1e-4``
        regardless of inputs.
    """
    val = jnp.asarray(bps * 1e-4)

    def adj(
        graph: CurveGraph,
        t0: Int[Array, ""],
        t1: Int[Array, ""],
    ) -> Float[Array, ""]:
        del graph, t0, t1  # constant adjustment, ignores graph & dates
        return val

    return adj
