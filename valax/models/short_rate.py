"""Shared contract of the exact-fit short-rate models (HW, G2++, Cheyette).

Each of these models is *exact-fitted* to an initial discount curve
:math:`P^M(0, t)`: the deterministic shift (Hull-White's :math:`\\theta`,
G2++'s :math:`\\varphi`, Cheyette's initial forward) is built so the model
reprices it. The curve is therefore market data the model is *bound to*, not
one of its parameters:

* the model carries ``initial_curve: DiscountCurve | None = None`` and a static
  ``curve_id: str | None = None``;
* :func:`valax.core.pricing.price` binds ``initial_curve`` from the
  ``MarketState`` at pricing time — the curve named by ``curve_id``, else the
  instrument's discount curve (see
  :func:`valax.core.binding.curve_requirements`) — so curve risk is reported
  once, on the market, and a calibrated model is a curve-free pytree that
  stacks under ``vmap``;
* kernel-level functions (``hw_bond_price``, the Jamshidian swaption, the PDE
  and MC builders) read the bound curve and raise through
  :func:`require_initial_curve` if it is missing. Bind one explicitly with
  :func:`valax.core.binding.bind_curves` when calling them directly.
"""

from __future__ import annotations

from typing import Any

from valax.curves.discount import DiscountCurve


def require_initial_curve(model: Any) -> DiscountCurve:
    """The curve an exact-fit short-rate model is bound to.

    Trace-safe: ``None`` is pytree structure, so the check is a Python
    branch resolved at trace time.

    Args:
        model: A short-rate model with an ``initial_curve`` field.

    Returns:
        The bound :class:`~valax.curves.discount.DiscountCurve`.

    Raises:
        ValueError: If the model is unbound (``initial_curve is None``).
    """
    curve = model.initial_curve
    if curve is None:
        raise ValueError(
            f"{type(model).__name__} is not bound to a curve "
            f"(initial_curve=None). Price it through valax.price, which binds "
            f"the curve from the MarketState, or bind one explicitly with "
            f"valax.core.binding.bind_curves(model, curve) before calling a "
            f"kernel directly."
        )
    return curve


__all__ = ["require_initial_curve"]
