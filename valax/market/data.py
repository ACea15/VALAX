"""Market data container: a single pytree holding all market state."""

import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.curves.discount import DiscountCurve
from valax.curves.graph import CurveGraph


class MarketData(eqx.Module):
    """Complete market state for pricing a portfolio.

    All array fields are differentiable. The nested DiscountCurve is also
    a pytree with differentiable discount_factors, so jax.grad through
    a pricing function that takes a MarketData gives sensitivities to
    every spot, vol, dividend yield, and curve pillar simultaneously.

    Attributes:
        spots: Spot prices per asset.
        vols: Implied volatilities per asset.
        dividends: Continuous dividend yields per asset.
        discount_curve: Term structure of discount factors.
        curve_graph: Optional multi-curve container built by
            :func:`valax.curves.bootstrap_curve_graph`. ``None`` keeps
            the single-curve setup; when present, its curves are
            differentiable pytree leaves like ``discount_curve``.
    """

    spots: Float[Array, " n_assets"]
    vols: Float[Array, " n_assets"]
    dividends: Float[Array, " n_assets"]
    discount_curve: DiscountCurve
    curve_graph: CurveGraph | None = None
