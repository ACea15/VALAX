"""Market data container: a single pytree holding all market state."""

import equinox as eqx
from jaxtyping import Float
from jax import Array

from valax.curves.discount import DiscountCurve


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
    """

    spots: Float[Array, " n_assets"]
    vols: Float[Array, " n_assets"]
    dividends: Float[Array, " n_assets"]
    discount_curve: DiscountCurve
