"""Closed-form cash-or-nothing digital option pricing under Black-Scholes."""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.options import DigitalOption
from valax.pricing.analytic.black_scholes import _d1d2


def digital_option_price(
    option: DigitalOption,
    spot: Float[Array, ""],
    vol: Float[Array, ""],
    rate: Float[Array, ""],
    dividend: Float[Array, ""],
) -> Float[Array, ""]:
    """Black-Scholes price of a cash-or-nothing digital option.

    A cash-or-nothing call pays ``payout`` if ``S_T > K`` and nothing
    otherwise; the price is ``payout * e^{-rT} N(d2)``. The put pays if
    ``S_T < K``, with price ``payout * e^{-rT} N(-d2)``.

    Args:
        option: Digital option contract.
        spot: Current spot price.
        vol: Volatility.
        rate: Risk-free rate.
        dividend: Continuous dividend yield.

    Returns:
        Option price.
    """
    _, d2 = _d1d2(spot, option.strike, option.expiry, vol, rate, dividend)
    df = jnp.exp(-rate * option.expiry)
    ncdf = jax.scipy.stats.norm.cdf
    if option.is_call:
        return option.payout * df * ncdf(d2)
    return option.payout * df * ncdf(-d2)
