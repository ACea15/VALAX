"""Bachelier (normal) model closed-form pricing for European options."""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.options import EuropeanOption


def bachelier_price(
    option: EuropeanOption,
    forward: Float[Array, ""],
    vol: Float[Array, ""],
    rate: Float[Array, ""],
) -> Float[Array, ""]:
    """Bachelier (normal model) price for a European option.

    Uses normal (absolute) volatility, where the underlying follows
    dF = sigma * dW rather than dF = sigma * F * dW.

    Args:
        option: European option contract (strike, expiry, is_call).
        forward: Current forward price.
        vol: Normal (absolute) volatility.
        rate: Risk-free discount rate (continuously compounded).

    Returns:
        Option price.
    """
    sqrt_tau = jnp.sqrt(option.expiry)
    d = (forward - option.strike) / (vol * sqrt_tau)

    df = jnp.exp(-rate * option.expiry)
    call = df * vol * sqrt_tau * (d * jax.scipy.stats.norm.cdf(d) + jax.scipy.stats.norm.pdf(d))

    if option.is_call:
        return call
    else:
        return call - df * (forward - option.strike)
