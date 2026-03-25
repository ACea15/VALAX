"""Black-76 closed-form pricing for European options on futures/forwards."""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.options import EuropeanOption


def black76_price(
    option: EuropeanOption,
    forward: Float[Array, ""],
    vol: Float[Array, ""],
    rate: Float[Array, ""],
) -> Float[Array, ""]:
    """Black-76 price for a European option on a forward/futures contract.

    Args:
        option: European option contract (strike, expiry, is_call).
        forward: Current forward/futures price.
        vol: Black implied volatility (annualized, lognormal).
        rate: Risk-free discount rate (continuously compounded).

    Returns:
        Option price.
    """
    sqrt_tau = jnp.sqrt(option.expiry)
    d1 = (jnp.log(forward / option.strike) + 0.5 * vol**2 * option.expiry) / (vol * sqrt_tau)
    d2 = d1 - vol * sqrt_tau

    df = jnp.exp(-rate * option.expiry)
    call = df * (forward * jax.scipy.stats.norm.cdf(d1) - option.strike * jax.scipy.stats.norm.cdf(d2))

    if option.is_call:
        return call
    else:
        return call - df * (forward - option.strike)
