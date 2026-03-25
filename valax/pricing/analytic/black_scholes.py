"""Black-Scholes-Merton closed-form pricing for European options."""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.options import EuropeanOption


def _d1d2(
    spot: Float[Array, ""],
    strike: Float[Array, ""],
    tau: Float[Array, ""],
    vol: Float[Array, ""],
    rate: Float[Array, ""],
    dividend: Float[Array, ""],
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    """Compute d1 and d2 of the Black-Scholes formula."""
    sqrt_tau = jnp.sqrt(tau)
    d1 = (jnp.log(spot / strike) + (rate - dividend + 0.5 * vol**2) * tau) / (
        vol * sqrt_tau
    )
    d2 = d1 - vol * sqrt_tau
    return d1, d2


def black_scholes_price(
    option: EuropeanOption,
    spot: Float[Array, ""],
    vol: Float[Array, ""],
    rate: Float[Array, ""],
    dividend: Float[Array, ""],
) -> Float[Array, ""]:
    """Black-Scholes-Merton price for a European option.

    Args:
        option: European option contract.
        spot: Current underlying price.
        vol: Implied volatility (annualized).
        rate: Risk-free rate (continuously compounded).
        dividend: Continuous dividend yield.

    Returns:
        Option price.
    """
    d1, d2 = _d1d2(spot, option.strike, option.expiry, vol, rate, dividend)

    df = jnp.exp(-rate * option.expiry)
    fwd = spot * jnp.exp((rate - dividend) * option.expiry)

    call_price = df * (fwd * jax.scipy.stats.norm.cdf(d1) - option.strike * jax.scipy.stats.norm.cdf(d2))

    if option.is_call:
        return call_price
    else:
        # Put via put-call parity
        return call_price - df * (fwd - option.strike)


def black_scholes_implied_vol(
    option: EuropeanOption,
    spot: Float[Array, ""],
    rate: Float[Array, ""],
    dividend: Float[Array, ""],
    market_price: Float[Array, ""],
    n_iterations: int = 20,
) -> Float[Array, ""]:
    """Newton-Raphson implied volatility inversion.

    Args:
        option: European option contract.
        spot: Current underlying price.
        rate: Risk-free rate.
        dividend: Continuous dividend yield.
        market_price: Observed option price.
        n_iterations: Number of Newton steps.

    Returns:
        Implied volatility.
    """
    # Price and vega as functions of vol
    price_fn = lambda v: black_scholes_price(option, spot, v, rate, dividend)
    vega_fn = jax.grad(price_fn)

    def newton_step(vol, _):
        p = price_fn(vol)
        v = vega_fn(vol)
        # Clamp vega away from zero to avoid division issues
        v = jnp.maximum(v, 1e-10)
        return vol - (p - market_price) / v, None

    vol_init = jnp.array(0.2)
    vol, _ = jax.lax.scan(newton_step, vol_init, None, length=n_iterations)
    return vol
