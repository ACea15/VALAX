"""Bachelier (normal) model closed-form pricing for European options."""

import jax
import jax.numpy as jnp
import optimistix as optx
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


def bachelier_implied_vol(
    option: EuropeanOption,
    forward: Float[Array, ""],
    price: Float[Array, ""],
    rate: Float[Array, ""],
) -> Float[Array, ""]:
    """Invert :func:`bachelier_price` for the normal (absolute) volatility.

    Solves ``bachelier_price(option, forward, vol, rate) == price`` for ``vol``
    using a scalar Newton root-find. Implicit differentiation through
    ``optimistix`` yields clean Greeks (e.g. vega, curve sensitivities) without
    unrolling the solver.

    Args:
        option: European option contract (strike, expiry, is_call).
        forward: Current forward price.
        price: Observed (market) option price to invert.
        rate: Risk-free discount rate (continuously compounded).

    Returns:
        Normal (absolute) volatility that reprices ``price``.
    """
    df = jnp.exp(-rate * option.expiry)
    sqrt_tau = jnp.sqrt(option.expiry)

    # Initial guess from the option's time value via the ATM Bachelier
    # relation price_atm = df * vol * sqrt(T) / sqrt(2*pi).
    moneyness = forward - option.strike
    intrinsic_fwd = moneyness if option.is_call else -moneyness
    intrinsic = df * jnp.maximum(intrinsic_fwd, 0.0)
    time_value = jnp.maximum(price - intrinsic, 1e-12)
    vol0 = time_value * jnp.sqrt(2.0 * jnp.pi) / (df * sqrt_tau)
    vol0 = jnp.maximum(vol0, 1e-8)

    def residual(vol, _):
        return bachelier_price(option, forward, vol, rate) - price

    sol = optx.root_find(
        residual,
        optx.Newton(rtol=1e-12, atol=1e-12),
        vol0,
        max_steps=100,
        throw=False,
    )
    return sol.value
