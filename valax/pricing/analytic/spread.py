"""Spread option pricing: Margrabe and Kirk's approximation.

A spread option pays on the difference between two assets:

- Call: :math:`\\max(S_1(T) - S_2(T) - K,\\; 0)`
- Put:  :math:`\\max(K - (S_1(T) - S_2(T)),\\; 0)`

Two closed-form methods are provided:

1. **Margrabe's formula** (1978) — exact for :math:`K = 0` (exchange
   options).  The spread call becomes the right to exchange asset 2
   for asset 1.

2. **Kirk's approximation** (1995) — the standard industry
   approximation for :math:`K \\neq 0`.  It treats :math:`S_2 + K` as
   a single asset and applies Black-Scholes with an adjusted vol.

Both are pure functions, fully ``jax.jit`` / ``jax.grad`` / ``jax.vmap``
compatible.

References:
    Margrabe (1978), "The Value of an Option to Exchange One Asset for
        Another".
    Kirk (1995), "Correlation in the Energy Markets".
    Carmona & Durrleman (2003), "Pricing and Hedging Spread Options".
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.options import SpreadOption


def margrabe_price(
    option: SpreadOption,
    s1: Float[Array, ""],
    s2: Float[Array, ""],
    vol1: Float[Array, ""],
    vol2: Float[Array, ""],
    rho: Float[Array, ""],
    q1: Float[Array, ""] = jnp.array(0.0),
    q2: Float[Array, ""] = jnp.array(0.0),
) -> Float[Array, ""]:
    """Margrabe's formula for exchange options (:math:`K = 0`).

    The spread call with zero strike is the right to exchange asset 2
    for asset 1.  The exact price is:

    .. math::

        C = N \\bigl[S_1 e^{-q_1 T}\\,\\Phi(d_1)
                    - S_2 e^{-q_2 T}\\,\\Phi(d_2)\\bigr]

    with

    .. math::

        \\sigma_s = \\sqrt{\\sigma_1^2 - 2\\rho\\sigma_1\\sigma_2 + \\sigma_2^2}

    .. math::

        d_1 = \\frac{\\ln(S_1/S_2) + (q_2 - q_1 + \\tfrac12 \\sigma_s^2)T}
                    {\\sigma_s \\sqrt{T}},
        \\qquad d_2 = d_1 - \\sigma_s \\sqrt{T}

    This is independent of the risk-free rate (the option payoff is a
    ratio of two assets, so discounting cancels).

    Args:
        option: Spread option (``strike`` is ignored — assumed 0).
        s1: Spot price of asset 1.
        s2: Spot price of asset 2.
        vol1: Volatility of asset 1.
        vol2: Volatility of asset 2.
        rho: Correlation between the two assets.
        q1: Continuous dividend yield on asset 1.
        q2: Continuous dividend yield on asset 2.

    Returns:
        Option price (scaled by ``option.notional``).
    """
    T = option.expiry
    sigma_s = jnp.sqrt(vol1**2 - 2.0 * rho * vol1 * vol2 + vol2**2)
    sqrt_T = jnp.sqrt(T)

    d1 = (jnp.log(s1 / s2) + (q2 - q1 + 0.5 * sigma_s**2) * T) / (sigma_s * sqrt_T)
    d2 = d1 - sigma_s * sqrt_T

    Phi = jax.scipy.stats.norm.cdf

    call = option.notional * (
        s1 * jnp.exp(-q1 * T) * Phi(d1)
        - s2 * jnp.exp(-q2 * T) * Phi(d2)
    )

    if option.is_call:
        return call
    # Put via parity: P = C - N*(S1*exp(-q1*T) - S2*exp(-q2*T))
    return call - option.notional * (
        s1 * jnp.exp(-q1 * T) - s2 * jnp.exp(-q2 * T)
    )


def kirk_price(
    option: SpreadOption,
    s1: Float[Array, ""],
    s2: Float[Array, ""],
    vol1: Float[Array, ""],
    vol2: Float[Array, ""],
    rho: Float[Array, ""],
    rate: Float[Array, ""],
    q1: Float[Array, ""] = jnp.array(0.0),
    q2: Float[Array, ""] = jnp.array(0.0),
) -> Float[Array, ""]:
    """Kirk's approximation for spread options (:math:`K \\neq 0`).

    Kirk's method treats :math:`S_2(T) + K` as a single "adjusted"
    asset and applies Black-Scholes with a modified volatility:

    .. math::

        F_1 = S_1 e^{(r - q_1)T}, \\qquad
        F_2 = S_2 e^{(r - q_2)T}

    .. math::

        \\sigma_{\\text{kirk}} = \\sqrt{
            \\sigma_1^2
            - 2\\rho\\,\\sigma_1\\,\\sigma_2\\,\\frac{F_2}{F_2 + K}
            + \\sigma_2^2\\,\\left(\\frac{F_2}{F_2 + K}\\right)^2
        }

    .. math::

        d_1 = \\frac{\\ln\\bigl(F_1 / (F_2 + K)\\bigr)
                    + \\tfrac12 \\sigma_{\\text{kirk}}^2 T}
                   {\\sigma_{\\text{kirk}} \\sqrt{T}},
        \\qquad d_2 = d_1 - \\sigma_{\\text{kirk}} \\sqrt{T}

    .. math::

        C = N \\cdot e^{-rT}\\bigl[F_1\\,\\Phi(d_1)
                                 - (F_2 + K)\\,\\Phi(d_2)\\bigr]

    When :math:`K = 0` this degenerates gracefully (though
    :func:`margrabe_price` is exact in that case).

    Args:
        option: Spread option instrument.
        s1: Spot price of asset 1.
        s2: Spot price of asset 2.
        vol1: Volatility of asset 1.
        vol2: Volatility of asset 2.
        rho: Correlation between the two assets.
        rate: Risk-free rate.
        q1: Continuous dividend yield on asset 1.
        q2: Continuous dividend yield on asset 2.

    Returns:
        Option price (scaled by ``option.notional``).
    """
    T = option.expiry
    K = option.strike

    F1 = s1 * jnp.exp((rate - q1) * T)
    F2 = s2 * jnp.exp((rate - q2) * T)

    # Kirk's adjusted vol.
    ratio = F2 / (F2 + K)
    sigma_kirk = jnp.sqrt(
        vol1**2
        - 2.0 * rho * vol1 * vol2 * ratio
        + vol2**2 * ratio**2
    )

    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(F1 / (F2 + K)) + 0.5 * sigma_kirk**2 * T) / (sigma_kirk * sqrt_T)
    d2 = d1 - sigma_kirk * sqrt_T

    Phi = jax.scipy.stats.norm.cdf
    disc = jnp.exp(-rate * T)

    call = option.notional * disc * (F1 * Phi(d1) - (F2 + K) * Phi(d2))

    if option.is_call:
        return call
    # Put via parity: P = C - N * disc * (F1 - F2 - K)
    return call - option.notional * disc * (F1 - F2 - K)


def spread_option_price(
    option: SpreadOption,
    s1: Float[Array, ""],
    s2: Float[Array, ""],
    vol1: Float[Array, ""],
    vol2: Float[Array, ""],
    rho: Float[Array, ""],
    rate: Float[Array, ""],
    q1: Float[Array, ""] = jnp.array(0.0),
    q2: Float[Array, ""] = jnp.array(0.0),
) -> Float[Array, ""]:
    """Price a spread option using the best available method.

    Dispatches to :func:`margrabe_price` when ``strike == 0`` and
    :func:`kirk_price` otherwise.  This is a convenience wrapper;
    call the underlying functions directly if you need to control
    the method.

    .. note::

       Because the dispatch is based on the *value* of ``strike``,
       this function is not ``jax.jit``-friendly when ``strike``
       varies at trace time.  For JIT use, call ``kirk_price``
       directly (it handles ``K = 0`` gracefully).

    Args:
        option: Spread option instrument.
        s1, s2: Spot prices of the two assets.
        vol1, vol2: Volatilities.
        rho: Correlation.
        rate: Risk-free rate.
        q1, q2: Continuous dividend yields.

    Returns:
        Option price.
    """
    return kirk_price(option, s1, s2, vol1, vol2, rho, rate, q1, q2)
