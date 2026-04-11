"""Analytic pricing for variance swaps.

Under Black-Scholes, the fair variance strike equals the squared implied
volatility.  This is because a variance swap can be replicated by a
static portfolio of European options across all strikes (the "log contract"
replication), and under constant vol this portfolio costs exactly σ².

In practice (with a volatility smile), the fair strike is higher than
ATM vol² because the replicating portfolio overweights OTM puts, which
are more expensive due to skew.  The BSM formulas here provide the
constant-vol baseline; smile-consistent pricing requires numerical
integration over the vol surface (not yet implemented).

References
----------
- Demeterfi, K. et al. (1999). "More Than You Ever Wanted to Know About
  Volatility Swaps." Goldman Sachs Quantitative Strategies Research Notes.
- Carr, P. and Madan, D. (1998). "Towards a Theory of Volatility Trading."
  *Volatility: New Estimation Techniques for Pricing Derivatives*, Risk Books.
"""

import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.options import VarianceSwap


def variance_swap_fair_strike(
    vol: Float[Array, ""],
) -> Float[Array, ""]:
    """Fair variance strike under Black-Scholes (constant vol).

    Under BSM, the fair variance strike is simply σ²:

    .. math::

        K_{\\text{var}}^{\\text{fair}} = \\sigma^2

    This is the strike at which the variance swap has zero value at
    inception.  It represents the market's expectation of future
    realized variance under risk-neutral pricing.

    Args:
        vol: Black-Scholes implied volatility.

    Returns:
        Fair variance strike (σ²).
    """
    return vol**2


def variance_swap_price(
    swap: VarianceSwap,
    vol: Float[Array, ""],
    rate: Float[Array, ""],
) -> Float[Array, ""]:
    """Mark-to-market value of a variance swap under Black-Scholes.

    The value to the variance buyer (long realized variance) is:

    .. math::

        V = N_{\\text{var}} \\cdot (\\sigma^2 - K_{\\text{var}}) \\cdot e^{-rT}

    This is the discounted expected payoff, since under BSM the
    expected realized variance equals σ².

    A positive value means implied vol has risen above the strike
    (or equivalently, the market now expects higher realized variance).

    Args:
        swap: Variance swap instrument.
        vol: Current Black-Scholes implied volatility.
        rate: Risk-free rate for discounting.

    Returns:
        Mark-to-market value.
    """
    df = jnp.exp(-rate * swap.expiry)
    return swap.notional_var * (vol**2 - swap.strike_var) * df


def variance_swap_price_seasoned(
    swap: VarianceSwap,
    realized_var_so_far: Float[Array, ""],
    elapsed: Float[Array, ""],
    vol: Float[Array, ""],
    rate: Float[Array, ""],
) -> Float[Array, ""]:
    """Mark-to-market of a partially elapsed (seasoned) variance swap.

    For a swap with total observation period ``T = elapsed + remaining``,
    the expected total realized variance is a weighted average of the
    variance already observed and the variance expected over the
    remaining life:

    .. math::

        \\mathbb{E}[\\hat{\\sigma}^2] = \\frac{t}{T} \\sigma_{\\text{realized}}^2
        + \\frac{T - t}{T} \\sigma_{\\text{implied}}^2

    The mark-to-market is then:

    .. math::

        V = N_{\\text{var}} \\cdot \\left(
            \\frac{t}{T} \\sigma_{\\text{realized}}^2
            + \\frac{T - t}{T} \\sigma^2
            - K_{\\text{var}}
        \\right) \\cdot e^{-r(T-t)}

    Args:
        swap: Variance swap (``expiry`` is the **total** observation period T).
        realized_var_so_far: Annualized realized variance over ``[0, elapsed]``.
        elapsed: Time already passed (year fractions).
        vol: Current implied vol for the remaining period.
        rate: Risk-free rate.

    Returns:
        Mark-to-market value.
    """
    T = swap.expiry
    remaining = T - elapsed
    weight_past = elapsed / T
    weight_future = remaining / T
    expected_total_var = weight_past * realized_var_so_far + weight_future * vol**2
    df = jnp.exp(-rate * remaining)
    return swap.notional_var * (expected_total_var - swap.strike_var) * df
