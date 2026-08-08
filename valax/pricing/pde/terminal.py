"""Terminal (payoff) conditions evaluated on the spatial mesh.

Unlike the Monte Carlo payoff functions (which return per-path cashflows), these
return **grid-shaped** arrays: the option value at expiry at each spatial node.
For a 1-D log-spot grid the spot at node ``x`` is ``S = exp(x)``.
"""

import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.pricing.pde.grids import Grid1D


def vanilla_terminal(
    grid: Grid1D,
    strike: Float[Array, ""],
    is_call: bool,
) -> Float[Array, " n"]:
    """Terminal payoff of a vanilla call/put on a log-spot grid.

    Args:
        grid: Log-spot grid.
        strike: Option strike.
        is_call: True for a call, False for a put.

    Returns:
        Payoff ``max(S - K, 0)`` (call) or ``max(K - S, 0)`` (put) at each node.
    """
    spot = jnp.exp(grid.nodes)
    if is_call:
        return jnp.maximum(spot - strike, 0.0)
    return jnp.maximum(strike - spot, 0.0)


def intrinsic_payoff(
    grid: Grid1D,
    strike: Float[Array, ""],
    is_call: bool,
) -> Float[Array, " n"]:
    """Intrinsic (immediate-exercise) value on the grid — alias of the vanilla
    terminal, used as the obstacle in American early-exercise projection.
    """
    return vanilla_terminal(grid, strike, is_call)


def digital_terminal(
    grid: Grid1D,
    strike: Float[Array, ""],
    payout: Float[Array, ""],
    is_call: bool,
) -> Float[Array, " n"]:
    """Terminal payoff of a cash-or-nothing digital on a log-spot grid.

    The payoff is a step function (``payout`` if in-the-money, else ``0``); the
    resulting oscillations under Crank-Nicolson are damped by Rannacher start-up
    (see :mod:`valax.pricing.pde.schemes`).

    Args:
        grid: Log-spot grid.
        strike: Strike.
        payout: Fixed cash payout if in-the-money.
        is_call: True for a digital call, False for a digital put.

    Returns:
        The grid-shaped terminal payoff.
    """
    spot = jnp.exp(grid.nodes)
    if is_call:
        return jnp.where(spot > strike, payout, 0.0)
    return jnp.where(spot < strike, payout, 0.0)
