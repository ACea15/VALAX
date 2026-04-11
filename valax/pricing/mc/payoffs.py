"""Payoff functions for Monte Carlo pricing.

Each payoff takes paths and instrument data, returns per-path cashflows.
All payoffs must be differentiable for pathwise Greeks.
"""

import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.options import (
    EuropeanOption,
    EquityBarrierOption,
    AsianOption,
    LookbackOption,
    VarianceSwap,
)


def european_payoff(
    paths: Float[Array, "n_paths n_steps"],
    option: EuropeanOption,
) -> Float[Array, " n_paths"]:
    """European option payoff: max(S_T - K, 0) for call."""
    terminal = paths[:, -1]
    if option.is_call:
        return jnp.maximum(terminal - option.strike, 0.0)
    else:
        return jnp.maximum(option.strike - terminal, 0.0)


def asian_payoff(
    paths: Float[Array, "n_paths n_steps"],
    option: EuropeanOption,
) -> Float[Array, " n_paths"]:
    """Arithmetic Asian option payoff based on average price."""
    avg_price = jnp.mean(paths[:, 1:], axis=1)  # exclude initial spot
    if option.is_call:
        return jnp.maximum(avg_price - option.strike, 0.0)
    else:
        return jnp.maximum(option.strike - avg_price, 0.0)


def barrier_payoff(
    paths: Float[Array, "n_paths n_steps"],
    option: EuropeanOption,
    barrier: Float[Array, ""],
    is_up: bool,
    is_knock_in: bool,
    smoothing: float = 0.0,
) -> Float[Array, " n_paths"]:
    """Barrier option payoff with optional smoothing for differentiability.

    Args:
        paths: Simulated price paths.
        option: Underlying European option.
        barrier: Barrier level.
        is_up: True for up barrier, False for down barrier.
        is_knock_in: True for knock-in, False for knock-out.
        smoothing: Width of sigmoid smoothing (0 = hard barrier).
    """
    terminal = paths[:, -1]

    if is_up:
        max_price = jnp.max(paths, axis=1)
        if smoothing > 0:
            barrier_hit = jax_sigmoid((max_price - barrier) / smoothing)
        else:
            barrier_hit = (max_price >= barrier).astype(terminal.dtype)
    else:
        min_price = jnp.min(paths, axis=1)
        if smoothing > 0:
            barrier_hit = jax_sigmoid((barrier - min_price) / smoothing)
        else:
            barrier_hit = (min_price <= barrier).astype(terminal.dtype)

    if option.is_call:
        vanilla = jnp.maximum(terminal - option.strike, 0.0)
    else:
        vanilla = jnp.maximum(option.strike - terminal, 0.0)

    if is_knock_in:
        return vanilla * barrier_hit
    else:
        return vanilla * (1.0 - barrier_hit)


def equity_barrier_payoff(
    paths: Float[Array, "n_paths n_steps"],
    option: EquityBarrierOption,
) -> Float[Array, " n_paths"]:
    """Equity barrier option payoff using the instrument's own fields.

    Convenience wrapper around :func:`barrier_payoff` that reads the
    barrier parameters from the :class:`EquityBarrierOption` instrument
    directly, rather than requiring them as separate arguments.
    """
    # Build a temporary EuropeanOption for the vanilla payoff calc
    vanilla = EuropeanOption(
        strike=option.strike, expiry=option.expiry, is_call=option.is_call,
    )
    return barrier_payoff(
        paths, vanilla, option.barrier,
        is_up=option.is_up,
        is_knock_in=option.is_knock_in,
        smoothing=option.smoothing,
    )


def asian_option_payoff(
    paths: Float[Array, "n_paths n_steps"],
    option: AsianOption,
) -> Float[Array, " n_paths"]:
    """Asian option payoff with arithmetic or geometric averaging.

    Averages over all path steps excluding the initial spot (``paths[:, 1:]``).

    Args:
        paths: Simulated price paths, shape ``(n_paths, n_steps)``.
        option: Asian option instrument.

    Returns:
        Per-path payoff.
    """
    observations = paths[:, 1:]  # exclude initial spot

    if option.averaging == "geometric":
        avg = jnp.exp(jnp.mean(jnp.log(observations), axis=1))
    else:
        avg = jnp.mean(observations, axis=1)

    if option.is_call:
        return jnp.maximum(avg - option.strike, 0.0)
    return jnp.maximum(option.strike - avg, 0.0)


def lookback_payoff(
    paths: Float[Array, "n_paths n_steps"],
    option: LookbackOption,
) -> Float[Array, " n_paths"]:
    """Lookback option payoff.

    **Floating strike** (``is_fixed_strike=False``):

    - Call: ``S_T - min(S_t)``  — the right to buy at the lowest price
    - Put:  ``max(S_t) - S_T``  — the right to sell at the highest price

    **Fixed strike** (``is_fixed_strike=True``):

    - Call: ``max(max(S_t) - K, 0)``
    - Put:  ``max(K - min(S_t), 0)``

    Args:
        paths: Simulated price paths.
        option: Lookback option instrument.

    Returns:
        Per-path payoff (always non-negative for floating strike).
    """
    terminal = paths[:, -1]
    path_max = jnp.max(paths, axis=1)
    path_min = jnp.min(paths, axis=1)

    if option.is_fixed_strike:
        if option.is_call:
            return jnp.maximum(path_max - option.strike, 0.0)
        return jnp.maximum(option.strike - path_min, 0.0)
    else:
        if option.is_call:
            return terminal - path_min  # always >= 0
        return path_max - terminal  # always >= 0


def variance_swap_payoff(
    paths: Float[Array, "n_paths n_steps"],
    swap: VarianceSwap,
    annual_factor: Float[Array, ""] = jnp.array(252.0),
) -> Float[Array, " n_paths"]:
    """Variance swap payoff from simulated paths.

    Computes the annualized realized variance from discrete log returns
    and pays the difference vs. the strike:

    .. math::

        \\text{payoff} = N_{\\text{var}} \\cdot (\\hat{\\sigma}^2 - K_{\\text{var}})

    where :math:`\\hat{\\sigma}^2 = \\frac{\\text{annual\\_factor}}{n}
    \\sum_{i=1}^{n} (\\ln S_i / S_{i-1})^2`.

    Note: we use the mean-zero estimator (sum of squared log returns
    without subtracting the mean), which is the market convention for
    variance swaps.

    Args:
        paths: Simulated price paths, shape ``(n_paths, n_steps)``.
        swap: Variance swap instrument.
        annual_factor: Annualization factor (252 for daily, 52 for weekly).

    Returns:
        Per-path payoff.
    """
    log_returns = jnp.diff(jnp.log(paths), axis=1)  # (n_paths, n_steps-1)
    n_obs = log_returns.shape[1]
    realized_var = (annual_factor / n_obs) * jnp.sum(log_returns**2, axis=1)
    return swap.notional_var * (realized_var - swap.strike_var)


def jax_sigmoid(x):
    """Smooth approximation to Heaviside step for differentiable barriers."""
    import jax
    return jax.nn.sigmoid(x)
