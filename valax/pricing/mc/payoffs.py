"""Payoff functions for Monte Carlo pricing.

Each payoff takes paths and instrument data, returns per-path cashflows.
All payoffs must be differentiable for pathwise Greeks.
"""

import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.instruments.options import EuropeanOption


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


def jax_sigmoid(x):
    """Smooth approximation to Heaviside step for differentiable barriers."""
    import jax
    return jax.nn.sigmoid(x)
