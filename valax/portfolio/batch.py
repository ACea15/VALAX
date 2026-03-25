"""Vectorized batch pricing via jax.vmap."""

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

import equinox as eqx

from valax.greeks.autodiff import greeks as compute_greeks


def batch_price(
    pricing_fn: Callable,
    instruments: eqx.Module,
    spots: Float[Array, " n"],
    vols: Float[Array, " n"],
    rates: Float[Array, " n"],
    dividends: Float[Array, " n"],
) -> Float[Array, " n"]:
    """Price a batch of instruments in parallel via vmap.

    Args:
        pricing_fn: Scalar pricing function.
        instruments: A pytree where each leaf has a leading batch dimension.
        spots, vols, rates, dividends: Batched market inputs.

    Returns:
        Vector of prices.
    """
    return jax.vmap(pricing_fn)(instruments, spots, vols, rates, dividends)


def batch_greeks(
    pricing_fn: Callable,
    instruments: eqx.Module,
    spots: Float[Array, " n"],
    vols: Float[Array, " n"],
    rates: Float[Array, " n"],
    dividends: Float[Array, " n"],
) -> dict[str, Float[Array, " n"]]:
    """Compute price + Greeks for a batch of instruments via vmap.

    Returns:
        Dict where each value is a vector of length n.
    """
    return jax.vmap(lambda inst, s, v, r, d: compute_greeks(pricing_fn, inst, s, v, r, d))(
        instruments, spots, vols, rates, dividends
    )
