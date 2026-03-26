"""Residual and loss functions for model calibration.

Functions follow the optimistix signature: fn(y, args) -> residuals/scalar.
"""

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.calibration.transforms import TransformSpec, unconstrained_to_model


def vol_residuals(
    raw_params: dict[str, Float[Array, ""]],
    args: tuple,
) -> Float[Array, " n"]:
    """Weighted residuals in implied-vol space.

    Args:
        raw_params: Unconstrained parameter dict (optimized by solver).
        args: Tuple of (transforms, template, vol_fn, strikes, market_vols,
              forward, expiry, weights).

    Returns:
        Residual vector: weights * (model_vols - market_vols).
    """
    transforms, template, vol_fn, strikes, market_vols, forward, expiry, weights = args
    model = unconstrained_to_model(raw_params, transforms, template)
    model_vols = jax.vmap(lambda K: vol_fn(model, forward, K, expiry))(strikes)
    return weights * (model_vols - market_vols)


def price_residuals(
    raw_params: dict[str, Float[Array, ""]],
    args: tuple,
) -> Float[Array, " n"]:
    """Weighted residuals in price space.

    Args:
        raw_params: Unconstrained parameter dict.
        args: Tuple of (transforms, template, price_fn, instruments,
              market_prices, forward, rate, weights).

    Returns:
        Residual vector: weights * (model_prices - market_prices).
    """
    transforms, template, price_fn, instruments, market_prices, forward, rate, weights = args
    model = unconstrained_to_model(raw_params, transforms, template)
    model_prices = jax.vmap(
        lambda inst: price_fn(inst, forward, rate, model)
    )(instruments)
    return weights * (model_prices - market_prices)


def weighted_sse(
    raw_params: dict[str, Float[Array, ""]],
    args: tuple,
) -> Float[Array, ""]:
    """Scalar sum of squared residuals in vol space.

    For use with scalar minimizers (BFGS, optax).
    """
    residuals = vol_residuals(raw_params, args)
    return jnp.sum(residuals ** 2)
