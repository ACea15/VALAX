"""Heston model calibration to option prices or implied vols.

Requires a semi-analytic Heston pricer (not yet implemented) or a
Monte Carlo pricer passed as `pricing_fn`. The pricing function is
injected to decouple the calibration machinery from the pricing engine.
"""

from typing import Callable

import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix
import optax
from jaxtyping import Float
from jax import Array

from valax.models.heston import HestonModel
from valax.calibration.transforms import (
    HESTON_TRANSFORMS,
    model_to_unconstrained,
    unconstrained_to_model,
)


def _heston_price_residuals(
    raw_params: dict[str, Float[Array, ""]],
    args: tuple,
) -> Float[Array, " n"]:
    """Residuals in price space for Heston calibration."""
    (transforms, template, pricing_fn,
     strikes, market_prices, spot, rate, dividend, expiry, weights) = args
    model = unconstrained_to_model(raw_params, transforms, template)
    model_prices = jax.vmap(
        lambda K: pricing_fn(model, K, spot, rate, dividend, expiry)
    )(strikes)
    return weights * (model_prices - market_prices)


def _heston_price_sse(
    raw_params: dict[str, Float[Array, ""]],
    args: tuple,
) -> Float[Array, ""]:
    """Scalar SSE loss for Heston calibration."""
    residuals = _heston_price_residuals(raw_params, args)
    return jnp.sum(residuals ** 2)


def _default_heston_guess(
    rate: Float[Array, ""],
    dividend: Float[Array, ""],
) -> HestonModel:
    """Heuristic initial guess for Heston calibration."""
    return HestonModel(
        v0=jnp.array(0.04),
        kappa=jnp.array(2.0),
        theta=jnp.array(0.04),
        xi=jnp.array(0.5),
        rho=jnp.array(-0.7),
        rate=rate,
        dividend=dividend,
    )


def calibrate_heston(
    strikes: Float[Array, " n"],
    market_prices: Float[Array, " n"],
    spot: Float[Array, ""],
    rate: Float[Array, ""],
    dividend: Float[Array, ""],
    expiry: Float[Array, ""],
    pricing_fn: Callable,
    initial_guess: HestonModel | None = None,
    weights: Float[Array, " n"] | None = None,
    solver: str = "levenberg_marquardt",
    max_steps: int = 512,
) -> tuple[HestonModel, optimistix.Solution]:
    """Calibrate Heston parameters to option prices.

    Args:
        strikes: Option strikes.
        market_prices: Observed option prices.
        spot: Current spot price.
        rate: Risk-free rate.
        dividend: Continuous dividend yield.
        expiry: Time to expiry.
        pricing_fn: Callable with signature
            ``(model, strike, spot, rate, dividend, expiry) -> price``.
            Can be a semi-analytic pricer, Monte Carlo, or a neural surrogate.
        initial_guess: Starting HestonModel. If None, uses a heuristic.
        weights: Per-strike weights. Default: uniform.
        solver: ``"levenberg_marquardt"`` (default), ``"bfgs"``, or
            ``"optax_adam"``.
        max_steps: Maximum optimizer iterations.

    Returns:
        (fitted_model, solution) — the fitted HestonModel and the
        optimistix Solution with convergence diagnostics.
    """
    if initial_guess is None:
        initial_guess = _default_heston_guess(rate, dividend)

    transforms = dict(HESTON_TRANSFORMS)

    if weights is None:
        weights = jnp.ones_like(strikes)

    y0 = model_to_unconstrained(initial_guess, transforms)
    args = (transforms, initial_guess, pricing_fn,
            strikes, market_prices, spot, rate, dividend, expiry, weights)

    if solver == "levenberg_marquardt":
        opt = optimistix.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
        sol = optimistix.least_squares(
            _heston_price_residuals, opt, y0, args=args, max_steps=max_steps,
        )
    elif solver == "bfgs":
        opt = optimistix.BFGS(rtol=1e-8, atol=1e-8)
        sol = optimistix.minimise(
            _heston_price_sse, opt, y0, args=args, max_steps=max_steps,
        )
    elif solver == "optax_adam":
        opt = optimistix.OptaxMinimiser(
            optax.adam(1e-3), rtol=1e-6, atol=1e-6,
        )
        sol = optimistix.minimise(
            _heston_price_sse, opt, y0, args=args, max_steps=max_steps,
        )
    else:
        raise ValueError(f"Unknown solver: {solver!r}")

    fitted_model = unconstrained_to_model(sol.value, transforms, initial_guess)
    return fitted_model, sol
