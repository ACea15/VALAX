"""SABR model calibration to a volatility smile."""

import jax.numpy as jnp
import equinox as eqx
import optimistix
import optax
from jaxtyping import Float
from jax import Array

from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol
from valax.calibration.transforms import (
    SABR_TRANSFORMS,
    model_to_unconstrained,
    unconstrained_to_model,
    positive,
    correlation,
)
from valax.calibration.loss import vol_residuals, weighted_sse


def _default_sabr_guess(
    forward: Float[Array, ""],
    market_vols: Float[Array, " n"],
    beta: Float[Array, ""],
) -> SABRModel:
    """Heuristic initial guess for SABR calibration."""
    atm_vol = jnp.median(market_vols)
    alpha = atm_vol * forward ** (1.0 - beta)
    return SABRModel(
        alpha=alpha,
        beta=beta,
        rho=jnp.array(-0.2),
        nu=jnp.array(0.3),
    )


def calibrate_sabr(
    strikes: Float[Array, " n"],
    market_vols: Float[Array, " n"],
    forward: Float[Array, ""],
    expiry: Float[Array, ""],
    initial_guess: SABRModel | None = None,
    fixed_beta: Float[Array, ""] | None = None,
    weights: Float[Array, " n"] | None = None,
    solver: str = "levenberg_marquardt",
    max_steps: int = 256,
) -> tuple[SABRModel, optimistix.Solution]:
    """Calibrate SABR parameters to a volatility smile.

    Args:
        strikes: Option strikes.
        market_vols: Observed implied volatilities at each strike.
        forward: Forward price.
        expiry: Time to expiry.
        initial_guess: Starting SABRModel. If None, uses a heuristic.
        fixed_beta: If provided, beta is fixed (not calibrated).
            Common: 0.5 for equity, 0.0 for rates.
        weights: Per-strike weights for residuals. Default: uniform.
        solver: ``"levenberg_marquardt"`` (default), ``"bfgs"``, or
            ``"optax_adam"``.
        max_steps: Maximum optimizer iterations.

    Returns:
        (fitted_model, solution) — the fitted SABRModel and the
        optimistix Solution with convergence diagnostics.
    """
    beta = fixed_beta if fixed_beta is not None else jnp.array(0.5)

    if initial_guess is None:
        initial_guess = _default_sabr_guess(forward, market_vols, beta)

    if fixed_beta is not None:
        initial_guess = eqx.tree_at(
            lambda m: m.beta, initial_guess, fixed_beta
        )

    # Build transforms — exclude beta if fixed
    if fixed_beta is not None:
        transforms = {
            "alpha": positive(),
            "rho": correlation(),
            "nu": positive(),
        }
    else:
        transforms = dict(SABR_TRANSFORMS)

    if weights is None:
        weights = jnp.ones_like(strikes)

    y0 = model_to_unconstrained(initial_guess, transforms)
    args = (transforms, initial_guess, sabr_implied_vol, strikes,
            market_vols, forward, expiry, weights)

    if solver == "levenberg_marquardt":
        opt = optimistix.LevenbergMarquardt(rtol=1e-8, atol=1e-8)
        sol = optimistix.least_squares(
            vol_residuals, opt, y0, args=args, max_steps=max_steps,
        )
    elif solver == "bfgs":
        opt = optimistix.BFGS(rtol=1e-8, atol=1e-8)
        sol = optimistix.minimise(
            weighted_sse, opt, y0, args=args, max_steps=max_steps,
        )
    elif solver == "optax_adam":
        opt = optimistix.OptaxMinimiser(
            optax.adam(1e-3), rtol=1e-6, atol=1e-6,
        )
        sol = optimistix.minimise(
            weighted_sse, opt, y0, args=args, max_steps=max_steps,
        )
    else:
        raise ValueError(f"Unknown solver: {solver!r}")

    fitted_model = unconstrained_to_model(sol.value, transforms, initial_guess)
    return fitted_model, sol
