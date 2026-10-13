"""Heston model calibration to option prices.

The pricing function is injected (a Fourier pricer such as
:func:`valax.pricing.analytic.heston.cos_pricer`, Monte Carlo, or a neural
surrogate) so the calibration machinery is decoupled from the pricing engine.
For a calibration through the :func:`~valax.core.pricing.price` registry,
bound to a calibrated market, use
:func:`~valax.calibration.nodes.price_fit_node` instead.
"""

from collections.abc import Mapping
from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.calibration.level import LevelInputs
from valax.calibration.result import CalibrationProvenance, CalibrationResult
from valax.calibration.settings import FIT_SETTINGS, SolverSettings
from valax.calibration.solve import solve_level
from valax.calibration.transforms import (
    HESTON_TRANSFORMS,
    model_to_unconstrained,
    unconstrained_to_model,
)
from valax.models.heston import HestonModel


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
    expiry: Float[Array, ""] | Float[Array, " n"],
    pricing_fn: Callable,
    *,
    initial_guess: HestonModel | None = None,
    weights: Float[Array, " n"] | None = None,
    fixed_params: Mapping[str, float | Array] | None = None,
    settings: SolverSettings | None = None,
    provenance: CalibrationProvenance | None = None,
    name: str = "heston",
) -> CalibrationResult:
    """Calibrate Heston parameters to option prices.

    A weighted least-squares level on ``w * (pricing_fn(...) - price)``. The
    quotes are the market prices; spot, rate, dividend, expiry, strikes,
    weights and any ``fixed_params`` are held-fixed inputs, so the level
    adjoint reports their sensitivities alongside the quote ones (the same
    convention as :func:`~valax.calibration.basket.basket_fit_level`).

    A single-expiry smile pins Heston poorly — ``κ``, ``ξ`` and ``v0`` trade
    off along a flat valley. Two remedies, combinable: quote several
    expiries (``expiry`` one per strike), and hold the weakly identified
    parameters fixed (``fixed_params={"kappa": 1.5}``).

    Args:
        strikes: Option strikes.
        market_prices: Observed option prices.
        spot: Current spot price.
        rate: Risk-free rate.
        dividend: Continuous dividend yield.
        expiry: Time to expiry — a scalar shared by every quote, or one
            expiry per strike (a multi-expiry fit).
        pricing_fn: Callable with signature
            ``(model, strike, spot, rate, dividend, expiry) -> price``
            (scalar expiry; it is vmapped per quote). Can be a semi-analytic
            pricer, Monte Carlo, or a neural surrogate.
        initial_guess: Starting HestonModel for the free parameters. If
            None, uses a heuristic. Its values of fixed fields are ignored
            (they come from ``fixed_params``).
        weights: Per-quote weights. Default: uniform.
        fixed_params: ``field -> value`` for Heston fields (``v0``,
            ``kappa``, ``theta``, ``xi``, ``rho``) held fixed; they are
            reported as fixed inputs (``result.inputs.fixed[field]``).
        settings: Solver settings; defaults to
            :data:`~valax.calibration.settings.FIT_SETTINGS`.
        provenance: The engine ``pricing_fn`` implements, recorded on the
            result (e.g. ``CalibrationProvenance("cos", "discretised")``).
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the fitted
        :class:`~valax.models.heston.HestonModel` is ``result.value``.

    Raises:
        ValueError: If ``settings.method`` is not a least-squares method, a
            fixed parameter is not a Heston parameter, every parameter is
            fixed, or ``expiry`` is an array not aligned with ``strikes``.
    """
    settings = FIT_SETTINGS if settings is None else settings
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    expiry = jnp.asarray(expiry, dtype=jnp.float64)
    if expiry.ndim > 0 and expiry.shape != strikes.shape:
        raise ValueError(
            f"calibrate_heston: expiry must be a scalar or aligned with "
            f"strikes {strikes.shape}, got shape {expiry.shape}"
        )
    fixed_params = dict(fixed_params or {})
    transforms = dict(HESTON_TRANSFORMS)
    for f in fixed_params:
        if f not in transforms:
            raise ValueError(
                f"calibrate_heston: unknown Heston parameter {f!r}; "
                f"fixable: {sorted(transforms)}"
            )
        del transforms[f]
    if not transforms:
        raise ValueError("calibrate_heston: every parameter is fixed")
    fixed = {
        "spot": jnp.asarray(spot, dtype=jnp.float64),
        "rate": jnp.asarray(rate, dtype=jnp.float64),
        "dividend": jnp.asarray(dividend, dtype=jnp.float64),
        "expiry": expiry,
        "strikes": strikes,
        "weights": jnp.ones_like(strikes) if weights is None
        else jnp.asarray(weights, dtype=jnp.float64),
    }
    fixed.update(
        {f: jnp.asarray(v, dtype=jnp.float64) for f, v in fixed_params.items()}
    )
    inputs = LevelInputs(
        quotes=jnp.asarray(market_prices, dtype=jnp.float64), fixed=fixed,
    )
    if initial_guess is None:
        initial_guess = _default_heston_guess(
            inputs.fixed["rate"], inputs.fixed["dividend"]
        )

    def assemble(theta: dict, inp: LevelInputs) -> HestonModel:
        fx = inp.fixed
        params = {
            f: fx[f] if f in fixed_params else getattr(initial_guess, f)
            for f in HESTON_TRANSFORMS
        }
        template = HestonModel(
            **params, rate=fx["rate"], dividend=fx["dividend"],
        )
        return unconstrained_to_model(theta, transforms, template)

    def residual(theta: dict, inp: LevelInputs) -> Float[Array, " n"]:
        model = assemble(theta, inp)
        fx = inp.fixed
        expiries = jnp.broadcast_to(fx["expiry"], fx["strikes"].shape)
        model_prices = jax.vmap(
            lambda k, t: pricing_fn(
                model, k, fx["spot"], fx["rate"], fx["dividend"], t
            )
        )(fx["strikes"], expiries)
        return fx["weights"] * (model_prices - inp.quotes)

    theta0 = jax.lax.stop_gradient(
        model_to_unconstrained(initial_guess, transforms)
    )
    return solve_level(
        name, "least_squares", residual, theta0, inputs, assemble, settings,
        provenance=provenance,
    )


__all__ = ["calibrate_heston"]
