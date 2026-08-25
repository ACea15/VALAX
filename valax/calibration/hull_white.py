r"""Hull-White calibration to a European swaption surface.

Hull-White has exactly two free parameters -- the mean-reversion speed
:math:`a` and the short-rate volatility :math:`\sigma` -- because the drift
:math:`\theta(t)` is pinned by the exact-fit condition to the initial curve.
Calibration therefore means fitting two numbers to a whole swaption surface,
which is heavily over-determined: a one-factor Gaussian model cannot reproduce
a smile, and it fits the ATM term structure only approximately.

That over-determination is the point. Desk practice is usually to **fix**
:math:`a` (from a historical estimate or a co-terminal fit) and let
:math:`\sigma` absorb the level, because the two parameters are strongly
correlated along the ATM surface -- :math:`a` controls how fast forward-rate
volatility decays with expiry while :math:`\sigma` sets its level, and over a
narrow expiry range a change in one is nearly offset by a change in the other.
Pass ``fixed_mean_reversion`` to work that way.

Model prices come from the Jamshidian decomposition
(:func:`valax.pricing.analytic.hull_white_swaptions.hw_swaption_price`), which
is closed-form and implicitly differentiable, so the least-squares Jacobian is
exact autodiff rather than a bumped approximation.

References:
    Brigo & Mercurio (2006), *Interest Rate Models*, §3.3.
    Hull & White (1990), "Pricing Interest-Rate-Derivative Securities".
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import optimistix
from jaxtyping import Float
from jax import Array

from valax.calibration.transforms import (
    TransformSpec,
    model_to_unconstrained,
    positive,
    unconstrained_to_model,
)
from valax.curves.discount import DiscountCurve
from valax.instruments.rates import Swaption
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.analytic.swaptions import (
    swaption_price_bachelier,
    swaption_price_black76,
)

HULL_WHITE_TRANSFORMS: dict[str, TransformSpec] = {
    "mean_reversion": positive(),
    "volatility": positive(),
}


def swaption_prices_from_vols(
    swaptions: Sequence[Swaption],
    curve: DiscountCurve,
    market_vols: Float[Array, " n_quotes"],
    normal: bool = False,
) -> Float[Array, " n_quotes"]:
    """Convert quoted swaption volatilities to prices.

    Calibration targets prices, but desks quote volatilities.  This helper
    applies the existing Black-76 / Bachelier pricers so the vol convention
    lives in exactly one place.

    Args:
        swaptions: Swaption contracts, one per quote.
        curve: Discount curve used for the annuity and forward swap rate.
        market_vols: Quoted volatility for each swaption.
        normal: ``True`` for Bachelier (normal) quotes, ``False`` for
            Black-76 (lognormal).

    Returns:
        Market price of each swaption.
    """
    pricer = swaption_price_bachelier if normal else swaption_price_black76
    return jnp.stack(
        [pricer(sw, curve, market_vols[i]) for i, sw in enumerate(swaptions)]
    )


def hw_swaption_prices(
    model: HullWhiteModel,
    swaptions: Sequence[Swaption],
) -> Float[Array, " n_quotes"]:
    """Hull-White model price for each swaption on the surface.

    The contracts have heterogeneous fixed-leg lengths, so this is a Python
    loop over a static sequence rather than a ``vmap``.  Surfaces are small
    (tens of quotes) and each price is closed-form, so the unrolled graph is
    cheap.

    Args:
        model: Hull-White model.
        swaptions: Swaption contracts, one per quote.

    Returns:
        Model price of each swaption.
    """
    return jnp.stack([hw_swaption_price(sw, model) for sw in swaptions])


def _hw_price_residuals(
    raw_params: dict[str, Float[Array, ""]],
    args: tuple,
) -> Float[Array, " n_quotes"]:
    """Weighted price residuals, in the optimistix ``(y, args)`` form.

    Residuals are scaled by the market price by default.  Swaption prices across
    an expiry/tenor grid span more than an order of magnitude, so absolute
    residuals would let the longest-dated quotes dominate the fit and would
    leave the Jacobian badly conditioned.
    """
    transforms, template, swaptions, market_prices, weights, scale = args
    model = unconstrained_to_model(raw_params, transforms, template)
    return weights * (hw_swaption_prices(model, swaptions) - market_prices) / scale


def _hw_sse(
    raw_params: dict[str, Float[Array, ""]],
    args: tuple,
) -> Float[Array, ""]:
    """Scalar objective for gradient-based minimisers."""
    return jnp.sum(_hw_price_residuals(raw_params, args) ** 2)


def calibrate_hull_white(
    swaptions: Sequence[Swaption],
    market_prices: Float[Array, " n_quotes"],
    curve: DiscountCurve,
    initial_guess: HullWhiteModel | None = None,
    fixed_mean_reversion: Float[Array, ""] | None = None,
    weights: Float[Array, " n_quotes"] | None = None,
    relative: bool = True,
    solver: str = "bfgs",
    max_steps: int = 256,
) -> tuple[HullWhiteModel, optimistix.Solution]:
    r"""Fit Hull-White :math:`(a, \sigma)` to a swaption surface.

    Args:
        swaptions: Swaption contracts, one per market quote.
        market_prices: Target price for each swaption.  Use
            :func:`swaption_prices_from_vols` to convert quoted volatilities.
        curve: Initial discount curve; Hull-White is exact-fitted to it, so it
            is *not* a calibration degree of freedom.
        initial_guess: Starting model.  Defaults to
            :math:`a = 0.05`, :math:`\sigma = 0.01`.
        fixed_mean_reversion: If given, :math:`a` is held at this value and
            only :math:`\sigma` is fitted.  Recommended -- see the module
            docstring on parameter correlation.
        weights: Per-quote residual weights.  Default: uniform.
        relative: Divide each residual by its market price (default).  Prices
            across an expiry/tenor grid span an order of magnitude, so absolute
            residuals would let long-dated quotes dominate.
        solver: ``"bfgs"`` (default) or ``"gauss_newton"``.

            ``"bfgs"`` minimises the sum of squares with a line search, so it
            is *damped* and converges from essentially any starting point.
            ``"gauss_newton"`` converges quadratically near the solution but is
            undamped: from a distant start it can overshoot into a region where
            the transformed parameters overflow, producing a non-finite step.
            Prefer it only with a good initial guess.

            ``"levenberg_marquardt"`` -- the usual damped least-squares choice
            -- is deliberately **not** offered.  Unlike
            :func:`valax.calibration.calibrate_sabr`, this residual closes over
            a sequence of instrument pytrees, and ``optimistix`` 0.1.0's
            Levenberg-Marquardt raises ``List arity mismatch`` when it
            recombines the Jacobian's static/dynamic partition in that case.
        max_steps: Maximum optimiser iterations.

    Returns:
        Tuple of the fitted model and the ``optimistix`` solution carrying
        convergence diagnostics.

    Raises:
        ValueError: If ``solver`` is not recognised, or if the number of
            quotes does not match the number of swaptions.
    """
    if len(swaptions) != market_prices.shape[0]:
        raise ValueError(
            f"Got {len(swaptions)} swaptions but "
            f"{market_prices.shape[0]} market prices."
        )

    if initial_guess is None:
        initial_guess = HullWhiteModel(
            mean_reversion=jnp.asarray(0.05),
            volatility=jnp.asarray(0.01),
            initial_curve=curve,
        )
    else:
        initial_guess = eqx.tree_at(
            lambda m: m.initial_curve, initial_guess, curve
        )

    if fixed_mean_reversion is not None:
        initial_guess = eqx.tree_at(
            lambda m: m.mean_reversion,
            initial_guess,
            jnp.asarray(fixed_mean_reversion),
        )
        transforms = {"volatility": positive()}
    else:
        transforms = dict(HULL_WHITE_TRANSFORMS)

    if weights is None:
        weights = jnp.ones_like(market_prices)

    if relative:
        # Guard against a zero quote (a worthless deep-OTM swaption).
        scale = jnp.maximum(jnp.abs(market_prices), 1e-8)
    else:
        scale = jnp.ones_like(market_prices)

    y0 = model_to_unconstrained(initial_guess, transforms)
    args = (
        transforms, initial_guess, tuple(swaptions), market_prices, weights, scale,
    )

    if solver == "bfgs":
        opt = optimistix.BFGS(rtol=1e-12, atol=1e-12)
        sol = optimistix.minimise(
            _hw_sse, opt, y0, args=args, max_steps=max_steps, throw=False,
        )
    elif solver == "gauss_newton":
        opt = optimistix.GaussNewton(rtol=1e-10, atol=1e-10)
        sol = optimistix.least_squares(
            _hw_price_residuals, opt, y0, args=args, max_steps=max_steps,
            throw=False,
        )
    else:
        raise ValueError(f"Unknown solver: {solver!r}")

    fitted = unconstrained_to_model(sol.value, transforms, initial_guess)
    return fitted, sol
