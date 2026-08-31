r"""G2++ calibration to a European swaption surface.

Unlike Hull-White's two parameters, G2++ has five free numbers -- the two
mean-reversion speeds :math:`a, b`, the two factor volatilities
:math:`\sigma, \eta`, and the correlation :math:`\rho` -- because the shift
:math:`\varphi(t)` is pinned by the exact-fit condition to the initial curve.
That extra freedom is exactly what buys the **decorrelation** a one-factor
model cannot produce, so G2++ is typically calibrated jointly to an ATM /
co-terminal swaption surface (and, when they are quoted, to
decorrelation-sensitive instruments such as CMS-spread options).

Model prices come from the Brigo-Mercurio semi-analytic integral
(:func:`valax.pricing.analytic.g2pp_swaptions.g2pp_swaption_price`), which is
implicitly differentiable, so the least-squares Jacobian is exact autodiff.

The default solver is a line-searched ``optimistix.BFGS`` minimiser of the sum
of squares.  As with :func:`valax.calibration.calibrate_hull_white`, the
residual closes over a *sequence* of instrument pytrees, which trips
``optimistix`` 0.1.0's Levenberg-Marquardt (``List arity mismatch``); BFGS is
damped and robust from a distant start, so it is preferred.

References:
    Brigo & Mercurio (2006), *Interest Rate Models*, ch. 4 (§4.2).
"""

from __future__ import annotations

from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
import optimistix
from jaxtyping import Float
from jax import Array

from valax.calibration.transforms import (
    G2PP_TRANSFORMS,
    TransformSpec,
    model_to_unconstrained,
    unconstrained_to_model,
)
from valax.curves.discount import DiscountCurve
from valax.instruments.rates import Swaption
from valax.models.g2pp import G2PPModel
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price


def g2pp_swaption_prices(
    model: G2PPModel,
    swaptions: Sequence[Swaption],
) -> Float[Array, " n_quotes"]:
    """G2++ model price for each swaption on the surface.

    The contracts have heterogeneous fixed-leg lengths, so this is a Python
    loop over a static sequence rather than a ``vmap``.  Surfaces are small and
    each price is a cheap 1-D quadrature, so the unrolled graph is fine.

    Args:
        model: G2++ model.
        swaptions: Swaption contracts, one per quote.

    Returns:
        Model price of each swaption.
    """
    return jnp.stack([g2pp_swaption_price(sw, model) for sw in swaptions])


def _g2pp_price_residuals(
    raw_params: dict[str, Float[Array, ""]],
    args: tuple,
) -> Float[Array, " n_quotes"]:
    """Weighted price residuals, in the optimistix ``(y, args)`` form.

    Residuals are scaled by the market price by default: swaption prices across
    an expiry/tenor grid span more than an order of magnitude, so absolute
    residuals would let the longest-dated quotes dominate the fit.
    """
    transforms, template, swaptions, market_prices, weights, scale = args
    model = unconstrained_to_model(raw_params, transforms, template)
    return weights * (g2pp_swaption_prices(model, swaptions) - market_prices) / scale


def _g2pp_sse(
    raw_params: dict[str, Float[Array, ""]],
    args: tuple,
) -> Float[Array, ""]:
    """Scalar objective for gradient-based minimisers."""
    return jnp.sum(_g2pp_price_residuals(raw_params, args) ** 2)


def calibrate_g2pp(
    swaptions: Sequence[Swaption],
    market_prices: Float[Array, " n_quotes"],
    curve: DiscountCurve,
    initial_guess: G2PPModel | None = None,
    fixed_params: dict[str, Float[Array, ""]] | None = None,
    weights: Float[Array, " n_quotes"] | None = None,
    relative: bool = True,
    solver: str = "bfgs",
    max_steps: int = 512,
) -> tuple[G2PPModel, optimistix.Solution]:
    r"""Fit G2++ :math:`(a, b, \sigma, \eta, \rho)` to a swaption surface.

    Args:
        swaptions: Swaption contracts, one per market quote.
        market_prices: Target price for each swaption.  Use
            :func:`valax.calibration.swaption_prices_from_vols` to convert
            quoted volatilities.
        curve: Initial discount curve; G2++ is exact-fitted to it, so it is
            *not* a calibration degree of freedom.
        initial_guess: Starting model.  Defaults to
            :math:`a = 0.5, b = 0.1, \sigma = 0.01, \eta = 0.008,
            \rho = -0.7` (well-separated mean reversions to avoid the
            :math:`a = b` degeneracy).
        fixed_params: Optional mapping of parameter name to a value to hold
            fixed (e.g. ``{"mean_reversion_x": 0.5}``).  A common workflow is
            to pin the two mean reversions and fit only the vols and
            correlation, since :math:`(a, b)` are weakly identified by the ATM
            surface alone.
        weights: Per-quote residual weights.  Default: uniform.
        relative: Divide each residual by its market price (default).
        solver: ``"bfgs"`` (default) or ``"gauss_newton"``.

            ``"bfgs"`` minimises the sum of squares with a line search, so it
            is damped and converges from essentially any starting point.
            ``"gauss_newton"`` converges quadratically near the solution but is
            undamped; prefer it only with a good initial guess.

            ``"levenberg_marquardt"`` is deliberately **not** offered: as in
            :func:`valax.calibration.calibrate_hull_white`, the residual closes
            over a sequence of instrument pytrees, which trips ``optimistix``
            0.1.0's Levenberg-Marquardt with a ``List arity mismatch``.
        max_steps: Maximum optimiser iterations.

    Returns:
        Tuple of the fitted model and the ``optimistix`` solution carrying
        convergence diagnostics.

    Raises:
        ValueError: If ``solver`` is not recognised, if a ``fixed_params`` key
            is not a G2++ parameter, or if the number of quotes does not match
            the number of swaptions.
    """
    if len(swaptions) != market_prices.shape[0]:
        raise ValueError(
            f"Got {len(swaptions)} swaptions but "
            f"{market_prices.shape[0]} market prices."
        )

    if initial_guess is None:
        initial_guess = G2PPModel(
            mean_reversion_x=jnp.asarray(0.5),
            mean_reversion_y=jnp.asarray(0.1),
            volatility_x=jnp.asarray(0.01),
            volatility_y=jnp.asarray(0.008),
            correlation=jnp.asarray(-0.7),
            initial_curve=curve,
        )
    else:
        initial_guess = eqx.tree_at(
            lambda m: m.initial_curve, initial_guess, curve
        )

    transforms: dict[str, TransformSpec] = dict(G2PP_TRANSFORMS)
    if fixed_params is not None:
        for name, value in fixed_params.items():
            if name not in G2PP_TRANSFORMS:
                raise ValueError(f"Unknown G2++ parameter: {name!r}")
            initial_guess = eqx.tree_at(
                lambda m, n=name: getattr(m, n),
                initial_guess,
                jnp.asarray(value),
            )
            del transforms[name]

    if weights is None:
        weights = jnp.ones_like(market_prices)

    if relative:
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
            _g2pp_sse, opt, y0, args=args, max_steps=max_steps, throw=False,
        )
    elif solver == "gauss_newton":
        opt = optimistix.GaussNewton(rtol=1e-10, atol=1e-10)
        sol = optimistix.least_squares(
            _g2pp_price_residuals, opt, y0, args=args, max_steps=max_steps,
            throw=False,
        )
    else:
        raise ValueError(f"Unknown solver: {solver!r}")

    fitted = unconstrained_to_model(sol.value, transforms, initial_guess)
    return fitted, sol
