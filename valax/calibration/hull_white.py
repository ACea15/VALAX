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

import dataclasses
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.calibration.basket import basket_fit_level
from valax.calibration.level import LevelInputs
from valax.calibration.result import CLOSED_FORM_PROVENANCE, CalibrationResult
from valax.calibration.settings import SolverSettings
from valax.calibration.transforms import TransformSpec, positive
from valax.curves.discount import DiscountCurve
from valax.instruments.rates import Swaption
from valax.models.hull_white import HullWhiteModel
from valax.pricing.analytic.hull_white_swaptions import hw_swaption_price
from valax.pricing.multicurve import with_discount_curve
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
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, " n_quotes"]:
    """Convert quoted swaption volatilities to prices.

    Calibration targets prices, but desks quote volatilities.  This helper
    applies the existing Black-76 / Bachelier pricers so the vol convention
    lives in exactly one place.

    With a distinct ``forward_curve`` the forward swap rate is projected off
    that curve while the annuity is discounted on ``curve`` — so the *same*
    quoted volatility implies a different price than it would single-curve.
    Getting this leg right matters as much as the model leg: a dual-curve model
    fitted to single-curve target prices is inconsistent in the opposite
    direction.

    Args:
        swaptions: Swaption contracts, one per quote.
        curve: Discount curve used for the annuity and, when no projection
            curve is given, the forward swap rate.
        market_vols: Quoted volatility for each swaption.
        normal: ``True`` for Bachelier (normal) quotes, ``False`` for
            Black-76 (lognormal).
        forward_curve: Optional projection curve for the underlying swaps'
            floating legs.  ``None`` is the single-curve convention.

    Returns:
        Market price of each swaption.
    """
    pricer = swaption_price_bachelier if normal else swaption_price_black76
    return jnp.stack(
        [
            pricer(sw, curve, market_vols[i], forward_curve)
            for i, sw in enumerate(swaptions)
        ]
    )


def hw_swaption_prices(
    model: HullWhiteModel,
    swaptions: Sequence[Swaption],
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, " n_quotes"]:
    """Hull-White model price for each swaption on the surface.

    The contracts have heterogeneous fixed-leg lengths, so this is a Python
    loop over a static sequence rather than a ``vmap``.  Surfaces are small
    (tens of quotes) and each price is closed-form, so the unrolled graph is
    cheap.

    Args:
        model: Hull-White model; its ``initial_curve`` is the discount curve.
        swaptions: Swaption contracts, one per quote.
        forward_curve: Optional projection curve, applied through the frozen
            tenor-basis overlay.  ``None`` is single-curve.

    Returns:
        Model price of each swaption.
    """
    return jnp.stack(
        [hw_swaption_price(sw, model, forward_curve) for sw in swaptions]
    )


def calibrate_hull_white(
    swaptions: Sequence[Swaption],
    market_prices: Float[Array, " n_quotes"],
    curve: DiscountCurve,
    *,
    initial_guess: HullWhiteModel | None = None,
    fixed_mean_reversion: Float[Array, ""] | None = None,
    weights: Float[Array, " n_quotes"] | None = None,
    relative: bool = True,
    settings: SolverSettings | None = None,
    forward_curve: DiscountCurve | None = None,
    curve_id: str | None = None,
    name: str = "hull_white",
) -> CalibrationResult:
    r"""Fit Hull-White :math:`(a, \sigma)` to a swaption surface.

    A weighted least-squares level (:func:`~valax.calibration.basket.basket_fit_level`)
    on Jamshidian prices: the quotes are the target prices, the curves are
    the level's upstream input (so its adjoint carries curve risk), and a
    fixed mean reversion is a held-fixed input (its sensitivity is reported).
    The engine is exact, so the provenance is audit metadata and the fitted
    parameters are portable to any engine.

    **Single- and dual-curve.** By default this is the single-curve fit.
    Passing ``forward_curve`` makes ``curve`` the *discount* (e.g. OIS) curve
    and projects the underlying swaps' floating legs off the projection
    curve, through the same frozen tenor-basis overlay the MC/PDE recipes
    use — so the fitted model is consistent with the engine that will later
    price with it.

    .. warning::

       Convert your target vols with the *same* curve pair
       (``swaption_prices_from_vols(..., forward_curve=...)``).  Fitting a
       dual-curve model to single-curve target prices pushes the basis into
       :math:`\sigma` and is exactly the inconsistency this argument exists to
       remove.

    Args:
        swaptions: Swaption contracts, one per market quote.
        market_prices: Target price for each swaption.  Use
            :func:`swaption_prices_from_vols` to convert quoted volatilities.
        curve: Initial discount curve; Hull-White is exact-fitted to it, so it
            is *not* a calibration degree of freedom.
        initial_guess: Starting model.  Defaults to
            :math:`a = 0.05`, :math:`\sigma = 0.01`.
        fixed_mean_reversion: If given, :math:`a` is held at this value (a
            reported fixed input) and only :math:`\sigma` is fitted. The
            default ``None`` fits both, which is right for a **surface or
            diagonal** basket spanning several expiries *and* tenors — the
            tenor dimension identifies :math:`a`. For a **co-terminal**
            basket (one Bermudan), :math:`a` and :math:`\sigma` are strongly
            correlated (see the module docstring), so fix :math:`a` — as
            :func:`~valax.calibration.nodes.swaption_basket_node` does by
            ``fixed_fields`` in the per-trade examples.
        weights: Per-quote residual weights.  Default: uniform.
        relative: Divide each residual by its market price (default).  Prices
            across an expiry/tenor grid span an order of magnitude, so absolute
            residuals would let long-dated quotes dominate.
        settings: Solver settings; defaults to
            :data:`~valax.calibration.basket.BASKET_SETTINGS`
            (line-searched BFGS to 1e-12). Any least-squares method is
            accepted (e.g. ``SolverSettings(method="gauss_newton")`` from a
            good initial guess).
        forward_curve: Optional projection curve for the underlying swaps'
            floating legs.  ``None`` (the default) is the single-curve fit.
        curve_id: Optional identifier of the discount curve in the
            :class:`~valax.core.market_state.MarketState` it came from,
            recorded on the fitted model.
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the fitted
        :class:`~valax.models.hull_white.HullWhiteModel` is ``result.value``
        — **curve-free** (``initial_curve=None``): price it through
        :func:`valax.price`, which binds the curve from the market, or bind
        one explicitly with :func:`valax.core.binding.bind_curves`.

    Raises:
        ValueError: If the number of quotes does not match the number of
            swaptions, or ``settings.method`` is not a least-squares method.
    """
    swaptions = tuple(swaptions)
    if len(swaptions) != market_prices.shape[0]:
        raise ValueError(
            f"Got {len(swaptions)} swaptions but "
            f"{market_prices.shape[0]} market prices."
        )
    if initial_guess is None:
        initial_guess = HullWhiteModel(
            mean_reversion=jnp.asarray(0.05),
            volatility=jnp.asarray(0.01),
        )
    # The fitted model is curve-free: the curve is market data, bound only
    # for pricing. ``curve_id`` is static, hence ``dataclasses.replace``.
    initial_guess = dataclasses.replace(
        initial_guess, initial_curve=None,
        curve_id=initial_guess.curve_id if curve_id is None else curve_id,
    )
    fixed = (
        None if fixed_mean_reversion is None
        else {"mean_reversion": fixed_mean_reversion}
    )

    def bind(model: HullWhiteModel, inp: LevelInputs) -> HullWhiteModel:
        return with_discount_curve(model, inp.upstream[0])

    def prices(model: HullWhiteModel, inp: LevelInputs) -> Float[Array, " n"]:
        # The basket travels in ``inputs.data`` (not a closure), so a result
        # returned from jit / vmap holds no stale tracer.
        return hw_swaption_prices(model, inp.data, inp.upstream[1])

    return basket_fit_level(
        name,
        quotes=market_prices,
        upstream=(curve, forward_curve),
        initial_guess=initial_guess,
        transforms=HULL_WHITE_TRANSFORMS,
        prices_fn=prices,
        bind_fn=bind,
        fixed_params=fixed,
        weights=weights,
        relative=relative,
        settings=settings,
        provenance=JAMSHIDIAN_PROVENANCE,
        model_label="Hull-White",
        data=swaptions,
    )


#: Alias of :data:`~valax.calibration.result.CLOSED_FORM_PROVENANCE`, kept for
#: the Hull-White (Jamshidian) call sites.
JAMSHIDIAN_PROVENANCE = CLOSED_FORM_PROVENANCE
