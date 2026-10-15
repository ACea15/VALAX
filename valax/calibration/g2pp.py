r"""G2++ calibration to a European swaption surface.

Unlike Hull-White's two parameters, G2++ has five free numbers -- the two
mean-reversion speeds :math:`a, b`, the two factor volatilities
:math:`\sigma, \eta` (fitted as :math:`\sigma` and the ratio
:math:`r = \eta/\sigma`), and the correlation :math:`\rho` -- because the shift
:math:`\varphi(t)` is pinned by the exact-fit condition to the initial curve.
That extra freedom is exactly what buys the **decorrelation** a one-factor
model cannot produce, so G2++ is typically calibrated jointly to an ATM /
co-terminal swaption surface (and, when they are quoted, to
decorrelation-sensitive instruments such as CMS-spread options).

Model prices come from the Brigo-Mercurio semi-analytic integral
(:func:`valax.pricing.analytic.g2pp_swaptions.g2pp_swaption_price`), which is
implicitly differentiable, so the least-squares Jacobian is exact autodiff.

The default solver is a line-searched BFGS
(:data:`~valax.calibration.basket.BASKET_SETTINGS`): damped and robust from a
distant start. Any least-squares method can be selected through
``settings``.

References:
    Brigo & Mercurio (2006), *Interest Rate Models*, ch. 4 (§4.2).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.calibration.basket import BASKET_SETTINGS, basket_fit_level
from valax.calibration.level import LevelInputs
from valax.calibration.result import CLOSED_FORM_PROVENANCE, CalibrationResult
from valax.calibration.settings import SolverSettings
from valax.calibration.transforms import G2PP_TRANSFORMS
from valax.pricing.multicurve import with_discount_curve
from valax.curves.discount import DiscountCurve
from valax.instruments.rates import Swaption
from valax.models.g2pp import G2PPModel
from valax.models.short_rate import as_buckets
from valax.pricing.analytic.g2pp_swaptions import g2pp_swaption_price

#: G2++ fields that hold one value per volatility bucket.
_BUCKETED = ("volatility_x", "volatility_ratio")


def g2pp_swaption_prices(
    model: G2PPModel,
    swaptions: Sequence[Swaption],
    forward_curve: DiscountCurve | None = None,
) -> Float[Array, " n_quotes"]:
    """G2++ model price for each swaption on the surface.

    The contracts have heterogeneous fixed-leg lengths, so this is a Python
    loop over a static sequence rather than a ``vmap``.  Surfaces are small and
    each price is a cheap 1-D quadrature, so the unrolled graph is fine.

    Args:
        model: G2++ model; its ``initial_curve`` is the discount curve.
        swaptions: Swaption contracts, one per quote.
        forward_curve: Optional projection curve, applied through the frozen
            tenor-basis overlay.  ``None`` is single-curve.

    Returns:
        Model price of each swaption.
    """
    return jnp.stack(
        [g2pp_swaption_price(sw, model, forward_curve) for sw in swaptions]
    )


def calibrate_g2pp(
    swaptions: Sequence[Swaption],
    market_prices: Float[Array, " n_quotes"],
    curve: DiscountCurve,
    *,
    initial_guess: G2PPModel | None = None,
    fixed_params: dict[str, Float[Array, ""]] | None = None,
    weights: Float[Array, " n_quotes"] | None = None,
    relative: bool = True,
    settings: SolverSettings | None = None,
    forward_curve: DiscountCurve | None = None,
    curve_id: str | None = None,
    name: str = "g2pp",
) -> CalibrationResult:
    r"""Fit G2++ :math:`(a, b, \sigma, r, \rho)` to a swaption surface.

    A weighted least-squares level (:func:`~valax.calibration.basket.basket_fit_level`)
    on Brigo-Mercurio prices: the quotes are the target prices, the curves
    the upstream input, and any ``fixed_params`` held-fixed inputs whose
    sensitivities are reported.

    The second factor's volatility is fitted through the **ratio**
    :math:`r = \eta / \sigma` (field ``volatility_ratio``), so
    :math:`\eta = r\sigma` is ``result.value.volatility_y``. The fit is over
    the model's volatility buckets as given by ``initial_guess`` (one bucket,
    a constant model, by default).

    **Single- and dual-curve.** Passing ``forward_curve`` makes ``curve`` the
    *discount* (e.g. OIS) curve and projects the underlying swaps' floating
    legs off the projection curve through the frozen deterministic tenor-basis
    overlay, matching the MC/PDE engines.

    Args:
        swaptions: Swaption contracts, one per market quote.
        market_prices: Target price for each swaption.  Use
            :func:`valax.calibration.swaption_prices_from_vols` to convert
            quoted volatilities.
        curve: Initial discount curve; G2++ is exact-fitted to it, so it is
            *not* a calibration degree of freedom.
        initial_guess: Starting model.  Defaults to
            :math:`a = 0.5, b = 0.1, \sigma = 0.01, r = 0.8` (so
            :math:`\eta = 0.008`), :math:`\rho = -0.7` (well-separated mean
            reversions to avoid the :math:`a = b` degeneracy).
        fixed_params: Optional mapping of parameter name to a value to hold
            fixed (e.g. ``{"mean_reversion_x": 0.5}``; the parameter names
            are ``mean_reversion_x``, ``mean_reversion_y``, ``volatility_x``,
            ``volatility_ratio``, ``correlation``).  A scalar volatility value
            is promoted to one bucket.  A common workflow is to pin the two
            mean reversions and fit only the vols and correlation, since
            :math:`(a, b)` are weakly identified by the ATM surface alone.
        weights: Per-quote residual weights.  Default: uniform.
        relative: Divide each residual by its market price (default).
        settings: Solver settings; defaults to
            :data:`~valax.calibration.basket.BASKET_SETTINGS` (BFGS to 1e-12,
            here with a 512-step budget).
        forward_curve: Optional projection curve for the underlying swaps'
            floating legs.  ``None`` (the default) is the single-curve fit.
            Convert your target vols with the same curve pair.
        curve_id: Optional identifier of the discount curve in the
            :class:`~valax.core.market_state.MarketState` it came from,
            recorded on the fitted model.
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the fitted
        :class:`~valax.models.g2pp.G2PPModel` is ``result.value`` —
        curve-free (``initial_curve=None``; see :func:`calibrate_hull_white`).

    Raises:
        ValueError: If a ``fixed_params`` key is not a G2++ parameter, or the
            number of quotes does not match the number of swaptions.
    """
    swaptions = tuple(swaptions)
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
            volatility_ratio=jnp.asarray(0.8),
            correlation=jnp.asarray(-0.7),
        )
    initial_guess = dataclasses.replace(
        initial_guess, initial_curve=None,
        curve_id=initial_guess.curve_id if curve_id is None else curve_id,
    )
    if settings is None:
        settings = dataclasses.replace(BASKET_SETTINGS, max_steps=512)
    if fixed_params is not None:
        # Volatility fields are bucket arrays: promote a scalar to one bucket.
        fixed_params = {
            k: as_buckets(jnp.asarray(v)) if k in _BUCKETED else v
            for k, v in fixed_params.items()
        }

    def bind(model: G2PPModel, inp: LevelInputs) -> G2PPModel:
        return with_discount_curve(model, inp.upstream[0])

    def prices(model: G2PPModel, inp: LevelInputs) -> Float[Array, " n"]:
        # Basket from ``inputs.data``, not a closure (tracer-free results).
        return g2pp_swaption_prices(model, inp.data, inp.upstream[1])

    return basket_fit_level(
        name,
        quotes=market_prices,
        upstream=(curve, forward_curve),
        initial_guess=initial_guess,
        transforms=G2PP_TRANSFORMS,
        prices_fn=prices,
        bind_fn=bind,
        fixed_params=fixed_params,
        weights=weights,
        relative=relative,
        settings=settings,
        provenance=CLOSED_FORM_PROVENANCE,
        model_label="G2++",
        data=tuple(swaptions),
    )
