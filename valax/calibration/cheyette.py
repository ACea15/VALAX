r"""Cheyette (quasi-Gaussian) calibration on top of the PDE.

The Cheyette model is calibrated *through its finite-difference pricer* rather
than an approximate-analytic swaption formula, because the PDE covers the full
range of calibration instruments — caps, European **and** Bermudan swaptions,
callables — on one consistent numerical engine. Model prices come from the
frozen-``y`` recipes registered on
:func:`~valax.pricing.pde.pde_price_dispatch` (the production engine), or, for
one factor, from the exact-``y`` solver
:func:`~valax.pricing.pde.cheyette.cheyette_exact_price` (bias-free). Both are
cleanly reverse-mode differentiable — the backward sweep is ``lineax``
tridiagonal solves inside a ``lax.scan`` with nothing detached from autodiff —
so the least-squares Jacobian is exact autodiff through the PDE; no
finite-difference Greeks and no implicit-function-theorem adjoint are needed.

Identification
--------------
**One factor** :math:`(\kappa, \sigma, \beta)`. Fix the mean reversion
:math:`\kappa` (weakly identified by the ATM level alone); :math:`\sigma` then
sets the ATM volatility level and :math:`\beta` the skew. Calibrating
:math:`\beta` requires a **strike slice** of caplets or swaptions — the skew is
invisible at a single (ATM) strike.

**Two factors** :math:`(\kappa_1, \kappa_2, \sigma_1, \sigma_2, \beta_1,
\beta_2, \rho)`. Fix :math:`\kappa_1, \kappa_2` (well separated to avoid the
:math:`\kappa_1 = \kappa_2` degeneracy) and start :math:`\rho` from a prior; a
decorrelation-sensitive instrument (e.g. a CMS-spread option) is needed to pin
:math:`\rho` beyond the ATM surface.

Solver and provenance
---------------------
The default is a line-searched BFGS (:data:`CHEYETTE_SETTINGS`), damped and
robust from a distant start. Because the PDE sits inside the optimiser loop,
use a moderate grid and a small instrument set. Each result records its engine
(:func:`cheyette_provenance`): frozen-``y`` parameters are engine-bound.

References:
    Andersen, L. & Piterbarg, V. (2010), *Interest Rate Modeling*, ch. 13.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.calibration.basket import basket_fit_level
from valax.calibration.level import LevelInputs
from valax.calibration.result import (
    CalibrationProvenance,
    CalibrationResult,
    engine_provenance,
)
from valax.calibration.settings import SolverSettings
from valax.calibration.transforms import (
    CHEYETTE_2F_TRANSFORMS,
    CHEYETTE_TRANSFORMS,
    TransformSpec,
)
from valax.curves.discount import DiscountCurve
from valax.models.cheyette import CheyetteModel, CheyetteModel2F
from valax.pricing.multicurve import require_single_curve, with_discount_curve
from valax.pricing.pde import PDEConfig, PDEConfig2D, Scheme, pde_price_dispatch
from valax.pricing.pde.cheyette import cheyette_exact_price

# Moderate default meshes for PDE-in-the-loop calibration: fine enough for a
# ~1e-3 relative fit, coarse enough that the optimiser stays cheap.
_DEFAULT_CONFIG_FROZEN_1F = PDEConfig(
    n_spot=121, n_time=60, spot_range=6.0, scheme=Scheme.CRANK_NICOLSON
)
_DEFAULT_CONFIG_EXACT_1F = PDEConfig2D(
    n_x=81, n_y=61, n_time=72, x_range=6.0, y_max=3.0
)
_DEFAULT_CONFIG_FROZEN_2F = PDEConfig2D(
    n_x=81, n_y=81, n_time=50, x_range=6.0, scheme=Scheme.CRAIG_SNEYD
)


def cheyette_pde_prices(
    model: CheyetteModel | CheyetteModel2F,
    instruments: Sequence[Any],
    engine: str,
    config: PDEConfig | PDEConfig2D,
) -> Float[Array, " n_quotes"]:
    """PDE model price for each calibration instrument.

    Contracts have heterogeneous schedules, so this is a Python loop over a
    static sequence (unrolled at trace time) rather than a ``vmap``. The grid /
    operator are rebuilt per instrument but the mesh is reused across optimiser
    iterations because ``config`` is fixed.

    Args:
        model: One- or two-factor Cheyette model.
        instruments: Calibration instruments, one per quote.
        engine: ``"frozen"`` (frozen-``y`` recipes via ``pde_price_dispatch``)
            or ``"exact"`` (one-factor exact-``y`` solver).
        config: PDE grid configuration matching the engine.

    Returns:
        Model price of each instrument.
    """
    if engine == "exact":
        return jnp.stack(
            [cheyette_exact_price(inst, model, config).price for inst in instruments]
        )
    return jnp.stack(
        [pde_price_dispatch(inst, model, config).price for inst in instruments]
    )


def cheyette_provenance(
    engine: str, config: PDEConfig | PDEConfig2D
) -> CalibrationProvenance:
    """Provenance of a Cheyette fit on ``engine``.

    Frozen-``y`` replaces the stochastic accumulator by its deterministic
    limit — a **modelling** bias that does not converge away under grid
    refinement — so its parameters are only valid on that engine
    (:func:`~valax.calibration.result.check_engine` refuses another).
    Exact-``y`` carries only discretisation bias.

    Args:
        engine: ``"frozen"`` or ``"exact"``.
        config: The PDE grid the fit was priced on.

    Returns:
        The :class:`~valax.calibration.result.CalibrationProvenance`.
    """
    if engine == "exact":
        return CalibrationProvenance(
            method="cheyette_exact_y", engine_kind="discretised", config=config
        )
    return CalibrationProvenance(
        method="pde", engine_kind="modelling_bias", config=config
    )


@engine_provenance.register(CheyetteModel)
@engine_provenance.register(CheyetteModel2F)
def _cheyette_engine_provenance(
    model: CheyetteModel | CheyetteModel2F, method: str, config: Any = None
) -> CalibrationProvenance:
    """Cheyette through ``price``: the PDE recipes are the frozen-``y`` engine.

    Every Cheyette PDE recipe registered with ``price`` is frozen-``y``, so a
    basket priced with ``method="pde"`` is engine-bound
    (:func:`cheyette_provenance`). Monte Carlo simulates the exact ``y`` and is
    only discretised.

    Args:
        model: The Cheyette template.
        method: The concrete pricing method of the basket.
        config: The basket's pricing config.

    Returns:
        The provenance to record.
    """
    del model
    if method == "pde":
        return cheyette_provenance("frozen", config)
    return CalibrationProvenance(
        method=method, engine_kind="discretised", config=config
    )


def _cheyette_fit(
    name: str,
    instruments: tuple,
    market_prices: Float[Array, " n_quotes"],
    curve: DiscountCurve,
    initial_guess: Any,
    transforms: dict[str, TransformSpec],
    fixed_params: dict | None,
    weights: Float[Array, " n_quotes"] | None,
    relative: bool,
    settings: SolverSettings | None,
    engine: str,
    config: PDEConfig | PDEConfig2D,
    model_label: str,
) -> CalibrationResult:
    """Shared 1F / 2F body: a basket fit on the Cheyette PDE."""

    initial_guess = dataclasses.replace(initial_guess, initial_curve=None)

    def bind(model: Any, inp: LevelInputs) -> Any:
        return with_discount_curve(model, inp.upstream)

    def prices(model: Any, inp: LevelInputs) -> Float[Array, " n"]:
        # Basket from ``inputs.data``, not a closure (tracer-free results).
        return cheyette_pde_prices(model, inp.data, engine, config)

    return basket_fit_level(
        name,
        quotes=market_prices,
        upstream=curve,
        initial_guess=initial_guess,
        transforms=transforms,
        prices_fn=prices,
        bind_fn=bind,
        fixed_params=fixed_params,
        weights=weights,
        relative=relative,
        settings=CHEYETTE_SETTINGS if settings is None else settings,
        provenance=cheyette_provenance(engine, config),
        data=tuple(instruments),
        model_label=model_label,
    )


#: Default settings of a PDE-in-the-loop Cheyette fit: BFGS to 1e-10
#: (looser than the closed-form fits: the PDE has a discretisation floor).
CHEYETTE_SETTINGS = SolverSettings(
    method="bfgs", rtol=1e-10, atol=1e-10, max_steps=256
)


def calibrate_cheyette(
    instruments: Sequence[Any],
    market_prices: Float[Array, " n_quotes"],
    curve: DiscountCurve,
    *,
    initial_guess: CheyetteModel | None = None,
    fixed_params: dict[str, Float[Array, ""]] | None = None,
    weights: Float[Array, " n_quotes"] | None = None,
    relative: bool = True,
    settings: SolverSettings | None = None,
    engine: str = "frozen",
    config: PDEConfig | PDEConfig2D | None = None,
    forward_curve: DiscountCurve | None = None,
    name: str = "cheyette",
) -> CalibrationResult:
    r"""Fit one-factor Cheyette :math:`(\kappa, \sigma, \beta)` through the PDE.

    A weighted least-squares level
    (:func:`~valax.calibration.basket.basket_fit_level`) priced on the PDE.
    The result records the engine as provenance: a ``"frozen"`` fit carries
    a modelling bias, so its parameters are engine-bound
    (:func:`~valax.calibration.result.check_engine` refuses to price them on
    another engine or grid); an ``"exact"`` fit only a discretisation bias.

    **Single-curve only.** The Cheyette PDE engine has no discount/projection
    split (unlike the Hull-White and G2++ recipes), so ``curve`` both discounts
    and projects.  Passing a distinct ``forward_curve`` *raises* rather than
    being silently dropped — a dual-curve fit that quietly ignored the
    projection curve would be a wrong calibration with no signal.

    Args:
        instruments: Calibration instruments, one per market quote (e.g.
            swaptions across a strike slice, so the skew is identified).
        market_prices: Target PDE-consistent price for each instrument.
        curve: Initial discount curve; Cheyette is exact-fitted to it, so it is
            *not* a calibration degree of freedom.
        initial_guess: Starting model. Defaults to
            :math:`\kappa = 0.05, \sigma = 0.01, \beta = 0`.
        fixed_params: Optional mapping of parameter name to a value to hold
            fixed (e.g. ``{"mean_reversion": 0.05}``). The recommended workflow
            pins :math:`\kappa` and fits :math:`(\sigma, \beta)`.
        weights: Per-quote residual weights. Default: uniform.
        relative: Divide each residual by its market price (default).
        settings: Solver settings; defaults to :data:`CHEYETTE_SETTINGS`
            (BFGS to 1e-10).
        engine: ``"frozen"`` (default; the fast frozen-``y`` production PDE) or
            ``"exact"`` (the bias-free exact-``y`` 2-D PDE — more expensive per
            solve, but removes the frozen-``y`` skew bias).
        config: PDE grid configuration. Defaults to a moderate mesh matching the
            chosen engine (:class:`~valax.pricing.pde.PDEConfig` for ``frozen``,
            :class:`~valax.pricing.pde.PDEConfig2D` for ``exact``).
        forward_curve: Must be ``None`` or ``curve`` (single-curve only).
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the fitted
        :class:`~valax.models.cheyette.CheyetteModel` is ``result.value``.

    Raises:
        ValueError: If ``engine`` is not recognised, if a ``fixed_params`` key
            is not a Cheyette parameter, if the number of quotes does not
            match the number of instruments, or if a distinct
            ``forward_curve`` is passed.
    """
    if len(instruments) != market_prices.shape[0]:
        raise ValueError(
            f"Got {len(instruments)} instruments but "
            f"{market_prices.shape[0]} market prices."
        )
    if engine not in ("frozen", "exact"):
        raise ValueError(f"Unknown engine: {engine!r} (use 'frozen' or 'exact').")
    require_single_curve(curve, forward_curve, "Cheyette calibration")
    if config is None:
        config = (
            _DEFAULT_CONFIG_EXACT_1F if engine == "exact"
            else _DEFAULT_CONFIG_FROZEN_1F
        )
    if initial_guess is None:
        initial_guess = CheyetteModel(
            mean_reversion=jnp.asarray(0.05),
            volatility=jnp.asarray(0.01),
            skew=jnp.asarray(0.0),
        )
    return _cheyette_fit(
        name, tuple(instruments), market_prices, curve, initial_guess,
        CHEYETTE_TRANSFORMS, fixed_params, weights, relative, settings,
        engine, config, "Cheyette",
    )


def calibrate_cheyette_2f(
    instruments: Sequence[Any],
    market_prices: Float[Array, " n_quotes"],
    curve: DiscountCurve,
    *,
    initial_guess: CheyetteModel2F | None = None,
    fixed_params: dict[str, Float[Array, ""]] | None = None,
    weights: Float[Array, " n_quotes"] | None = None,
    relative: bool = True,
    settings: SolverSettings | None = None,
    config: PDEConfig2D | None = None,
    forward_curve: DiscountCurve | None = None,
    name: str = "cheyette_2f",
) -> CalibrationResult:
    r"""Fit two-factor Cheyette through the frozen-``y`` 2-D ADI PDE.

    Two factors are always calibrated on the **frozen-``y``** engine — the
    exact-``y`` two-factor PDE is 5-D and not tractable in a calibration loop
    (see ``docs/research-ideas.md``) — so the result's provenance is always
    engine-bound (modelling bias).

    **Single-curve only**, for the same reason as
    :func:`calibrate_cheyette`: a distinct ``forward_curve`` raises rather than
    being silently ignored.

    Args:
        instruments: Calibration instruments, one per market quote.
        market_prices: Target PDE-consistent price for each instrument.
        curve: Initial discount curve; exact-fitted, not a degree of freedom.
        initial_guess: Starting model. Defaults to
            :math:`\kappa_1 = 0.5, \kappa_2 = 0.1, \sigma_1 = 0.01,
            \sigma_2 = 0.008, \beta_1 = \beta_2 = 0, \rho = -0.7` (well-separated
            mean reversions avoid the :math:`\kappa_1 = \kappa_2` degeneracy).
        fixed_params: Optional mapping of parameter name to a value to hold
            fixed. The recommended workflow pins :math:`\kappa_1, \kappa_2`.
        weights: Per-quote residual weights. Default: uniform.
        relative: Divide each residual by its market price (default).
        settings: Solver settings; defaults to :data:`CHEYETTE_SETTINGS`.
        config: 2-D PDE grid configuration. Defaults to a moderate Craig-Sneyd
            mesh.
        forward_curve: Must be ``None`` or ``curve`` (single-curve only).
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the fitted
        :class:`~valax.models.cheyette.CheyetteModel2F` is ``result.value``.

    Raises:
        ValueError: If a ``fixed_params`` key is not a Cheyette parameter, if
            the number of quotes does not match the number of instruments, or
            if a distinct ``forward_curve`` is passed.
    """
    if len(instruments) != market_prices.shape[0]:
        raise ValueError(
            f"Got {len(instruments)} instruments but "
            f"{market_prices.shape[0]} market prices."
        )
    require_single_curve(curve, forward_curve, "Cheyette 2F calibration")
    if config is None:
        config = _DEFAULT_CONFIG_FROZEN_2F
    if initial_guess is None:
        initial_guess = CheyetteModel2F(
            mean_reversion_1=jnp.asarray(0.5),
            mean_reversion_2=jnp.asarray(0.1),
            volatility_1=jnp.asarray(0.01),
            volatility_2=jnp.asarray(0.008),
            skew_1=jnp.asarray(0.0),
            skew_2=jnp.asarray(0.0),
            correlation=jnp.asarray(-0.7),
        )
    return _cheyette_fit(
        name, tuple(instruments), market_prices, curve, initial_guess,
        CHEYETTE_2F_TRANSFORMS, fixed_params, weights, relative, settings,
        "frozen", config, "Cheyette",
    )
