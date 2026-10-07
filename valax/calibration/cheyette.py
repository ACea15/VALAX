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

Solver
------
The default is a line-searched ``optimistix.BFGS`` minimiser of the sum of
squares. As with :func:`~valax.calibration.calibrate_g2pp`, the residual closes
over a *sequence* of instrument pytrees, which trips ``optimistix`` 0.1.0's
Levenberg-Marquardt (``List arity mismatch``); BFGS is damped and robust from a
distant start, so it is preferred. Because the PDE sits inside the optimiser
loop, use a moderate grid and a small instrument set.

References:
    Andersen, L. & Piterbarg, V. (2010), *Interest Rate Modeling*, ch. 13.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import optimistix
from jaxtyping import Float
from jax import Array

from valax.calibration.transforms import (
    CHEYETTE_2F_TRANSFORMS,
    CHEYETTE_TRANSFORMS,
    TransformSpec,
    model_to_unconstrained,
    unconstrained_to_model,
)
from valax.curves.discount import DiscountCurve
from valax.models.cheyette import CheyetteModel, CheyetteModel2F
from valax.pricing.multicurve import require_single_curve
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


def _cheyette_price_residuals(
    raw_params: dict[str, Float[Array, ""]],
    args: tuple,
) -> Float[Array, " n_quotes"]:
    """Weighted relative price residuals, in the optimistix ``(y, args)`` form."""
    transforms, template, instruments, market_prices, weights, scale, engine, config = args
    model = unconstrained_to_model(raw_params, transforms, template)
    prices = cheyette_pde_prices(model, instruments, engine, config)
    return weights * (prices - market_prices) / scale


def _cheyette_sse(
    raw_params: dict[str, Float[Array, ""]],
    args: tuple,
) -> Float[Array, ""]:
    """Scalar objective for gradient-based minimisers."""
    return jnp.sum(_cheyette_price_residuals(raw_params, args) ** 2)


def _run_solver(solver: str, y0, args, max_steps: int):
    """Dispatch to the requested optimistix solver (shared 1F / 2F)."""
    if solver == "bfgs":
        opt = optimistix.BFGS(rtol=1e-10, atol=1e-10)
        return optimistix.minimise(
            _cheyette_sse, opt, y0, args=args, max_steps=max_steps, throw=False
        )
    if solver == "gauss_newton":
        opt = optimistix.GaussNewton(rtol=1e-8, atol=1e-8)
        return optimistix.least_squares(
            _cheyette_price_residuals,
            opt,
            y0,
            args=args,
            max_steps=max_steps,
            throw=False,
        )
    raise ValueError(f"Unknown solver: {solver!r}")


def calibrate_cheyette(
    instruments: Sequence[Any],
    market_prices: Float[Array, " n_quotes"],
    curve: DiscountCurve,
    initial_guess: CheyetteModel | None = None,
    fixed_params: dict[str, Float[Array, ""]] | None = None,
    weights: Float[Array, " n_quotes"] | None = None,
    relative: bool = True,
    solver: str = "bfgs",
    engine: str = "frozen",
    config: PDEConfig | PDEConfig2D | None = None,
    max_steps: int = 256,
    forward_curve: DiscountCurve | None = None,
) -> tuple[CheyetteModel, optimistix.Solution]:
    r"""Fit one-factor Cheyette :math:`(\kappa, \sigma, \beta)` through the PDE.

    **Single-curve only.** The Cheyette PDE engine has no discount/projection
    split (unlike the Hull-White and G2++ recipes), so ``curve`` both discounts
    and projects.  Passing a distinct ``forward_curve`` *raises* rather than
    being silently dropped — a dual-curve fit that quietly ignored the
    projection curve would be a wrong calibration with no signal.  Dual-curve
    Cheyette is tracked on the roadmap.

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
        solver: ``"bfgs"`` (default) or ``"gauss_newton"``.
            ``"levenberg_marquardt"`` is deliberately not offered (the residual
            closes over a sequence of instrument pytrees, tripping
            ``optimistix`` 0.1.0's LM with a ``List arity mismatch``).
        engine: ``"frozen"`` (default; the fast frozen-``y`` production PDE) or
            ``"exact"`` (the bias-free exact-``y`` 2-D PDE — more expensive per
            solve, but removes the frozen-``y`` skew bias). Only one factor
            supports ``"exact"``.
        config: PDE grid configuration. Defaults to a moderate mesh matching the
            chosen engine (:class:`~valax.pricing.pde.PDEConfig` for ``frozen``,
            :class:`~valax.pricing.pde.PDEConfig2D` for ``exact``).
        max_steps: Maximum optimiser iterations.

    Returns:
        Tuple of the fitted model and the ``optimistix`` solution.

    Raises:
        ValueError: If ``solver`` or ``engine`` is not recognised, if a
            ``fixed_params`` key is not a Cheyette parameter, or if the number
            of quotes does not match the number of instruments.
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
            initial_curve=curve,
        )
    else:
        initial_guess = eqx.tree_at(
            lambda m: m.initial_curve, initial_guess, curve
        )

    transforms: dict[str, TransformSpec] = dict(CHEYETTE_TRANSFORMS)
    if fixed_params is not None:
        for name, value in fixed_params.items():
            if name not in CHEYETTE_TRANSFORMS:
                raise ValueError(f"Unknown Cheyette parameter: {name!r}")
            initial_guess = eqx.tree_at(
                lambda m, n=name: getattr(m, n),
                initial_guess,
                jnp.asarray(value),
            )
            del transforms[name]

    if weights is None:
        weights = jnp.ones_like(market_prices)
    scale = (
        jnp.maximum(jnp.abs(market_prices), 1e-8)
        if relative
        else jnp.ones_like(market_prices)
    )

    y0 = model_to_unconstrained(initial_guess, transforms)
    args = (
        transforms,
        initial_guess,
        tuple(instruments),
        market_prices,
        weights,
        scale,
        engine,
        config,
    )
    sol = _run_solver(solver, y0, args, max_steps)
    fitted = unconstrained_to_model(sol.value, transforms, initial_guess)
    return fitted, sol


def calibrate_cheyette_2f(
    instruments: Sequence[Any],
    market_prices: Float[Array, " n_quotes"],
    curve: DiscountCurve,
    initial_guess: CheyetteModel2F | None = None,
    fixed_params: dict[str, Float[Array, ""]] | None = None,
    weights: Float[Array, " n_quotes"] | None = None,
    relative: bool = True,
    solver: str = "bfgs",
    config: PDEConfig2D | None = None,
    max_steps: int = 256,
    forward_curve: DiscountCurve | None = None,
) -> tuple[CheyetteModel2F, optimistix.Solution]:
    r"""Fit two-factor Cheyette through the frozen-``y`` 2-D ADI PDE.

    Two factors are always calibrated on the **frozen-``y``** engine — the
    exact-``y`` two-factor PDE is 5-D and not tractable in a calibration loop
    (see ``docs/research-ideas.md``).

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
        solver: ``"bfgs"`` (default) or ``"gauss_newton"``.
        config: 2-D PDE grid configuration. Defaults to a moderate Craig-Sneyd
            mesh.
        max_steps: Maximum optimiser iterations.

    Returns:
        Tuple of the fitted model and the ``optimistix`` solution.

    Raises:
        ValueError: If ``solver`` is not recognised, if a ``fixed_params`` key
            is not a Cheyette parameter, or if the number of quotes does not
            match the number of instruments.
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
            initial_curve=curve,
        )
    else:
        initial_guess = eqx.tree_at(
            lambda m: m.initial_curve, initial_guess, curve
        )

    transforms: dict[str, TransformSpec] = dict(CHEYETTE_2F_TRANSFORMS)
    if fixed_params is not None:
        for name, value in fixed_params.items():
            if name not in CHEYETTE_2F_TRANSFORMS:
                raise ValueError(f"Unknown Cheyette parameter: {name!r}")
            initial_guess = eqx.tree_at(
                lambda m, n=name: getattr(m, n),
                initial_guess,
                jnp.asarray(value),
            )
            del transforms[name]

    if weights is None:
        weights = jnp.ones_like(market_prices)
    scale = (
        jnp.maximum(jnp.abs(market_prices), 1e-8)
        if relative
        else jnp.ones_like(market_prices)
    )

    y0 = model_to_unconstrained(initial_guess, transforms)
    args = (
        transforms,
        initial_guess,
        tuple(instruments),
        market_prices,
        weights,
        scale,
        "frozen",
        config,
    )
    sol = _run_solver(solver, y0, args, max_steps)
    fitted = unconstrained_to_model(sol.value, transforms, initial_guess)
    return fitted, sol
