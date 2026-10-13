"""SABR model calibration to a volatility smile."""

import functools
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.calibration.level import LevelInputs
from valax.calibration.result import CalibrationResult
from valax.calibration.settings import FIT_SETTINGS, SolverSettings
from valax.calibration.solve import (
    solve_batched_level,
    solve_level,
    stack_slices,
)
from valax.calibration.transforms import (
    SABR_TRANSFORMS,
    correlation,
    model_to_unconstrained,
    positive,
    unconstrained_to_model,
)
from valax.models.sabr import SABRModel
from valax.pricing.analytic.sabr import sabr_implied_vol, sabr_normal_implied_vol


#: Transforms of a fixed-beta SABR fit (beta is then a held-fixed input).
SABR_FIXED_BETA_TRANSFORMS = {
    "alpha": positive(),
    "rho": correlation(),
    "nu": positive(),
}


def _default_sabr_guess(
    forward: Float[Array, ""],
    market_vols: Float[Array, " n"],
    beta: Float[Array, ""],
    is_normal: bool = False,
) -> SABRModel:
    """Heuristic initial guess for SABR calibration.

    The alpha seed inverts the leading-order ATM backbone in the relevant
    quoting convention: lognormal ``sigma_B ~ alpha * F^(beta-1)`` gives
    ``alpha ~ sigma_B * F^(1-beta)``, whereas normal ``sigma_N ~ alpha * F^beta``
    gives ``alpha ~ sigma_N * F^(-beta)``.
    """
    atm_vol = jnp.median(market_vols)
    if is_normal:
        alpha = atm_vol * forward ** (-beta)
    else:
        alpha = atm_vol * forward ** (1.0 - beta)
    return SABRModel(
        alpha=alpha,
        beta=beta,
        rho=jnp.array(-0.2),
        nu=jnp.array(0.3),
    )


def sabr_slice_inputs(
    strikes: Float[Array, " n"],
    market_vols: Float[Array, " n"],
    forward: Float[Array, ""],
    expiry: Float[Array, ""],
    *,
    fixed_beta: Float[Array, ""] | None = None,
    weights: Float[Array, " n"] | None = None,
) -> LevelInputs:
    """Pack one smile's data as the inputs of a SABR level.

    The quotes are the market vols; the forward, expiry, strikes, weights
    and (if fixed) beta are held-fixed inputs, so the level adjoint reports
    the forward sensitivity and the beta sensitivity alongside vega.

    Args:
        strikes: Option strikes.
        market_vols: Quoted implied vols at ``strikes``.
        forward: Forward of the underlying.
        expiry: Time to expiry.
        fixed_beta: Held-fixed beta, or ``None`` when beta is calibrated.
        weights: Per-strike residual weights; defaults to ones.

    Returns:
        The :class:`~valax.calibration.level.LevelInputs`.
    """
    strikes = jnp.asarray(strikes, dtype=jnp.float64)
    fixed = {
        "forward": jnp.asarray(forward, dtype=jnp.float64),
        "expiry": jnp.asarray(expiry, dtype=jnp.float64),
        "strikes": strikes,
        "weights": jnp.ones_like(strikes) if weights is None
        else jnp.asarray(weights, dtype=jnp.float64),
    }
    if fixed_beta is not None:
        fixed["beta"] = jnp.asarray(fixed_beta, dtype=jnp.float64)
    return LevelInputs(
        quotes=jnp.asarray(market_vols, dtype=jnp.float64), fixed=fixed
    )


def sabr_slice_residual_fn(
    vol_fn: Callable = sabr_implied_vol,
    *,
    fit_beta: bool = False,
) -> tuple[Callable, Callable]:
    """Residual and assemble maps of one SABR smile fit.

    Args:
        vol_fn: SABR implied-vol expansion ``(model, F, K, T) -> vol``.
        fit_beta: Whether beta is an unknown (else read from
            ``inputs.fixed["beta"]``).

    Returns:
        ``(residual_fn, assemble_fn)`` over unconstrained parameters and
        :func:`sabr_slice_inputs`-shaped inputs.
    """
    transforms = dict(SABR_TRANSFORMS) if fit_beta else SABR_FIXED_BETA_TRANSFORMS

    def assemble(theta: dict, inputs: LevelInputs) -> SABRModel:
        beta = jnp.asarray(0.5) if fit_beta else inputs.fixed["beta"]
        template = SABRModel(
            alpha=jnp.asarray(0.0), beta=beta, rho=jnp.asarray(0.0),
            nu=jnp.asarray(0.0),
        )
        return unconstrained_to_model(theta, transforms, template)

    def residual(theta: dict, inputs: LevelInputs) -> Float[Array, " n"]:
        model = assemble(theta, inputs)
        fx = inputs.fixed
        model_vols = jax.vmap(
            lambda k: vol_fn(model, fx["forward"], k, fx["expiry"])
        )(fx["strikes"])
        return fx["weights"] * (model_vols - inputs.quotes)

    return residual, assemble


def calibrate_sabr(
    strikes: Float[Array, " n"],
    market_vols: Float[Array, " n"],
    forward: Float[Array, ""],
    expiry: Float[Array, ""],
    *,
    initial_guess: SABRModel | None = None,
    fixed_beta: Float[Array, ""] | float | None = 0.5,
    weights: Float[Array, " n"] | None = None,
    settings: SolverSettings | None = None,
    vol_fn: Callable = sabr_implied_vol,
    is_normal: bool = False,
    name: str = "sabr",
) -> CalibrationResult:
    """Calibrate SABR parameters to a volatility smile.

    A weighted least-squares level (see
    :func:`~valax.calibration.solve.solve_level`): the residual is
    ``w * (vol_fn(model, F, K, T) - market_vol)`` in unconstrained
    parameters, and non-convergence is *reported*
    (``result.diagnostics.converged``; ``result.check()`` raises) rather
    than raised mid-trace.

    Args:
        strikes: Option strikes.
        market_vols: Observed implied volatilities at each strike.
        forward: Forward price.
        expiry: Time to expiry.
        initial_guess: Starting SABRModel. If None, uses a heuristic.
        fixed_beta: Beta held fixed (a fixed input of the level) — **0.5 by
            default**, the same default as the chain nodes
            (``sabr_smile_node``, ``swaption_cube_node``,
            ``optionlet_sabr_node``): beta and alpha are poorly jointly
            identified from one smile (freeing beta costs ~260x in
            conditioning), so market practice is to fix it. ``0.0`` for
            normal-backbone rates smiles; ``None`` calibrates beta too.
        weights: Per-strike weights for residuals. Default: uniform.
        settings: Solver settings; defaults to
            :data:`~valax.calibration.settings.FIT_SETTINGS`
            (Levenberg-Marquardt). Any ``"least_squares"`` method is
            accepted.
        vol_fn: SABR implied-vol function used to build residuals. Defaults to
            the lognormal :func:`sabr_implied_vol`; pass
            :func:`sabr_normal_implied_vol` (optionally pre-bound with a shift
            via ``functools.partial``) to calibrate against normal quotes.
        is_normal: Whether ``vol_fn`` is a normal-vol expansion. Only affects
            the default initial-guess backbone (see :func:`_default_sabr_guess`).
        name: Level name used in diagnostics.

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the fitted
        :class:`~valax.models.sabr.SABRModel` is ``result.value``.

    Raises:
        ValueError: If ``settings.method`` is not a least-squares method.
    """
    settings = FIT_SETTINGS if settings is None else settings
    beta = jnp.asarray(0.5 if fixed_beta is None else fixed_beta)
    inputs = sabr_slice_inputs(
        strikes, market_vols, forward, expiry,
        fixed_beta=fixed_beta, weights=weights,
    )
    if initial_guess is None:
        initial_guess = _default_sabr_guess(
            inputs.fixed["forward"], inputs.quotes, beta, is_normal=is_normal
        )
    if fixed_beta is not None:
        initial_guess = eqx.tree_at(lambda m: m.beta, initial_guess, beta)

    fit_beta = fixed_beta is None
    transforms = dict(SABR_TRANSFORMS) if fit_beta else SABR_FIXED_BETA_TRANSFORMS
    residual, assemble = sabr_slice_residual_fn(vol_fn, fit_beta=fit_beta)
    # The guess only seeds the solver; keep it out of the gradient.
    theta0 = jax.lax.stop_gradient(
        model_to_unconstrained(initial_guess, transforms)
    )
    return solve_level(
        name, "least_squares", residual, theta0, inputs, assemble, settings
    )


def sabr_vol_fn(is_normal: bool = False, shift: Float[Array, ""] = 0.0) -> Callable:
    """The SABR implied-vol expansion of a quoting convention.

    Args:
        is_normal: Normal (Bachelier) quotes if True, else lognormal
            (Black-76).
        shift: Displacement of the normal expansion (ignored when lognormal).

    Returns:
        ``(model, forward, strike, expiry) -> vol``.
    """
    if is_normal:
        return functools.partial(sabr_normal_implied_vol, shift=jnp.asarray(shift))
    return sabr_implied_vol


def sabr_batch_level(
    name: str,
    inputs: LevelInputs,
    node_inputs_fn: Callable[[LevelInputs], LevelInputs],
    value_fn: Callable[[SABRModel, LevelInputs, LevelInputs], Any],
    *,
    labels: Sequence[str],
    fit_beta: bool = False,
    vol_fn: Callable = sabr_implied_vol,
    is_normal: bool = False,
    settings: SolverSettings | None = None,
) -> CalibrationResult:
    """One SABR smile per node, solved as one batched level.

    The engine behind every SABR slice / cube loop (expiry slices of a
    :class:`~valax.surfaces.sabr_surface.SABRVolSurface`, optionlet expiries,
    swaption-cube cells) and the rates-vol chain nodes. Each node is fitted
    independently (its own convergence flag, reported by ``labels``) inside
    one ``vmap``; the level residual stacks the per-node first-order
    conditions, so the IFT adjoint covers the whole grid at once.

    Args:
        name: Level name used in diagnostics.
        inputs: The level inputs (quotes, upstream, fixed) as stored on the
            result; the adjoint differentiates with respect to these.
        node_inputs_fn: ``inputs -> LevelInputs`` with a leading node axis:
            ``quotes`` ``(n, m)`` vols and ``fixed`` holding ``forward``
            ``(n,)``, ``expiry`` ``(n,)``, ``strikes`` ``(n, m)``, ``weights``
            ``(n, m)`` and, unless ``fit_beta``, ``beta`` ``(n,)``. Forwards
            computed here from ``inputs.upstream`` stay differentiable.
        value_fn: ``(models, node_inputs, inputs) -> calibrated object``,
            where ``models`` is a batched :class:`SABRModel` (fields ``(n,)``).
        labels: One name per node.
        fit_beta: Calibrate beta per node (else read from the node inputs).
        vol_fn: SABR implied-vol expansion (see :func:`sabr_vol_fn`).
        is_normal: Whether ``vol_fn`` is a normal expansion (initial-guess
            backbone only).
        settings: Per-node solver settings; defaults to
            :data:`~valax.calibration.settings.FIT_SETTINGS`.

    Returns:
        The batched :class:`~valax.calibration.result.CalibrationResult`.
    """
    settings = FIT_SETTINGS if settings is None else settings
    transforms = dict(SABR_TRANSFORMS) if fit_beta else SABR_FIXED_BETA_TRANSFORMS
    residual, assemble_one = sabr_slice_residual_fn(vol_fn, fit_beta=fit_beta)

    def guess(node: LevelInputs) -> dict:
        beta = jnp.asarray(0.5) if fit_beta else node.fixed["beta"]
        model = _default_sabr_guess(
            node.fixed["forward"], node.quotes, beta, is_normal=is_normal
        )
        return model_to_unconstrained(model, transforms)

    theta0 = jax.lax.stop_gradient(jax.vmap(guess)(node_inputs_fn(inputs)))

    def assemble(theta: dict, inp: LevelInputs) -> Any:
        nodes = node_inputs_fn(inp)
        models = jax.vmap(assemble_one)(theta, nodes)
        return value_fn(models, nodes, inp)

    return solve_batched_level(
        name, "least_squares", residual, theta0, inputs, node_inputs_fn,
        assemble, settings, labels=labels,
    )


def stacked_sabr_inputs(
    strikes_per_node: Sequence[Float[Array, " n_k"]],
    market_vols_per_node: Sequence[Float[Array, " n_k"]],
    forwards: Float[Array, " n"],
    expiries: Float[Array, " n"],
    *,
    weights_per_node: Sequence[Float[Array, " n_k"]] | None = None,
    fixed_beta: Float[Array, ""] | None = None,
) -> LevelInputs:
    """Stack per-node smiles into the batched inputs of :func:`sabr_batch_level`.

    Ragged strike grids are padded with zero-weight entries
    (:func:`~valax.calibration.solve.stack_slices`).

    Args:
        strikes_per_node: One strike array per node.
        market_vols_per_node: One vol array per node.
        forwards: Forward per node.
        expiries: Expiry per node.
        weights_per_node: Optional weights per node.
        fixed_beta: Held-fixed beta (``None`` when beta is calibrated).

    Returns:
        Batched :class:`~valax.calibration.level.LevelInputs` (pass
        ``node_inputs_fn=lambda inp: inp``).
    """
    strikes, vols, weights = stack_slices(
        strikes_per_node, market_vols_per_node, weights_per_node
    )
    n = strikes.shape[0]
    fixed = {
        "forward": jnp.asarray(forwards, dtype=jnp.float64).reshape(n),
        "expiry": jnp.asarray(expiries, dtype=jnp.float64).reshape(n),
        "strikes": strikes,
        "weights": weights,
    }
    if fixed_beta is not None:
        fixed["beta"] = jnp.full((n,), jnp.asarray(fixed_beta, dtype=jnp.float64))
    return LevelInputs(quotes=vols, fixed=fixed)


__all__ = [
    "SABR_FIXED_BETA_TRANSFORMS",
    "calibrate_sabr",
    "sabr_batch_level",
    "sabr_slice_inputs",
    "sabr_slice_residual_fn",
    "sabr_vol_fn",
    "stacked_sabr_inputs",
]
