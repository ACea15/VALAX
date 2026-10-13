r"""Model-to-basket price fits: the level behind every short-rate calibrator.

Hull-White, G2++ and Cheyette (one and two factors) are all calibrated the
same way: a model with a handful of parameters, exact-fitted to an initial
curve, is fitted by weighted least squares to the prices of a basket of
instruments,

.. math::

    r_k(\theta) = w_k\,\frac{P_k(M(\theta); P^d, P^f) - q_k}{s_k},
    \qquad s_k = \max(|q_k|, 10^{-8}) \text{ if relative else } 1,

priced by one engine (Jamshidian, the Brigo-Mercurio integral, a PDE). This
module is that level, once. The quotes ``q`` are the level's quotes; the
curves are its **upstream** input (so the adjoint reports the curve risk of
the fit); parameters held fixed (``fixed_params``) are its **fixed** inputs
(so their sensitivities are reported as :math:`\bar p`). Relative scaling is
computed from the quotes *inside* the residual, so the solved system and the
differentiated one are the same function.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.calibration.level import LevelInputs
from valax.calibration.result import CalibrationProvenance, CalibrationResult
from valax.calibration.settings import SolverSettings
from valax.calibration.solve import solve_level
from valax.calibration.transforms import (
    TransformSpec,
    model_to_unconstrained,
    unconstrained_to_model,
)


#: Default settings of a short-rate basket fit: line-searched BFGS to 1e-12.
#: BFGS, not Levenberg-Marquardt, because the closed-form swaption engines
#: nest a root-find (Jamshidian's critical rate) whose own implicit
#: derivative is non-finite far from the solution, where an undamped LM step
#: can land; BFGS's line search keeps the iterates where the pricer is
#: defined. Tight tolerances because these engines have no noise floor.
BASKET_SETTINGS = SolverSettings(
    method="bfgs", rtol=1e-12, atol=1e-12, max_steps=256
)


def basket_fit_level(
    name: str,
    *,
    quotes: Float[Array, " n"],
    upstream: Any,
    initial_guess: eqx.Module,
    transforms: Mapping[str, TransformSpec],
    prices_fn: Callable[[Any, LevelInputs], Float[Array, " n"]],
    bind_fn: Callable[[Any, LevelInputs], Any] | None = None,
    fixed_params: Mapping[str, Any] | None = None,
    weights: Float[Array, " n"] | None = None,
    relative: bool = True,
    settings: SolverSettings | None = None,
    provenance: CalibrationProvenance | None = None,
    model_label: str = "model",
    data: Any = None,
) -> CalibrationResult:
    """Fit a model's parameters to a basket of prices as one level.

    Args:
        name: Level name used in diagnostics.
        quotes: Target prices, one per basket instrument.
        upstream: Whatever ``prices_fn`` / ``bind_fn`` read from
            ``inputs.upstream`` (e.g. ``(discount_curve, forward_curve)``).
        initial_guess: Starting model; also supplies every field that is
            neither calibrated nor fixed.
        transforms: ``field -> TransformSpec`` for every calibratable field.
        prices_fn: ``(bound_model, inputs) -> prices``.
        bind_fn: ``(model, inputs) -> bound_model`` — e.g. attach the
            upstream curve for pricing. Identity when ``None``. The
            *returned* model is the unbound one: the curve is market data
            the model is bound to at pricing time, not a calibrated output.
        fixed_params: ``field -> value`` held fixed; read from
            ``inputs.fixed`` so their sensitivities are reported.
        weights: Per-quote weights; default ones.
        relative: Divide each residual by ``max(|quote|, 1e-8)``.
        settings: Solver settings; defaults to :data:`BASKET_SETTINGS`.
        provenance: The pricing engine of ``prices_fn``.
        model_label: Model name used in validation errors.
        data: Trade data ``prices_fn`` reads from ``inputs.data`` (e.g. the
            basket instruments) instead of closing over it — needed when the
            basket may be traced, so the returned result holds no stale
            tracer (see :class:`~valax.calibration.level.LevelInputs`).

    Returns:
        The :class:`~valax.calibration.result.CalibrationResult`; the fitted
        model is ``result.value`` — as assembled from ``initial_guess``,
        *not* passed through ``bind_fn`` (curve-free for the short-rate
        models, so it prices through ``valax.price`` and stacks under
        ``vmap``).

    Raises:
        ValueError: If a fixed parameter is not calibratable.
    """
    settings = BASKET_SETTINGS if settings is None else settings
    fixed_params = dict(fixed_params or {})
    free = dict(transforms)
    for f in fixed_params:
        if f not in transforms:
            raise ValueError(f"Unknown {model_label} parameter: {f!r}")
        del free[f]
    quotes = jnp.asarray(quotes, dtype=jnp.float64)
    fixed = {f: jnp.asarray(v, dtype=jnp.float64) for f, v in fixed_params.items()}
    fixed["weights"] = (
        jnp.ones_like(quotes) if weights is None
        else jnp.asarray(weights, dtype=jnp.float64)
    )
    inputs = LevelInputs(quotes=quotes, upstream=upstream, fixed=fixed, data=data)
    bind = (lambda m, _inp: m) if bind_fn is None else bind_fn

    def assemble(theta: dict, inp: LevelInputs) -> Any:
        model = initial_guess
        for f in fixed_params:
            model = eqx.tree_at(
                lambda m, _f=f: getattr(m, _f), model, inp.fixed[f]
            )
        return unconstrained_to_model(theta, free, model)

    def residual(theta: dict, inp: LevelInputs) -> Float[Array, " n"]:
        q = inp.quotes
        scale = jnp.maximum(jnp.abs(q), 1e-8) if relative else jnp.ones_like(q)
        prices = prices_fn(bind(assemble(theta, inp), inp), inp)
        return inp.fixed["weights"] * (prices - q) / scale

    theta0 = jax.lax.stop_gradient(model_to_unconstrained(initial_guess, free))
    return solve_level(
        name, "least_squares", residual, theta0, inputs, assemble, settings,
        provenance=provenance,
    )


__all__ = ["BASKET_SETTINGS", "basket_fit_level"]
