r"""One calibration level, ``R(theta; inputs) = 0``, and its adjoint.

Every calibration in VALAX — a curve bootstrap, a smile fit, a model fit to a
basket — is written in one shape: a residual ``R`` driven to zero over the
unknowns ``theta``, with everything else held as ``inputs``. The implicit
function theorem then differentiates the level without differentiating the
optimiser:

.. math::

    \left(\frac{\partial R}{\partial\theta}\right)^{\!T}\lambda = \bar\theta,
    \qquad
    \overline{\text{inputs}} = -\left(\frac{\partial R}{\partial\,\text{inputs}}\right)^{\!T}\lambda.

When the inputs are packed as :class:`LevelInputs` — the level's own quotes,
the upstream calibrated market and the held-fixed inputs — the one
vector-Jacobian product above returns all three adjoints of the audit chain at
once: quote sensitivities :math:`\bar q`, the cotangent added to upstream
levels, and fixed-input sensitivities :math:`\bar p`. That is what lets a chain
of arbitrary levels be differentiated by one generic loop
(:func:`valax.greeks.quote_risk.chain_adjoint`).

Over-determined fits enter as their first-order condition
:math:`\nabla_\theta \tfrac12\|r\|^2 = 0`, which is square; ``jax.grad`` of
that gradient is the *full* Hessian :math:`J^TWJ + \sum_k r_k\nabla^2 r_k`,
the documented choice (see ``docs/theory/calibration-chains.md``).
"""

from __future__ import annotations

import math
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import lineax as lx
from jax.flatten_util import ravel_pytree
from jaxtyping import Float
from jax import Array


class LevelInputs(eqx.Module):
    """The three input blocks of a calibration level.

    Splitting the inputs this way is what makes a single VJP of the residual
    produce the three adjoints the chain needs, without any per-level code.

    Attributes:
        quotes: This level's market quotes, usually a
            :class:`~valax.market.quotes.QuoteSet` restricted to the keys the
            level consumes.
        upstream: The calibrated market built by the levels before this one
            (a :class:`~valax.core.market_state.MarketState`), or ``None`` for
            a level with no upstream dependency.
        fixed: Inputs held fixed by the calibration (SABR beta, mean
            reversion, convexity adjustments, …), usually a
            :class:`~valax.market.quotes.QuoteSet`, or ``None``.
    """

    quotes: Any
    upstream: Any = None
    fixed: Any = None


class CalibrationLevel(eqx.Module):
    """One calibration written as ``R(theta; inputs) = 0``.

    The uniform shape every VALAX calibration is expressed in so that a
    single adjoint routine serves curve bootstraps, volatility stripping
    and model fits alike. Holding the residual, the solved unknowns and
    the inputs as three explicit members is what makes the partials of
    ``R`` evaluable at the solution.

    Attributes:
        residual_fn: Callable ``(theta, inputs) -> residual`` returning
            the residual vector. Must be the *same* code path the solver
            drove to zero — a re-implementation for the adjoint would
            silently decouple the two.
        theta: The solved unknowns, at the converged solution.
        inputs: Everything ``residual_fn`` treats as fixed: market
            quotes, upstream calibrated objects, and non-solved
            parameters. Prefer :class:`LevelInputs`.
    """

    residual_fn: Callable[[Any, Any], Float[Array, " n_residuals"]] = eqx.field(
        static=True
    )
    theta: Any
    inputs: Any


class LevelAdjoint(eqx.Module):
    """Result of one implicit-function-theorem step.

    Attributes:
        lam: The adjoint multiplier solving
            ``(dR/dtheta)^T lam = theta_bar``. One entry per residual.
        inputs_bar: ``-(dR/dinputs)^T lam`` — the sensitivity of the
            price to this level's inputs, shaped like
            :attr:`CalibrationLevel.inputs` (``None`` at non-differentiable
            leaves such as integer dates).
        residual_norm: ``max|R|`` at ``theta``. A large value means the
            level was not converged tightly enough for the theorem to
            apply, and every sensitivity downstream of it is suspect.
    """

    lam: Float[Array, " n_residuals"]
    inputs_bar: Any
    residual_norm: Float[Array, ""]


def level_adjoint(
    level: CalibrationLevel,
    theta_bar: Any,
    *,
    solver: lx.AbstractLinearSolver | None = None,
) -> LevelAdjoint:
    """Differentiate one calibration level by the implicit function theorem.

    Solves ``(dR/dtheta)^T lam = theta_bar`` and contracts the result
    against ``dR/dinputs``. Both Jacobian contractions are
    vector-Jacobian products: no Jacobian of the inputs is formed, so the
    cost is independent of the number of inputs. Inputs may be any pytree,
    including a whole ``MarketState`` with integer leaves; those leaves
    receive ``None``.

    Args:
        level: The converged calibration level.
        theta_bar: Cotangent of the price w.r.t. ``level.theta``, same
            pytree structure as ``level.theta``.
        solver: Linear solver for the transposed system. Defaults to
            ``lineax.AutoLinearSolver(well_posed=True)``.

    Returns:
        A :class:`LevelAdjoint` carrying ``lam``, the input sensitivities
        and the residual norm at the solution.

    Examples:
        >>> lvl = CalibrationLevel(residual_fn=r, theta=th, inputs=q)
        >>> adj = level_adjoint(lvl, theta_bar)
        >>> adj.inputs_bar.shape == q.shape
        True
    """
    if solver is None:
        solver = lx.AutoLinearSolver(well_posed=True)

    inputs = level.inputs

    def residual_of_theta(theta: Any) -> Float[Array, " n_residuals"]:
        return level.residual_fn(theta, inputs)

    residual_value, vjp_theta = jax.vjp(residual_of_theta, level.theta)

    # (dR/dtheta)^T as a linear operator: lam |-> vjp(lam).  Solving
    # against it yields lam without ever transposing a stored matrix.
    operator = lx.FunctionLinearOperator(
        lambda lam: vjp_theta(lam)[0],
        jax.eval_shape(lambda: residual_value),
    )
    lam = lx.linear_solve(operator, theta_bar, solver).value

    def residual_of_inputs(inp: Any) -> Float[Array, " n_residuals"]:
        return level.residual_fn(level.theta, inp)

    _, vjp_inputs = eqx.filter_vjp(residual_of_inputs, inputs)
    (inputs_bar,) = vjp_inputs(lam)

    residual_flat, _ = ravel_pytree(residual_value)
    return LevelAdjoint(
        lam=lam,
        inputs_bar=jax.tree.map(lambda x: -x, inputs_bar),
        residual_norm=jnp.max(jnp.abs(residual_flat)),
    )


# ── Level-generic conditioning diagnostics ───────────────────────────


#: Condition number of ``dR/dtheta`` above which a calibration level is
#: reported as ill-conditioned.
#:
#: The implicit function theorem needs ``dR/dtheta`` to be invertible, and
#: *how* invertible governs how much of the double-precision mantissa
#: survives the transposed solve: a level at condition number
#: :math:`\kappa` retains on the order of ``16 - log10(kappa)``
#: significant digits. At ``1e10`` that is six digits — prices are still
#: fine, and so is a 1 bp sensitivity, but the fifth decimal of a quote
#: delta is noise. Past this point the right response is to fix the
#: *calibration* (drop a redundant quote, re-space pillars, add a
#: regularising weight) rather than to tighten the solver, because the
#: ill-conditioning is a property of the chosen instrument set and no
#: amount of Newton iteration removes it.
#:
#: Single source of truth for the threshold: every alert and every
#: docstring that mentions one refers here rather than restating the
#: number.
ILL_CONDITIONED_THRESHOLD = 1e10


def level_condition_number(level: CalibrationLevel) -> Float[Array, ""]:
    """2-norm condition number of ``dR/dtheta`` at a level's solution.

    The level-generic counterpart of
    :attr:`~valax.greeks.adjoint.CurveAdjoint.jacobian_condition_number`, so
    curve bootstraps, volatility fits and model calibrations can all be
    reported on the same scale with the same threshold.

    Unlike :func:`level_adjoint`, this **does** materialise
    ``dR/dtheta`` — a condition number is a statement about the whole
    matrix, so there is no vector-Jacobian shortcut. It is therefore a
    per-snapshot diagnostic, not something to call per trade.

    Accepts a ``theta`` of any pytree shape by flattening before
    differentiating, so the reported number is the conditioning of the
    system actually solved and no caller has to pre-flatten.

    Rectangular levels are allowed and meaningful. A square level is the
    one the implicit function theorem applies to, and there the condition
    number governs the transposed solve. A least-squares level's
    ``dR/dtheta`` is a design matrix, and its condition number governs
    the normal equations the fit solves instead.

    Args:
        level: A converged calibration level. ``level.residual_fn`` must
            return a flat residual vector.

    Returns:
        The ratio of largest to smallest singular value of
        ``dR/dtheta``, evaluated at ``level.theta``. A singular level
        returns a non-finite value, which :func:`condition_alert`
        reports as such.

    Examples:
        >>> kappa = level_condition_number(result.level)
        >>> condition_alert(kappa, level_name="OIS curve") is None
        True
    """
    theta_flat, unravel = ravel_pytree(level.theta)
    inputs = level.inputs

    def residual_of_flat(
        flat: Float[Array, " n_theta"],
    ) -> Float[Array, " n_residuals"]:
        return level.residual_fn(unravel(flat), inputs)

    jacobian = jax.jacrev(residual_of_flat)(theta_flat)
    return jnp.linalg.cond(jacobian)


def condition_alert(
    condition_number: Float[Array, ""],
    *,
    level_name: str = "calibration level",
    threshold: float = ILL_CONDITIONED_THRESHOLD,
) -> str | None:
    """Alert text for an ill-conditioned level, or ``None`` if healthy.

    The reporting half of the conditioning story: a condition number that
    nothing ever looks at is not a control. Returns a message rather than
    warning or raising so the caller decides the channel (log line, risk
    report row, build failure) — and so this stays a pure function.

    A non-finite condition number always alerts: that is a singular
    ``dR/dtheta``, where the implicit function theorem does not apply at
    all and every sensitivity downstream is meaningless rather than
    merely imprecise.

    Args:
        condition_number: Condition number of ``dR/dtheta``, e.g. from
            :func:`level_condition_number`.
        level_name: Human-readable name of the level, interpolated into
            the message so a multi-level report says *which* level.
        threshold: Alert threshold. Defaults to
            :data:`ILL_CONDITIONED_THRESHOLD`.

    Returns:
        ``None`` when ``condition_number <= threshold``; otherwise a
        one-line message naming the level, the condition number and the
        approximate number of significant digits lost.

    Examples:
        >>> condition_alert(1e12, level_name="3M projection curve")
        "3M projection curve is ill-conditioned: ..."
    """
    kappa = float(condition_number)
    if jnp.isfinite(kappa) and kappa <= threshold:
        return None
    if not jnp.isfinite(kappa):
        return (
            f"{level_name}: dR/dtheta is singular (condition number "
            f"{kappa}). The implicit function theorem does not apply; "
            f"quote sensitivities from this level are meaningless, not "
            f"merely imprecise. Check for a redundant or duplicated "
            f"calibration instrument."
        )
    digits_lost = math.log10(kappa)
    return (
        f"{level_name} is ill-conditioned: condition number {kappa:.3e} "
        f"exceeds the {threshold:.0e} threshold, so roughly "
        f"{digits_lost:.0f} of 16 significant digits are lost in the "
        f"transposed solve. Prefer re-specifying the calibration "
        f"(redundant quote, pillar spacing, regularisation) over "
        f"tightening the solver."
    )


__all__ = [
    "CalibrationLevel",
    "ILL_CONDITIONED_THRESHOLD",
    "LevelAdjoint",
    "LevelInputs",
    "condition_alert",
    "level_adjoint",
    "level_condition_number",
]
