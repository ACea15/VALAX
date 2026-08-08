"""Early-exercise handling for the backward PDE sweep.

Two strategies, both differentiable so Greeks flow from ``jax.grad`` (per
``AGENTS.md``, never finite differences):

- :func:`penalty_solver` — the **penalty method** for continuous American
  exercise. It returns a drop-in replacement for :func:`tridiagonal_solve` that
  augments the per-step linear system with a forcing term pushing the solution
  above the exercise payoff. A *fixed* iteration count keeps the whole solve a
  static, differentiable computation graph.
- :func:`explicit_project` — explicit projection for discrete-date Bermudan /
  callable / puttable exercise (applied only at snapped event steps).
"""

from typing import Callable

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.pricing.pde.linalg import tridiagonal_solve

TridiagonalSolver = Callable[
    [Float[Array, " n_minus_1"], Float[Array, " n"], Float[Array, " n_minus_1"], Float[Array, " n"]],
    Float[Array, " n"],
]


def penalty_solver(
    payoff: Float[Array, " n"],
    rho: float,
    iters: int,
) -> TridiagonalSolver:
    """Return a tridiagonal solver that enforces ``V >= payoff`` (American).

    The penalty method solves, at each time step, a sequence of tridiagonal
    systems

    .. math::

        (A + \\rho\\, \\mathrm{diag}(m^k))\\, V^{k+1}
            = b + \\rho\\, m^k \\odot g,

    where ``g`` is the exercise payoff and ``m^k`` is the indicator that the
    current iterate violates ``V >= g``. As ``rho`` grows large the solution is
    pushed onto the free boundary. ``iters`` is fixed (static), so the loop
    unrolls into a differentiable graph.

    Args:
        payoff: Exercise (obstacle) value ``g`` at each grid node.
        rho: Penalty coefficient.
        iters: Fixed number of penalty iterations per time step.

    Returns:
        A callable ``(lower, diag, upper, rhs) -> V`` usable as the step solver.
    """
    rho_arr = jnp.asarray(rho)

    def solve(lower, diag, upper, rhs):
        v0 = tridiagonal_solve(lower, diag, upper, rhs)

        def body(v, _):
            binds = (v < payoff).astype(payoff.dtype)
            diag_pen = diag + rho_arr * binds
            rhs_pen = rhs + rho_arr * binds * payoff
            return tridiagonal_solve(lower, diag_pen, upper, rhs_pen), None

        v_final, _ = jax.lax.scan(body, v0, xs=None, length=iters)
        return v_final

    return solve


def explicit_project(
    values: Float[Array, " n"],
    exercise_value: Float[Array, " n"],
    *,
    is_min: bool,
) -> Float[Array, " n"]:
    """Project the continuation value against an exercise value.

    Args:
        values: Continuation values at the grid nodes.
        exercise_value: Immediate-exercise values at the grid nodes.
        is_min: If True, take the pointwise minimum (issuer-optimal, e.g. a
            callable bond); otherwise the maximum (holder-optimal, e.g. a
            puttable bond or American option).

    Returns:
        The projected values.
    """
    if is_min:
        return jnp.minimum(values, exercise_value)
    return jnp.maximum(values, exercise_value)
