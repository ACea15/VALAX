"""Theta-scheme backward time-stepping for 1-D solvers.

Implements the generic theta-scheme

.. math::

    (I - \\theta\\,\\Delta t\\,\\mathcal{L})\\,V^{n}
        = (I + (1-\\theta)\\,\\Delta t\\,\\mathcal{L})\\,V^{n+1} + \\text{BC},

with ``theta = 1/2`` recovering Crank-Nicolson and ``theta = 1`` fully-implicit
backward Euler. **Rannacher start-up** runs the first ``rannacher_steps`` steps
(nearest expiry) fully-implicit to damp oscillations from non-smooth terminal
data, then switches to the requested ``theta``.

The backward march is a single ``jax.lax.scan``. An optional ``solver_fn`` seam
replaces the plain tridiagonal solve — this is how the American penalty method
(:mod:`valax.pricing.pde.exercise`) plugs in.

**Time-dependent operators.** The ``operator`` argument may carry either 1-D
bands (length ``n``, constant in time — the original behaviour) or 2-D *stacked*
bands (shape ``(n_time, n)``, one row per backward step). The stacked form lets
the spatial operator vary from step to step, which is what local-volatility
(Dupire) pricing needs: the diffusion coefficient ``sigma^2_loc(x, tau)`` is
rebuilt at every time level. Row ``m`` of the stack is used at backward step
``m`` (nearest expiry first). See
:func:`~valax.pricing.pde.coefficients.lv_operator_stack`.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
from jaxtyping import Float
from jax import Array

from valax.pricing.pde.boundary import Boundary1D
from valax.pricing.pde.linalg import tridiagonal_matvec, tridiagonal_solve
from valax.pricing.pde.operators import Operator1D

StepSolver = Callable[
    [Float[Array, " n_minus_1"], Float[Array, " n"], Float[Array, " n_minus_1"], Float[Array, " n"]],
    Float[Array, " n"],
]


def solve_backward_1d(
    operator: Operator1D,
    boundary: Boundary1D,
    terminal: Float[Array, " n"],
    *,
    expiry: Float[Array, ""],
    n_time: int,
    theta: float,
    rannacher_steps: int = 0,
    solver_fn: Optional[StepSolver] = None,
) -> Float[Array, " n"]:
    """Backward time-march the terminal condition to ``t = 0``.

    Args:
        operator: Per-row spatial operator coefficients ``L``. Bands may be
            1-D (length ``n``, constant in time) or 2-D *stacked* (shape
            ``(n_time, n)``, one row per backward step — row ``m`` used at
            step ``m``, nearest expiry first) for time-dependent operators
            such as local volatility.
        boundary: Dirichlet boundary values as functions of time-remaining.
        terminal: Terminal (payoff) values on the grid.
        expiry: Time to expiry ``T`` (``dt = T / n_time``).
        n_time: Number of backward time steps.
        theta: Implicitness parameter after Rannacher start-up.
        rannacher_steps: Number of leading fully-implicit steps.
        solver_fn: Optional per-step linear solver (defaults to the plain
            tridiagonal solve). Used to inject the American penalty method.

    Returns:
        The solution field on the grid at ``t = 0``.
    """
    dt = expiry / n_time
    solve = solver_fn if solver_fn is not None else tridiagonal_solve

    a_lower = dt * operator.lower
    a_diag = dt * operator.diag
    a_upper = dt * operator.upper

    # Whether the operator carries per-step bands (shape (n_time, n)) rather
    # than a single set of bands (shape (n,)). Resolved at trace time from the
    # static rank, so the branch inside ``step`` is a Python conditional and
    # never becomes dynamic control flow.
    stacked = a_lower.ndim == 2

    def step(v, inputs):
        if stacked:
            m, theta_m, al, ad, au = inputs
        else:
            m, theta_m = inputs
            al, ad, au = a_lower, a_diag, a_upper
        tau_new = (n_time - m) * dt        # time-remaining at the known level
        tau_old = (n_time - m - 1) * dt    # time-remaining at the solved level

        expl = 1.0 - theta_m
        # RHS = (I + (1-theta) A) v
        rhs = tridiagonal_matvec(
            expl * al[1:],
            1.0 + expl * ad,
            expl * au[:-1],
            v,
        )
        # Boundary contributions (explicit at level n+1, implicit at level n).
        bc_lo_new = boundary.lower_fn(tau_new)
        bc_lo_old = boundary.lower_fn(tau_old)
        bc_hi_new = boundary.upper_fn(tau_new)
        bc_hi_old = boundary.upper_fn(tau_old)
        rhs = rhs.at[0].add(expl * al[0] * bc_lo_new + theta_m * al[0] * bc_lo_old)
        rhs = rhs.at[-1].add(expl * au[-1] * bc_hi_new + theta_m * au[-1] * bc_hi_old)

        # LHS = (I - theta A)
        lhs_lower = -theta_m * al[1:]
        lhs_diag = 1.0 - theta_m * ad
        lhs_upper = -theta_m * au[:-1]

        v_new = solve(lhs_lower, lhs_diag, lhs_upper, rhs)
        return v_new, None

    steps = jnp.arange(n_time)
    thetas = jnp.where(steps < rannacher_steps, 1.0, theta)
    if stacked:
        xs = (steps, thetas, a_lower, a_diag, a_upper)
    else:
        xs = (steps, thetas)
    v_final, _ = jax.lax.scan(step, terminal, xs)
    return v_final


def theta_for_scheme(scheme) -> float:
    """Map a :class:`~valax.pricing.pde.config.Scheme` to its theta value.

    Crank-Nicolson -> 0.5; fully-implicit -> 1.0. (2-D ADI schemes are handled
    by the ADI stepper in PR-2, not here.)
    """
    from valax.pricing.pde.config import Scheme

    if scheme == Scheme.IMPLICIT:
        return 1.0
    return 0.5
